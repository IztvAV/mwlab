import copy
import os
import random
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from common import WorkModel
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
from mwlab.nn.scalers import MinMaxScaler, StdScaler
from mwlab import TouchstoneDataset, TouchstoneLDataModule, TouchstoneDatasetAnalyzer, TouchstoneData

from filters import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers, SamplerTypes, MWFilter, CouplingMatrix
from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLModule, MWFilterBaseLMWithMetrics

from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift

import matplotlib.pyplot as plt
import lightning as L

from torch import nn
import models
import torch
from filters.mwfilter_optim.bfgs import optimize_cm
from losses import CustomLosses
import configs as cfg
import common

import numpy as np
from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample
import skrf as rf
import torch.nn.functional as F
import phase
from dataclasses import dataclass


work_model: WorkModel|None = None
inference_model = None
inference_model_ft = None
corr_model = None
corr_fast_calc = None
corr_optim = None
phase_extractor: phase.PhaseLoadingExtractor or None = None
calibration_res: dict or None = None
calibrated: bool = False


def _get_configs_from_manifest(manifest_path: str|os.PathLike|None=None) -> cfg.Configs:
    if manifest_path is None:
        print("Load default configs")
        configs = cfg.Configs.init_as_default("D:\\Burlakov\\pyprojects\\mwlab\\default.yml")
    else:
        print(f"Load configs from: {manifest_path}")
        configs = cfg.Configs(manifest_path)
    return configs


def train_model(manifest_path: str|os.PathLike|None=None):
    configs = _get_configs_from_manifest(manifest_path)
    global work_model
    work_model = common.WorkModel(configs, is_inference=False)
    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                    work_model.orig_filter.coupling_matrix.links] +
                                                                   ["Q"] + ["f0"] + ["bw"] + ["a11"] +
                                                                   ["a22"] + ["b11"] + ["b22"])
    codec = codec
    work_model.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )

    lit_model = work_model.train(
        optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
        scheduler_cfg={"name": "StepLR", "step_size": 25, "gamma": 0.09},
        loss_fn=CustomLosses("sqrt_mse_with_l1", weight_decay=1, weights=None),
        strategy_type="two stage"
    )

    return lit_model



def load_model(manifest_path: str|os.PathLike|None=None):
    """ Пока в таком формате. В дальнейшем надо будет стандартизировать названия моделей """
    global work_model
    configs = _get_configs_from_manifest(manifest_path)
    work_model = common.WorkModel(configs, is_inference=True)
    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                    work_model.orig_filter.coupling_matrix.links] +
                                                                   ["Q"] + ["f0"] + ["bw"] + ["a11"] +
                                                                   ["a22"] + ["b11"] + ["b22"])

    codec = codec
    work_model.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )

    global inference_model, phase_extractor, inference_model_ft
    inference_model = work_model.inference(configs.MODEL_CHECKPOINT_PATH)
    inference_model_ft = copy.deepcopy(inference_model)
    inference_model_ft.eval()
    for param in inference_model_ft.model.parameters():
        param.requires_grad = False

    for param in inference_model_ft.model._main_model.fc.parameters():
        param.requires_grad = True

    for param in inference_model_ft.model._correction_model.parameters():
        param.requires_grad = True
    params = [p for p in inference_model_ft.parameters() if p.requires_grad]
    global corr_optim
    corr_optim = torch.optim.AdamW(params=params, lr=0.0005370623202982373, weight_decay=1e-5)
    global corr_fast_calc
    corr_fast_calc = FastMN2toSParamCalculation(matrix_order=work_model.orig_filter.coupling_matrix.matrix_order,
                                                wlist=work_model.orig_filter.f_norm,
                                                Q=work_model.orig_filter.Q,
                                                fbw=work_model.orig_filter.fbw,
                                                device=inference_model_ft.device)

    phase_extractor = phase.PhaseLoadingExtractor(inference_model, work_model, work_model.orig_filter)


def phase_extract(fil: rf.Network):
    global phase_extractror, work_model, calibration_res, calibrated
    if not calibrated:
        print("Start calibration")
        w = work_model.orig_filter.f_norm
        calibration_res = phase_extractor.extract_all(fil, w_norm=w)
        ntw_de = calibration_res['ntw_deembedded']
        calibrated = True
    else:
        # TODO: Сделать коррекцию фазы одной функцией. Для этого нужно переработать класс коррекции фазы
        w = work_model.orig_filter.f_norm
        a11 = calibration_res['phi1_c']
        a22 = calibration_res['phi2_c']

        fil = phase.remove_phase_from_coeffs(fil, w, a11, 0, a22, 0, inverse_s12_s21=True)

        # phi1 = -2.0 * (a11 + 0 * np.asarray(w))
        # phi2 = -2.0 * (a22 + 0 * np.asarray(w))
        # fil.s[:, 0, 0] = phase.apply_phase_one(fil.s[:, 0, 0], phi1)
        # fil.s[:, 1, 1] = phase.apply_phase_one(fil.s[:, 1, 1], phi2)
        # fil.s[:, 0, 1] = -phase.apply_phase_one(fil.s[:, 0, 1], 0.5 * (phi1 + phi2))
        # fil.s[:, 1, 0] = -phase.apply_phase_one(fil.s[:, 1, 0], 0.5 * (phi1 + phi2))

        b11 = calibration_res['b11_opt']
        b22 = calibration_res['b22_opt']
        ntw_de = phase.remove_phase_from_coeffs(fil, w, 0, b11, 0, b22, inverse_s12_s21=False)
    return ntw_de


def prediction_with_optim_correct(fil: rf.Network):
    global inference_model, calibration_res
    w = work_model.orig_filter.f_norm
    s_resample = S_Resample(301)
    fil = s_resample(fil)
    # orig_fil_to_nn = copy.deepcopy(fil)
    ntw_de = phase_extract(fil)

    pred_prms = inference_model.predict_x(ntw_de)
    print(f"Предсказанные параметры: {pred_prms}")
    pred_fil = work_model.create_filter_from_prediction(fil, work_model.orig_filter, pred_prms)

    a11_final = calibration_res['phi1_c'] + pred_prms["a11"]
    a22_final = calibration_res['phi2_c'] + pred_prms["a22"]
    b11_final = pred_prms["b11"] + calibration_res['b11_opt']
    b22_final = pred_prms["b22"] + calibration_res['b22_opt']
    print(
        f"Финальный фазовый сдвиг, после корректировки ИИ: a11={a11_final:.6f} рад ({np.degrees(a11_final):.2f}°), a22={a22_final:.6f} рад ({np.degrees(a22_final):.2f}°)"
        f" b11 = {b11_final:.6f} рад ({np.degrees(b11_final):.2f}°), b22 = {b22_final:.6f} рад ({np.degrees(b22_final):.2f}°)")

    pred_fil = phase.remove_phase_from_coeffs(pred_fil, w, a11_final, b11_final, a22_final, b22_final, inverse_s12_s21=False)
    optim_filter, phase_opt = optimize_cm(pred_fil, fil, phase_init=(a11_final, a22_final, b11_final, b22_final), plot=False)
    print(f"Оптимизированные параметры: {optim_filter.coupling_matrix.factors}. Добротность: {optim_filter.Q}. Фаза: {np.degrees(phase_opt)}")

    optim_filter = phase.remove_phase_from_coeffs(optim_filter, w, *phase_opt, inverse_s12_s21=False)
    return optim_filter.s, optim_filter.coupling_matrix.matrix.numpy()


def prediction_with_online_correct(fil: rf.Network):
    def factors_from_preds(preds: dict, links: list, ref_tensor: torch.Tensor):
        """
        links: список пар (i, j) или объектов, из которых можно получить i,j
        preds: dict где есть ключи "m_i_j" (и/или "m_j_i")
        возвращает factors: torch.Tensor [n_links]
        """
        vals = []
        for (i, j) in links:
            k1 = f"m_{i}_{j}"
            k2 = f"m_{j}_{i}"  # на всякий случай
            if k1 in preds:
                v = preds[k1]
            elif k2 in preds:
                v = preds[k2]
            else:
                # если отсутствует — можно 0, но лучше явно падать, чтобы не молча портить матрицу
                v = torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=ref_tensor.device, dtype=ref_tensor.dtype)
            vals.append(v)
        return torch.stack(vals, dim=0)  # [n_links]

    global inference_model_ft, corr_optim, corr_fast_calc

    codec = work_model.codec

    loss = nn.L1Loss(reduction="sum")

    codec_db = copy.deepcopy(codec)
    codec_db.y_channels = ['S1_1.db', 'S2_1.db']
    codec_mag = copy.deepcopy(codec)
    codec_mag.y_channels = ['S1_1.mag', 'S2_1.mag']

    ntw_de = phase_extract(fil)
    pred_prms = inference_model_ft.predict_x(ntw_de)
    pred_fil = work_model.create_filter_from_prediction(fil, work_model.orig_filter, pred_prms)

    keys = pred_prms.keys()
    m_keys = list(keys)[:-7]

    ts = TouchstoneData(ntw_de)
    s = codec.encode(ts)[1].unsqueeze(0)
    s_db = codec_db.encode(ts)[1].unsqueeze(0)
    s_mag = codec_mag.encode(ts)[1].unsqueeze(0)

    print(f"Initial loss: {loss(s_db, codec_db.encode(TouchstoneData(pred_fil))[1].unsqueeze(0))}")

    s = s.to(inference_model_ft.device)
    s_db = s_db.to(inference_model_ft.device)
    s_mag = s_mag.to(inference_model_ft.device)

    start_time = time.time()
    w = work_model.orig_filter.f_norm.to(inference_model_ft.device)
    for j in range(50):
        x_pred = inference_model_ft(s)
        # print("x_pred grad:", x_pred.requires_grad)
        if inference_model_ft.scaler_out is not None:
            x_pred = inference_model_ft._apply_inverse(inference_model_ft.scaler_out, x_pred)
            # print("x_pred inverse grad:", x_pred.requires_grad)
        preds = dict(zip(keys, x_pred.squeeze(0)))
        m_factors = factors_from_preds(preds, pred_fil.coupling_matrix.links, x_pred).unsqueeze(0)
        # print("m_factors grad:", m_factors.requires_grad)

        M = CouplingMatrix.from_factors(m_factors, pred_fil.coupling_matrix.links,
                                        pred_fil.coupling_matrix.matrix_order)
        corr_fast_calc.update_Q(preds['Q'])
        _, s11_pred, s21_pred, s22_pred = corr_fast_calc.RespM2(M, with_s22=True)


        phi11 = -2*(preds['a11'] + preds['b22'] * w)
        phi22 = -2*(preds['a22'] + preds['b22'] * w)
        phi21 = 0.5 * (phi11 + phi22)

        s11_corr = s11_pred*torch.exp(-1j*phi11)
        s22_corr = s22_pred*torch.exp(-1j*phi22)
        s21_corr = s21_pred*torch.exp(-1j*phi21)

        s_corr = torch.stack([
            s11_corr,
            s21_corr,  # замените на S12, если у вас именно S1_2.db
            # s21_pred,  # либо правильно распакуйте, если RespM2 возвращает все 4
            # s22_corr,
        ]).unsqueeze(0)  # [B, 4, L]

        err = (loss(MWFilter.to_db(s_corr), s_db))
        reg = max(abs(torch.abs(s_corr) - s_mag).flatten())*2
        corr_optim.zero_grad()
        (err + reg).backward()
        corr_optim.step()
        # print(f"Error: {err}")

    with torch.no_grad():
        # m_factors = corr_model(s, preds)
        x_pred = inference_model_ft(s)
        if inference_model_ft.scaler_out is not None:
            x_pred = inference_model_ft._apply_inverse(inference_model_ft.scaler_out, x_pred)
            # print("x_pred inverse grad:", x_pred.requires_grad)
        preds = dict(zip(keys, x_pred.squeeze(0)))
        m_factors = factors_from_preds(preds, pred_fil.coupling_matrix.links, x_pred).unsqueeze(0)

        m_factors = m_factors.squeeze(0)
        total_pred_prms = dict(zip(m_keys, m_factors))
        # print(f"Origin parameters: {pred_prms}")
        # print(f"Tuned parameters: {total_pred_prms}")
        correct_pred_fil = work_model.create_filter_from_prediction(fil, work_model.orig_filter, total_pred_prms)
        # inference_model.plot_origin_vs_prediction(orig_fil, correct_pred_fil, title=f' {i} After correction')
    return  correct_pred_fil.s, correct_pred_fil.coupling_matrix.matrix.numpy()


def predict(fil: rf.Network):
    # s, m = prediction_with_optim_correct(fil)
    s, m = prediction_with_online_correct(fil)
    return s, m



def model_info():
    global work_model
    info = work_model.info()
    return info


def calc_s_params(M:np.array, f0:float, bw:float, Q:float, frange:list or np.array):
    fbw = bw/f0
    S = MWFilter.response_from_coupling_matrix(M=M, f0=f0, FBW=fbw, Q=Q, frange=frange)
    return S

