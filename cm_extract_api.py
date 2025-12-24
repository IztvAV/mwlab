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


class CorrectionNet(nn.Module):
    def __init__(self, s_shape, m_dim, hidden_dim=256):
        super().__init__()
        s_dim = s_shape[0] * s_shape[1]  # 301 * 8 = 2408 or 301 * 4 = 1204
        self.fc1 = nn.Linear(s_dim + m_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, m_dim)

    def forward(self, s_params, matrix_pred):
        s_params = s_params.view(s_params.size(0), -1)  # B x 2408 or B x 1204
        x = torch.cat([s_params, matrix_pred], dim=1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        delta = self.fc3(x)
        return matrix_pred + delta



def load_model(manifest_path: str|os.PathLike|None=None):
    """ Пока в таком формате. В дальнейшем надо будет стандартизировать названия моделей """
    try:
        global work_model
        if manifest_path is None:
            configs = cfg.Configs.init_as_default("D:\\Burlakov\\pyprojects\\mwlab\\default.yml")
        else:
            configs = cfg.Configs(manifest_path)
        work_model = common.WorkModel(configs, SamplerTypes.SAMPLER_SOBOL)
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
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e


def phase_extract(fil: rf.Network):
    global phase_extractror, work_model, calibration_res, calibrated
    if not calibrated:
        print("Start calibration")
        w = work_model.orig_filter.f_norm
        calibration_res = phase_extractor.extract_all(fil, w_norm=w)
        ntw_de = calibration_res['ntw_deembedded']
        calibrated = True
    else:
        w = work_model.orig_filter.f_norm
        a11 = calibration_res['phi1_c']
        a22 = calibration_res['phi2_c']
        b11 = calibration_res['b11_opt']
        b22 = calibration_res['b22_opt']
        ntw_de = phase.apply_phase_for_ntw(fil, w, a11, b11, a22, b22)
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
    pred_fil = work_model.create_filter_from_prediction(fil, work_model.orig_filter, pred_prms,
                                                        work_model.codec)

    a11_final = calibration_res['phi1_c'] + pred_prms["a11"]
    a22_final = calibration_res['phi2_c'] + pred_prms["a22"]
    b11_final = pred_prms["b11"] + calibration_res['b11_opt']
    b22_final = pred_prms["b22"] + calibration_res['b22_opt']
    print(
        f"Финальный фазовый сдвиг, после корректировки ИИ: a11={a11_final:.6f} рад ({np.degrees(a11_final):.2f}°), a22={a22_final:.6f} рад ({np.degrees(a22_final):.2f}°)"
        f" b11 = {b11_final:.6f} рад ({np.degrees(b11_final):.2f}°), b22 = {b22_final:.6f} рад ({np.degrees(b22_final):.2f}°)")

    pred_fil = phase.apply_phase_for_ntw(pred_fil, w, a11_final, b11_final, a22_final, b22_final)
    optim_filter, phase_opt = optimize_cm(pred_fil, fil, phase_init=(a11_final, a22_final, b11_final, b22_final), plot=False)
    print(f"Оптимизированные параметры: {optim_filter.coupling_matrix.factors}. Добротность: {optim_filter.Q}. Фаза: {np.degrees(phase_opt)}")

    optim_filter = phase.apply_phase_for_ntw(optim_filter, w, *phase_opt)
    return optim_filter.s, optim_filter.coupling_matrix.matrix.numpy()

pred_fil = None

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

    global inference_model_ft, corr_optim, corr_fast_calc, pred_fil

    if pred_fil is None:
        pred_fil = fil
    else:
        delta = abs(pred_fil.s) - abs(fil.s)
        pass

    codec = work_model.codec

    loss = nn.L1Loss()

    codec_db = copy.deepcopy(codec)
    codec_db.y_channels = ['S1_1.db', 'S2_1.db']
    codec_mag = copy.deepcopy(codec)
    codec_mag.y_channels = ['S1_1.mag', 'S2_1.mag']

    ntw_de = phase_extract(fil)
    pred_prms = inference_model_ft.predict_x(ntw_de)
    pred_fil = work_model.create_filter_from_prediction(fil, work_model.orig_filter, pred_prms, work_model.codec)

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
        correct_pred_fil = work_model.create_filter_from_prediction(fil, work_model.orig_filter, total_pred_prms, work_model.codec)
        # inference_model.plot_origin_vs_prediction(orig_fil, correct_pred_fil, title=f' {i} After correction')
    return  correct_pred_fil.s, correct_pred_fil.coupling_matrix.matrix.numpy()


def predict(fil: rf.Network):
    try:
        # s, m = prediction_with_optim_correct(fil)
        s, m = prediction_with_online_correct(fil)
        return s, m
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e



prev_x = None
tuned_prms = None
last_fil = None

def inherence_correct(orig_fil: rf.Network):
    global prev_x, tuned_prms, last_fil
    codec = work_model.codec
    fast_calc = FastMN2toSParamCalculation(matrix_order=work_model.orig_filter.coupling_matrix.matrix_order,
                                           wlist=work_model.orig_filter.f_norm,
                                           Q=work_model.orig_filter.Q,
                                           fbw=work_model.orig_filter.fbw)
    loss = CustomLosses("log_cosh")
    # loss = nn.L1Loss()
    codec_db = copy.deepcopy(codec)
    codec_db.y_channels = ['S1_1.db', 'S1_2.db', 'S2_2.db']

    # Инициализация перед циклом: стартуем с предсказания сети под текущую АЧХ
    if prev_x is None:
        pred_prms = inference_model.predict_x(orig_fil)
        pred_fil = inference_model.create_filter_from_prediction(orig_fil, work_model.orig_filter, pred_prms, work_model.codec)
        prev_x = pred_fil.coupling_matrix.factors.detach()
    else:
        pred_fil = last_fil

    # >>> new: аккумуляторы метрик по всем итерациям
    total_L, total_fit, total_time = 0.0, 0.0, 0.0

    ts = TouchstoneData(orig_fil)

    # целевая АЧХ (dB)
    s_target = codec_db.encode(ts)[1].unsqueeze(0)  # [1, C, L]

    # >>> new: старт замера времени итерации
    t0 = time.time()

    # холодный старт + смешивание с прошлым
    cold_prms = inference_model.predict_x(orig_fil)
    cold_fil = inference_model.create_filter_from_prediction(orig_fil, work_model.orig_filter, cold_prms, work_model.codec)
    cold_x = cold_fil.coupling_matrix.factors.detach()

    alpha = 0.5  # доля «памяти»
    x0 = (alpha * prev_x + (1 - alpha) * cold_x).clone().detach().requires_grad_(True)

    # (опционально) Q как параметр
    q0 = torch.nn.Parameter(torch.tensor(
        pred_fil.Q if hasattr(pred_fil, "Q") else work_model.orig_filter.Q,
        dtype=torch.float32))
    # gamma = torch.nn.Parameter(torch.tensor(1.0 / float(pred_fil.Q), dtype=torch.float32))
    # gamma_prev = gamma.detach().clone()

    opt = torch.optim.LBFGS([x0], lr=1, max_iter=1000, line_search_fn='strong_wolfe')
    mu = 1e-2  # сила якоря к прошлой матрице
    mu_gamma = 1e-2

    fast = FastMN2toSParamCalculation(
        matrix_order=pred_fil.coupling_matrix.matrix_order,
        fbw=work_model.orig_filter.fbw,
        Q=pred_fil.Q,  # стартовое Q (реальное число)
        wlist=work_model.orig_filter.f_norm,  # ваша сетка частот (нормированная)
    )

    # delta_gamma_allow = (0.1 * gamma_prev.abs()).clamp_min(1e-12)  # 5% от прошлого γ
    def closure():
        opt.zero_grad()
        M = CouplingMatrix.from_factors(x0, pred_fil.coupling_matrix.links, pred_fil.coupling_matrix.matrix_order)

        # q0 = 1/gamma
        # fast.update_Q(q0)
        _, s11, s21, s22 = fast.RespM2(M, with_s22=True)
        s_pred = torch.stack([MWFilter.to_db(s11), MWFilter.to_db(s21), MWFilter.to_db(s22)]).unsqueeze(0)

        fit = torch.sqrt(loss(s_pred, s_target))
        reg = mu * torch.median(torch.abs(x0 - prev_x))
        # reg_gamma = mu_gamma * ((gamma - gamma_prev)** 2)
        L = fit + reg
        L.backward()
        return L

    opt.step(closure)

    # >>> new: чистая оценка финального лосса и времени (без градиентов)
    @torch.no_grad()
    def eval_final():
        M = CouplingMatrix.from_factors(x0, pred_fil.coupling_matrix.links, pred_fil.coupling_matrix.matrix_order)
        # q0 = 1/gamma
        # fast.update_Q(q0)  # или q0.detach()
        _, s11, s21, s22 = fast.RespM2(M, with_s22=True)
        s_pred = torch.stack([MWFilter.to_db(s11), MWFilter.to_db(s21), MWFilter.to_db(s22)]).unsqueeze(0)
        fit = loss(s_pred, s_target)
        reg = mu * torch.median(torch.abs(x0 - prev_x))
        L = fit + reg
        print(f"Extracted Q: {q0.item():.3f}")  # 👈 печатайте .item()
        return fit.item(), reg.item(), L.item()
    fit_val, reg_val, L_val = eval_final()
    elapsed = time.time() - t0

    # >>> new: печать и аккумуляция метрик
    print(f"FINAL: L={L_val:.6f} (fit={fit_val:.6f}, prox={reg_val:.6f}) | time={elapsed:.2f}s")
    total_L += L_val
    total_fit += fit_val
    total_time += elapsed

    with torch.no_grad():
        # переносим как якорь на следующую итерацию
        prev_x = x0.detach().clone()
        tuned_prms = dict(zip(cold_prms.keys(), prev_x))
        pred_fil = MWFilterBaseLMWithMetrics.create_filter_from_prediction(orig_fil, work_model.orig_filter, tuned_prms, work_model.codec)
        # pred_fil._Q = 1/gamma
        # pred_fil.coupling_matrix.plot_matrix()
        # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
        last_fil = pred_fil
    return pred_fil



def model_info():
    try:
        global work_model
        info = work_model.info()
        return info
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e


def calc_s_params(M:np.array, f0:float, bw:float, Q:float, frange:list or np.array):
    fbw = bw/f0
    S = MWFilter.response_from_coupling_matrix(M=M, f0=f0, FBW=fbw, Q=Q, frange=frange)
    return S

