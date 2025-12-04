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
corr_model = None
corr_fast_calc = None
corr_optim = None
phase_extractor: phase.PhaseLoadingExtractor or None = None


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
        global corr_fast_calc
        corr_fast_calc = FastMN2toSParamCalculation(matrix_order=work_model.orig_filter.coupling_matrix.matrix_order,
                                                    wlist=work_model.orig_filter.f_norm,
                                                    Q=work_model.orig_filter.Q,
                                                    fbw=work_model.orig_filter.fbw)
        global corr_model
        corr_model = CorrectionNet(s_shape=(8, 301), m_dim=len(codec.x_keys))
        global corr_optim
        corr_optim = torch.optim.AdamW(params=corr_model.parameters(), lr=1e-5)

        codec = codec
        work_model.setup(
            model_name="resnet_with_correction",
            model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
            dm_codec=codec
        )
        global inference_model, phase_extractor
        inference_model = work_model.inference(configs.MODEL_CHECKPOINT_PATH)
        # inference_model = work_model.inference(f"D:\\Burlakov\\pyprojects\\mwlab\\saved_models\\EAMU4-KuIMUXT2-BPFC2\\best-epoch=27-train_loss=0.08527-val_loss=0.07506-val_r2=0.99731-val_mse=0.00021-val_mae=0.00531-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
        # inference_model = work_model.inference(
        #     "D:/Burlakov/pyprojects/mwlab/saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=26-train_loss=0.08889-val_loss=0.08248-val_r2=0.99727-val_mse=0.00020-val_mae=0.00647-batch_size=32-base_dataset_size=300000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
        phase_extractor = phase.PhaseLoadingExtractor(inference_model, work_model, work_model.orig_filter)
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e


def predict(fil: rf.Network):
    try:
        global inference_model
        w = work_model.orig_filter.f_norm
        s_resample = S_Resample(301)
        fil = s_resample(fil)
        orig_fil_to_nn = copy.deepcopy(fil)
        # res, S_def = phase.fit_phase_edges_curvefit(
        #     w, orig_fil_to_nn,
        #     # q=0.35,
        #     center_points=(-1.53, 1.53),
        #     correct_freq_dependence=True,
        #     fit_on_extrapolated=True,
        #     plot=False
        # )
        # orig_fil_to_nn.s = S_def

        res = phase_extractor.extract_all(orig_fil_to_nn, w_norm=w)
        ntw_de = res['ntw_deembedded']

        pred_prms = inference_model.predict_x(ntw_de)
        print(f"Предсказанные параметры: {pred_prms}")
        pred_fil = work_model.create_filter_from_prediction(orig_fil_to_nn, work_model.orig_filter, pred_prms,
                                                            work_model.codec)

        ps_shifts = [pred_prms["a11"], pred_prms["a22"], pred_prms["b11"], pred_prms["b22"]]
        a11_final = res['phi1_c'] + pred_prms["a11"]
        a22_final = res['phi2_c'] + pred_prms["a22"]
        b11_final = pred_prms["b11"] + 0.5*res['b11_opt']
        b22_final = pred_prms["b22"] + 0.5*res['b22_opt']
        print(
            f"Финальный фазовый сдвиг, после корректировки ИИ: a11={a11_final:.6f} рад ({np.degrees(a11_final):.2f}°), a22={a22_final:.6f} рад ({np.degrees(a22_final):.2f}°)"
            f" b11 = {b11_final:.6f} рад ({np.degrees(b11_final):.2f}°), b22 = {b22_final:.6f} рад ({np.degrees(b22_final):.2f}°)")
        pred_fil.s[:, 0, 0] *= np.array(torch.exp(-1j * 2 * (a11_final + w * b11_final)), dtype=np.complex64)
        pred_fil.s[:, 0, 1] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))),
                                        dtype=np.complex64)
        pred_fil.s[:, 1, 0] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))),
                                        dtype=np.complex64)
        pred_fil.s[:, 1, 1] *= np.array(torch.exp(-1j * 2 * (a22_final + w * b22_final)), dtype=np.complex64)
        # optim_matrix = optimize_cm(pred_fil, orig_fil)
        # print(f"Оптимизированные параметры: {optim_matrix.factors}. Добротность: {pred_fil.Q}. Фаза:")
        optim_matrix, Q, phase_opt = optimize_cm(pred_fil, fil,
                                                 phase_init=(a11_final, a22_final, b11_final, b22_final))
        print(
            f"Оптимизированные параметры: {optim_matrix.factors}. Добротность: {pred_fil.Q}. Фаза: {np.degrees(phase_opt)}")

        # pred_prms = inference_model.predict_x(fil)
        # print(f"Предсказанные параметры: {pred_prms}")
        # pred_fil = inference_model.create_filter_from_prediction(fil, work_model.orig_filter, pred_prms, work_model.codec)
        # # pred_fil = inherence_correct(fil)
        # # optim_matrix = pred_fil.coupling_matrix
        # optim_matrix = optimize_cm(pred_fil, fil, pred_fil.f_norm)
        # print(f"Оптимизированные параметры: {optim_matrix.factors}")
        S = calc_s_params(optim_matrix.matrix.numpy(), pred_fil.f0, pred_fil.bw, pred_fil.Q, fil.f/1e6)
        a11_final, a22_final, b11_final, b22_final = phase_opt
        S[:, 0, 0] *= np.array(torch.exp(-1j * 2 * (a11_final + w * b11_final)), dtype=np.complex64)
        S[:, 0, 1] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))), dtype=np.complex64)
        S[:, 1, 0] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))), dtype=np.complex64)
        S[:, 1, 1] *= np.array(torch.exp(-1j * 2 * (a22_final + w * b22_final)), dtype=np.complex64)
        return S, optim_matrix.matrix.numpy()
        # return pred_fil.s, pred_fil.coupling_matrix.matrix.numpy()
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e


def _online_correct(fil: rf.Network):
    codec = work_model.codec

    loss = CustomLosses("log_cosh")
    codec_db = copy.deepcopy(codec)
    codec_db.y_channels = ['S1_1.db', 'S2_1.db', 'S2_2.db']

    pred_prms = inference_model.predict_x(fil)
    pred_fil = inference_model.create_filter_from_prediction(fil, work_model.orig_filter, pred_prms, work_model.codec)
    keys = pred_prms.keys()
    ts = TouchstoneData(fil)
    preds = pred_fil.coupling_matrix.factors.unsqueeze(0)
    s = codec.encode(ts)[1].unsqueeze(0)
    s_db = codec_db.encode(ts)[1].unsqueeze(0)
    print(f"Initial loss: {loss(s_db, codec_db.encode(TouchstoneData(pred_fil))[1].unsqueeze(0))}")
    global corr_model, corr_optim, corr_fast_calc
    corr_model.train()
    start_time = time.time()
    for i in range(200):
        correction = corr_model(s, preds)
        total_pred = correction  # скорректированная матрица
        M = CouplingMatrix.from_factors(total_pred, pred_fil.coupling_matrix.links, pred_fil.coupling_matrix.matrix_order)
        _, s11_pred, s21_pred, s22_pred = corr_fast_calc.RespM2(M, with_s22=True)
        s_db_corr = torch.stack([
            MWFilter.to_db(s11_pred),
            # MWFilter.to_db(s21_pred),  # замените на S12, если у вас именно S1_2.db
            MWFilter.to_db(s21_pred),  # либо правильно распакуйте, если RespM2 возвращает все 4
            MWFilter.to_db(s22_pred),
        ]).unsqueeze(0)  # [B, 4, L] — подгоните под ваш codec_db
        err = loss(s_db_corr, s_db)
        print(f"[{i}] Error: {err.item()}")
        corr_optim.zero_grad()
        err.backward()
        corr_optim.step()
    stop_time = time.time()
    print(f"Tuning time: {stop_time-start_time}")

    with torch.no_grad():
        correction = corr_model(s, preds)
        total_pred = correction  # скорректированная матрица
        total_pred = total_pred.squeeze(0)
        total_pred_prms = dict(zip(keys, total_pred))
        print(f"Origin parameters: {pred_prms}")
        print(f"Tuned parameters: {total_pred_prms}")
        correct_pred_fil = inference_model.create_filter_from_prediction(fil, work_model.orig_filter, pred_prms, work_model.codec)
        # inference_model.plot_origin_vs_prediction(fil, correct_pred_fil)
    return  correct_pred_fil


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

