import copy
import os
import random
import time
from pathlib import Path

import numpy as np

import cauchy_method
import phase
import skrf as rf
import pandas as pd
# from utils import make_network
from cm_extract_api import inference_model
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
from mwlab import TouchstoneDataset, TouchstoneLDataModule, TouchstoneDatasetAnalyzer, TouchstoneData

from filters import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers, SamplerTypes, MWFilter, CouplingMatrix
from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLModule, MWFilterBaseLMWithMetrics

from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift

import matplotlib.pyplot as plt
import lightning as L
from scipy.optimize import minimize
from torch import nn
import models
import torch
from filters.mwfilter_optim.bfgs import optimize_cm
from losses import CustomLosses
import configs
import common
import torch.nn.functional as F
from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.nn.scalers import MinMaxScaler
torch.set_float32_matmul_precision("medium")


class MatrixCorrectionNet(nn.Module):
    def __init__(self, s_shape, m_dim, hidden_dim=256):
        super().__init__()
        s_dim = s_shape[0] * s_shape[1]
        self.fc1 = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, hidden_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim)
         )
        # self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, m_dim)

    def forward(self, s_params_diff):
        s_params_diff = s_params_diff.view(s_params_diff.size(0), -1)
        x = F.silu(self.fc1(s_params_diff))
        x = F.silu(self.fc2(x))
        # x = F.silu(self.fc3(x))
        delta = self.fc3(x)
        return delta


class CorrectionNet(nn.Module):
    def __init__(self, s_shape, m_dim, hidden_dim=256):
        super().__init__()
        s_dim = s_shape[0] * s_shape[1]  # 301 * 8 = 2408 or 301 * 4 = 1204
        self.fc1 = nn.Sequential(
            nn.LayerNorm(s_dim + m_dim),
            nn.Linear(s_dim + m_dim, hidden_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim)
         )
        self.fc3 = nn.Linear(hidden_dim, m_dim)

    def forward(self, s_params, matrix_pred):
        s_params = s_params.view(s_params.size(0), -1)  # B x 2408 or B x 1204
        x = torch.cat([s_params, matrix_pred], dim=1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        delta = self.fc3(x)
        return matrix_pred + delta


class SimpleCorrNet(nn.Module):
    """
    Простой корректор:
      S-параметры [B, C_s, L] -> Conv1d x2 -> GlobalAvgPool -> concat([feat, m_pred]) ->
      Linear -> ΔM -> (m_pred + ΔM)
    """
    def __init__(self,
                 s_channels: int,      # число каналов S (напр. 3: S11_dB, S21_dB, S22_dB)
                 m_dim: int,           # длина вектора факторов (len(codec.x_keys))
                 hidden_conv: int = 64,
                 hidden_mlp: int = 128,
                 kernel_size: int = 5,
                 delta_max: float = 2.0):   # мягкое ограничение амплитуды коррекции
        super().__init__()
        pad = (kernel_size - 1) // 2

        # 2 лёгкие свёртки по частоте
        self.conv1 = nn.Conv1d(s_channels, hidden_conv, kernel_size=kernel_size+2, padding=pad, bias=True, stride=2)
        self.conv2 = nn.Conv1d(hidden_conv, hidden_conv, kernel_size=kernel_size, padding=pad, bias=True)

        # глобальный pooling по частоте: [B, H, L] -> [B, H]
        self.gap = nn.AdaptiveAvgPool1d(1)

        # маленькая MLP-голова
        self.fc1 = nn.Linear(hidden_conv + m_dim, hidden_mlp)
        self.fc2 = nn.Linear(hidden_mlp, m_dim)

        self.delta_max = float(delta_max)

    def forward(self, s_params: torch.Tensor, m_pred: torch.Tensor) -> torch.Tensor:
        """
        s_params: [B, C_s, L]  (в дБ или линейке, как у вас)
        m_pred:   [B, m_dim]   (база от основной модели, обычно .detach())
        return:   [B, m_dim]   (скорректированные факторы = m_pred + ΔM)
        """
        x = F.silu(self.conv1(s_params))
        x = F.silu(self.conv2(x))
        x = self.gap(x).squeeze(-1)           # [B, hidden_conv]

        h = torch.cat([x, m_pred], dim=1)     # [B, hidden_conv + m_dim]
        h = F.silu(self.fc1(h))
        delta = torch.tanh(self.fc2(h))
        return m_pred + delta


def online_correct():
    work_model = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE, SamplerTypes.SAMPLER_SOBOL)
    # sensivity.run(work_model.orig_filter)
    # common.plot_distribution(work_model.ds, num_params=len(work_model.ds_gen.origin_filter.coupling_matrix.links))
    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 # keys_for_analysis=[f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links]+["Q"])
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                    work_model.orig_filter.coupling_matrix.links])
    codec = codec
    # work_model_inference_main = common.WorkModel(configs.ENV_DATASET_PATH, 1_000, SamplerTypes.SAMPLER_SOBOL)
    # work_model_inference_main_extra = common.WorkModel(configs.ENV_DATASET_PATH, 1_000, SamplerTypes.SAMPLER_SOBOL)
    work_model_wide = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE, SamplerTypes.SAMPLER_SOBOL)
    # # work_model_wide_extra = common.WorkModel(configs.ENV_DATASET_PATH, 1_000_000, SamplerTypes.SAMPLER_SOBOL)
    #
    # work_model_inference_main.setup(
    #     model_name="resnet_with_correction",
    #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # inf_model = work_model_inference_main.inference(
    #     "saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=29-train_loss=0.25454-val_loss=0.34502-val_r2=0.68672-val_mse=0.02497-val_mae=0.09413-batch_size=32-base_dataset_size=50000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    #
    # work_model_inference_main_extra.setup(
    #     model_name="resnet_with_correction",
    #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # work_model_inference_main_extra.model = inf_model.model
    # inf_model = work_model_inference_main_extra.inference(
    #     "saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=29-train_loss=0.26492-val_loss=0.27672-val_r2=0.81385-val_mse=0.01482-val_mae=0.06256-batch_size=32-base_dataset_size=300000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    #
    # work_model_wide.setup(
    #     model_name="resnet_with_wide_correction",
    #     model_cfg={"main_model": inf_model.model, "in_channels": len(codec.y_channels),
    #                "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # inference_model = work_model_wide.inference("saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=17-train_loss=0.25331-val_loss=0.25480-val_r2=0.86440-val_mse=0.01074-val_mae=0.05441-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    #
    work_model_wide.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )
    # inference_model = work_model_wide.inference(
    #     "saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=29-train_loss=0.23384-val_loss=0.25414-val_r2=0.88339-val_mse=0.00926-val_mae=0.05538-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # sch = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=40, gamma=0.1)


    # Загружаем лучшую модель
    # checkpoint_path="saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=12-val_loss=0.01266-train_loss=0.01224.ckpt",
    # checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=49-train_loss=0.03379-val_loss=0.03352-val_r2=0.84208-val_acc=0.25946-val_mae=0.05526-batch_size=32-dataset_size=100000.ckpt",
    # "saved_models/ERV-KuIMUXT1-BPFC1/best-epoch=22-train_loss=0.04863-val_loss=0.05762-val_r2=0.82785-val_mse=0.01366-val_mae=0.04395-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt"
    # inference_model = work_model.inference("saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=29-train_loss=0.04166-val_loss=0.04450-val_r2=0.92560-val_mse=0.00588-val_mae=0.03862-batch_size=32-base_dataset_size=1500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=29-train_loss=0.03546-val_loss=0.03841-val_r2=0.94190-val_mse=0.00459-val_mae=0.03381-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=34-train_loss=0.02388-val_loss=0.02641-val_r2=0.96637-val_mse=0.00265-val_mae=0.02376-batch_size=32-base_dataset_size=600000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=25-train_loss=0.02530-val_loss=0.02793-val_r2=0.96251-val_mse=0.00296-val_mae=0.02496-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference(
    #     "saved_models/EAMU4-KuIMUXT2-BPFC2/best-epoch=77-train_loss=0.13232-val_loss=0.28403-val_r2=0.76389-val_mse=0.01879-val_mae=0.06333-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    inference_model = work_model_wide.inference("saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=28-train_loss=0.14318-val_loss=0.14503-val_r2=0.97508-val_mse=0.00196-val_mae=0.01912-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference(lit_model.trainer.checkpoint_callback.best_model_path)

    tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/24.10.25/non-shifted")

    fast_calc = FastMN2toSParamCalculation(matrix_order=work_model_wide.orig_filter.coupling_matrix.matrix_order,
                                           wlist=work_model_wide.orig_filter.f_norm,
                                           Q=work_model_wide.orig_filter.Q,
                                           fbw=work_model_wide.orig_filter.fbw)
    loss = CustomLosses("log_cosh")
    codec_db = copy.deepcopy(codec)
    codec_db.y_channels = ['S1_1.db', 'S1_2.db', 'S2_2.db']

    corr_model = MatrixCorrectionNet(m_dim=len(codec.x_keys), hidden_dim=512)
    # corr_model = CorrectionNet(s_shape=(3, 301), m_dim=len(codec.x_keys), hidden_dim=512)
    corr_model.train()
    total_err = 0
    optim = torch.optim.AdamW(params=corr_model.parameters(), lr=1e-5, weight_decay=1e-2)
    for i in range(len(tds)):
        # i = random.randint(0, len(tds))
        start_time = time.time()
        orig_fil = tds[i][1]

        # phi11, phi22, phi21 = phase.fit_phases(work_model.orig_filter.f_norm, orig_fil.s[:, 0, 0], orig_fil.s[:, 1, 1], orig_fil.s[:, 1, 0], plot=False)
        # print(f'phi11: {phi11}, phi22: {phi22}, phi21: {phi21}')
        # S11_corrected, S22_corrected, S21_corrected = phase.correct_phase_shift(work_model.orig_filter.f_norm, orig_fil.s[:, 0, 0],
        #                                                                         orig_fil.s[:, 1, 1], orig_fil.s[:, 1, 0], phi11, phi22, phi21,
        #                                                                   plot=False)
        # orig_fil.s[:, 0, 0] = S11_corrected
        # orig_fil.s[:, 0, 1] = S21_corrected
        # orig_fil.s[:, 1, 0] = S21_corrected
        # orig_fil.s[:, 1, 1] = S22_corrected
        # start_time = time.time()
        pred_prms = inference_model.predict_x(orig_fil)
        # stop_time = time.time()
        # print(f"Predict time: {stop_time - start_time:.3f} sec")
        pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, work_model_wide.meta)
        # optim_matrix = optimize_cm(pred_fil, orig_fil, pred_fil.f_norm, plot=False)
        # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
        keys = pred_prms.keys()
        ts = TouchstoneData(orig_fil)
        preds = pred_fil.coupling_matrix.factors.unsqueeze(0)
        # preds = optim_matrix.factors.unsqueeze(0)
        s = codec.encode(ts)[1].unsqueeze(0)
        s_db = codec_db.encode(ts)[1].unsqueeze(0)
        print(f"Initial loss: {loss(s_db, codec_db.encode(TouchstoneData(pred_fil))[1].unsqueeze(0))}")
        err = 0
        sch = torch.optim.lr_scheduler.StepLR(optim, step_size=120, gamma=0.5)
        for j in range(500):

            M = CouplingMatrix.from_factors(preds, pred_fil.coupling_matrix.links, pred_fil.coupling_matrix.matrix_order)
            _, s11_pred, s21_pred, s22_pred = fast_calc.RespM2(M, with_s22=True)
            s_db_corr = torch.stack([
                MWFilter.to_db(s11_pred),
                MWFilter.to_db(s21_pred),  # замените на S12, если у вас именно S1_2.db
                # MWFilter.to_db(s21_pred),  # либо правильно распакуйте, если RespM2 возвращает все 4
                MWFilter.to_db(s22_pred),
            ]).unsqueeze(0)  # [B, 4, L] — подгоните под ваш codec_db
            s_diff = s_db - s_db_corr

            m_factors = preds + corr_model(s_diff)

            M = CouplingMatrix.from_factors(m_factors, pred_fil.coupling_matrix.links,
                                            pred_fil.coupling_matrix.matrix_order)
            _, s11_pred, s21_pred, s22_pred = fast_calc.RespM2(M, with_s22=True)
            s_db_corr = torch.stack([
                MWFilter.to_db(s11_pred),
                MWFilter.to_db(s21_pred),  # замените на S12, если у вас именно S1_2.db
                # MWFilter.to_db(s21_pred),  # либо правильно распакуйте, если RespM2 возвращает все 4
                MWFilter.to_db(s22_pred),
            ]).unsqueeze(0)  # [B, 4, L]

            err = loss(s_db_corr, s_db)
            optim.zero_grad()
            err.backward()
            optim.step()
            sch.step(j)
            # print(f"Error: {err}")

        with torch.no_grad():
            m_factors = corr_model( preds)
            m_factors = m_factors.squeeze(0)
            total_pred_prms = dict(zip(keys, m_factors))
            print(f"Origin parameters: {pred_prms}")
            print(f"Tuned parameters: {total_pred_prms}")
            correct_pred_fil = MWFilterBaseLMWithMetrics.create_filter_from_prediction(orig_fil, total_pred_prms, work_model_wide.meta)
            inference_model.plot_origin_vs_prediction(orig_fil, correct_pred_fil)
        correct_pred_fil.coupling_matrix.plot_matrix()
        # optim_matrix = optimize_cm(pred_fil, orig_fil, work_model.orig_filter.f_norm)
        # optim_matrix.plot_matrix()
        total_err += err
        stop_time = time.time()
        print(f"[{i}] Total error: {err.item()}. Tuning time: {stop_time - start_time}")
    print(f"Mean error: {total_err/len(tds)}")


def inherence_correct():
    work_model = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE, SamplerTypes.SAMPLER_SOBOL)
    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 # keys_for_analysis=[f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links]+["Q"])
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                    work_model.orig_filter.coupling_matrix.links])
    codec = codec
    work_model.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )

    # Загружаем лучшую модель
    inference_model = work_model.inference("saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=28-train_loss=0.14318-val_loss=0.14503-val_r2=0.97508-val_mse=0.00196-val_mae=0.01912-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")

    tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/24.10.25/non-shifted")

    fast_calc = FastMN2toSParamCalculation(matrix_order=work_model.orig_filter.coupling_matrix.matrix_order,
                                           wlist=work_model.orig_filter.f_norm,
                                           Q=work_model.orig_filter.Q,
                                           fbw=work_model.orig_filter.fbw)
    loss = CustomLosses("log_cosh")
    codec_db = copy.deepcopy(codec)
    codec_db.y_channels = ['S1_1.db', 'S1_2.db', 'S2_2.db']

    corr_model = MatrixCorrectionNet(m_dim=len(codec.x_keys), hidden_dim=1024, s_shape=(len(codec_db.y_channels), 301))
    # corr_model = CorrectionNet(s_shape=(3, 301), m_dim=len(codec.x_keys), hidden_dim=512)
    corr_model.train()
    total_err = 0
    optim = torch.optim.AdamW(params=corr_model.parameters(), lr=1e-5, weight_decay=1e-2)

    # Инициализация перед циклом: стартуем с предсказания сети под текущую АЧХ
    orig_fil = tds[0][1]
    pred_prms = inference_model.predict_x(orig_fil)
    pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, work_model.meta)

    prev_x = pred_fil.coupling_matrix.factors.detach()

    # >>> new: аккумуляторы метрик по всем итерациям
    total_L, total_fit, total_time = 0.0, 0.0, 0.0

    for i in range(len(tds)):
        orig_fil = tds[i][1]
        ts = TouchstoneData(orig_fil)

        # целевая АЧХ (dB)
        s_target = codec_db.encode(ts)[1].unsqueeze(0)  # [1, C, L]

        # >>> new: старт замера времени итерации
        t0 = time.time()

        # холодный старт + смешивание с прошлым
        cold_prms = inference_model.predict_x(orig_fil)
        cold_fil = inference_model.create_filter_from_prediction(orig_fil, cold_prms, work_model.meta)
        cold_x = cold_fil.coupling_matrix.factors.detach()

        alpha = 1.0  # доля «памяти»
        x0 = (alpha * prev_x + (1 - alpha) * cold_x).clone().detach().requires_grad_(True)

        # (опционально) Q как параметр
        q0 = torch.nn.Parameter(torch.tensor(
            pred_fil.Q if hasattr(pred_fil, "Q") else work_model.orig_filter.Q,
            dtype=torch.float32))
        # gamma = torch.nn.Parameter(torch.tensor(1.0 / float(pred_fil.Q), dtype=torch.float32))
        # gamma_prev = gamma.detach().clone()

        opt = torch.optim.LBFGS([x0, q0], lr=1, max_iter=1000, line_search_fn='strong_wolfe')
        mu = 1e-1  # сила якоря к прошлой матрице
        mu_gamma = 1e-2

        fast = FastMN2toSParamCalculation(
            matrix_order=pred_fil.coupling_matrix.matrix_order,
            fbw=work_model.orig_filter.fbw,
            Q=pred_fil.Q,  # стартовое Q (реальное число)
            wlist=work_model.orig_filter.f_norm,  # ваша сетка частот (нормированная)
        )

        # delta_gamma_allow = (0.05 * gamma_prev.abs()).clamp_min(1e-12)  # 5% от прошлого γ
        def closure():
            opt.zero_grad()
            M = CouplingMatrix.from_factors(x0, pred_fil.coupling_matrix.links, pred_fil.coupling_matrix.matrix_order)

            # q0 = 1/gamma
            fast.update_Q(q0)
            _, s11, s21, s22 = fast.RespM2(M, with_s22=True)
            s_pred = torch.stack([MWFilter.to_db(s11), MWFilter.to_db(s21), MWFilter.to_db(s22)]).unsqueeze(0)

            fit = loss(s_pred, s_target)
            reg = mu * torch.mean((x0 - prev_x) ** 2)
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
            fast.update_Q(q0)  # или q0.detach()
            _, s11, s21, s22 = fast.RespM2(M, with_s22=True)
            s_pred = torch.stack([MWFilter.to_db(s11), MWFilter.to_db(s21), MWFilter.to_db(s22)]).unsqueeze(0)
            fit = loss(s_pred, s_target)
            reg = mu * torch.mean((x0 - prev_x) ** 2)
            L = fit + reg
            print(f"Extracted Q: {q0.item():.3f}")  # 👈 печатайте .item()
            return fit.item(), reg.item(), L.item()
        fit_val, reg_val, L_val = eval_final()
        elapsed = time.time() - t0

        # >>> new: печать и аккумуляция метрик
        print(f"[{i}] FINAL: L={L_val:.6f} (fit={fit_val:.6f}, prox={reg_val:.6f}) | time={elapsed:.2f}s")
        total_L += L_val
        total_fit += fit_val
        total_time += elapsed

        with torch.no_grad():
            # переносим как якорь на следующую итерацию
            prev_x = x0.detach().clone()
            tuned_prms = dict(zip(cold_prms.keys(), prev_x))
            pred_fil = MWFilterBaseLMWithMetrics.create_filter_from_prediction(orig_fil, tuned_prms, work_model.meta)
            # pred_fil.coupling_matrix.plot_matrix()
            inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)

    # >>> new: сводка по всему циклу
    n = len(tds)
    print(f"MEAN over {n} iters: L={total_L / n:.6f}, fit={total_fit / n:.6f}, time/iter={total_time / n:.2f}s")

def make_phase_objective(ntw_orig, inference_model, work_model, reference_filter, w_norm):
    """
    Возвращает функцию objective(params), в которой:
      - S_raw и f кэшированы
      - есть один постоянный ntw_de, в котором мы только обновляем .s
    """
    f = ntw_orig.f
    S_raw = ntw_orig.s  # (N,2,2), без копии – и так достаточно

    # Разложим один раз
    S11_raw_full = S_raw[:, 0, 0]
    S21_raw_full = S_raw[:, 0, 1]
    S22_raw_full = S_raw[:, 1, 1]

    # Один reusable Network для подачи в нейросеть
    ntw_de = ntw_orig.copy()
    ntw_de.name = "De-embedded_for_NN"

    w_norm = np.asarray(w_norm, dtype=np.float64)

    def objective(params, verbose=False):
        b11_opt, b22_opt = params

        # 1) DE-EMBED кандидата (снимаем фазу)
        S11_de, S21_de, S22_de = phase.apply_phase_all(
            S11_raw_full, S21_raw_full, S22_raw_full,
            w_norm,
            a11=0.0, b11=-b11_opt,
            a22=0.0, b22=-b22_opt,
        )

        # Собираем новый S в один массив (без создания нового Network)
        S_new = np.empty_like(S_raw)
        S_new[:, 0, 0] = S11_de
        S_new[:, 0, 1] = S21_de
        S_new[:, 1, 0] = S21_de
        S_new[:, 1, 1] = S22_de

        ntw_de.s = S_new

        # 2) NN → CM + своя фазовая нагрузка
        pred_params = inference_model.predict_x(ntw_de)

        pred_filter = work_model.create_filter_from_prediction(
            ntw_de, reference_filter, pred_params, work_model.codec
        )
        S_pred = pred_filter.s
        S11_ideal = S_pred[:, 0, 0]
        S21_ideal = S_pred[:, 0, 1]
        S22_ideal = S_pred[:, 1, 1]

        # 3) Фаза от сети
        a11_nn = float(pred_params.get("a11", 0.0))
        b11_nn = float(pred_params.get("b11", 0.0))
        a22_nn = float(pred_params.get("a22", 0.0))
        b22_nn = float(pred_params.get("b22", 0.0))

        # 4) Суммарная фаза
        a11_total = a11_nn
        a22_total = a22_nn
        b11_total = b11_opt + b11_nn
        b22_total = b22_opt + b22_nn

        # 5) Вешаем суммарную фазу на идеальный отклик
        S11_final, S21_final, S22_final = phase.apply_phase_all(
            S11_ideal, S21_ideal, S22_ideal,
            w_norm,
            a11=a11_total, b11=b11_total,
            a22=a22_total, b22=b22_total,
        )

        # 6) Ошибка по комплексным S
        loss = (
            np.mean(np.abs(S11_final - S11_raw_full)) +
            np.mean(np.abs(S21_final - S21_raw_full)) +
            np.mean(np.abs(S22_final - S22_raw_full))
        )

        if verbose:
            print(
                f"[b11_opt={b11_opt:.4f}, b22_opt={b22_opt:.4f}] "
                f"b11_nn={b11_nn:.4f}, b22_nn={b22_nn:.4f}, "
                f"loss={loss:.6f}"
            )

        return loss

    return objective


def coarse_grid_search(obj, b_min=-np.pi, b_max=np.pi, n_grid=25):
    b_vals = np.linspace(b_min, b_max, n_grid)

    best_loss = np.inf
    best_b11 = None
    best_b22 = None

    for b11 in b_vals:
        for b22 in b_vals:
            loss = obj((b11, b22))
            if loss < best_loss:
                best_loss = loss
                best_b11, best_b22 = b11, b22

    print(f"[GRID] best_loss={best_loss:.6f} at b11_opt={best_b11:.4f}, b22_opt={best_b22:.4f}")
    return np.array([best_b11, best_b22])


def refine_local(x0, obj):
    res = minimize(
        lambda x: obj(x),
        x0,
        method='Nelder-Mead',
        options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    print(f"[NELDER-MEAD] x_opt={res.x}, loss={res.fun:.6f}, success={res.success}")
    return res


_last_x0 = np.array([0.0, 0.0])

def optimize_phase_loading(ntw_orig, inference_model, work_model, reference_filter, w_norm,
                           use_grid=False):
    global _last_x0
    start_time = time.time()

    w_norm = np.asarray(w_norm, dtype=np.float64)
    obj = make_phase_objective(ntw_orig, inference_model, work_model, reference_filter, w_norm)

    if use_grid:
        x0 = coarse_grid_search(obj, b_min=-np.pi, b_max=np.pi, n_grid=25)
    else:
        x0 = _last_x0  # тёплый старт из прошлого оптимума

    res = refine_local(x0, obj)
    _last_x0 = res.x.copy()

    stop_time = time.time()
    print(f"Optimize phase loading time: {stop_time - start_time:.3f} sec")
    return res


def main():
    work_model = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE, SamplerTypes.SAMPLER_SOBOL)
    # common.plot_distribution(work_model.ds, num_params=len(work_model.ds_gen.origin_filter.coupling_matrix.links))
    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links]+["Q"] + ["f0"] + ["bw"] + ["a11"] + ["a22"] + ["b11"] + ["b22"])
                                                 # keys_for_analysis=[f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links])
    # codec.y_channels = ['S1_1.real', 'S1_2.real', 'S2_1.real', 'S2_2.real', 'S1_1.imag', 'S1_2.imag', 'S2_1.imag', 'S2_2.imag', 'S1_1.db', 'S1_2.db', 'S2_1.db', 'S2_2.db']
    codec = codec

    work_model.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )

    # synth_analyzer = TouchstoneDatasetAnalyzer(work_model.ds)
    # fig = synth_analyzer.plot_s_stats(
    #     port_out=2, port_in=1, metric='db', stats=['mean', 'std', 'min', 'max']
    # )
    # fig.suptitle("Статистика по синтетике S21 (дБ)")
    # fig = synth_analyzer.plot_s_stats(
    #     port_out=1, port_in=1, metric='db', stats=['mean', 'std', 'min', 'max']
    # )
    # fig.suptitle("Статистика по синтетике S11 (дБ)")
    #

    tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/24.10.25/non-shifted",
                            s_tf=S_Resample(301))

    # tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/narrowband",
    #                         s_tf=S_Resample(301))

    # meas_analyzer = TouchstoneDatasetAnalyzer(tds)
    # fig = meas_analyzer.plot_s_stats(
    #     port_out=2, port_in=1, metric='db', stats=['mean', 'std', 'min', 'max']
    # )
    # fig.suptitle("Статистика по измерениям S21 (дБ)")
    # fig = meas_analyzer.plot_s_stats(
    #     port_out=1, port_in=1, metric='db', stats=['mean', 'std', 'min', 'max']
    # )
    # fig.suptitle("Статистика по измерениям S11 (дБ)")

    # lit_model = work_model.train(
    #     optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
    #     scheduler_cfg={"name": "StepLR", "step_size": 25, "gamma": 0.09},
    #     # optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
    #     # scheduler_cfg={"name": "StepLR", "step_size": 30, "gamma": 0.01},
    #     loss_fn=CustomLosses("sqrt_mse_with_l1", weight_decay=1, weights=None)
    #     )

    # work_model_inference_extractor = common.WorkModel(configs.ENV_DATASET_PATH, 50000, SamplerTypes.SAMPLER_SOBOL)
    # work_model_inference_extractor.setup(
    #     model_name="resnet",
    #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # work_model_inference_extractor.inference("saved_models/EAMU4-KuIMUXT2-BPFC2/best-epoch=99-train_loss=0.20445-val_loss=0.32555-val_r2=0.69374-val_mse=0.02450-val_mae=0.08155-batch_size=32-base_dataset_size=50000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")

    # work_model_inference_main_extra = common.WorkModel(configs.ENV_DATASET_PATH, 300_000, SamplerTypes.SAMPLER_SOBOL)
    # work_model_wide = common.WorkModel(configs.ENV_DATASET_PATH, 1_000_000, SamplerTypes.SAMPLER_SOBOL)
    # work_model_wide_extra = common.WorkModel(configs.ENV_DATASET_PATH, 1_000_000, SamplerTypes.SAMPLER_SOBOL)

    # lit_model = work_model_inference_main.train(
    #     optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
    #     scheduler_cfg={"name": "StepLR", "step_size": 28, "gamma": 0.09},
    #     # optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
    #     # scheduler_cfg={"name": "StepLR", "step_size": 30, "gamma": 0.01},
    #     loss_fn=CustomLosses("sqrt_mse_with_l1", weight_decay=1, weights=None)
    #     )

    # work_model_inference_main_extra.setup(
    #     model_name="resnet_with_correction",
    #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # lit_model = work_model_inference_main_extra.train(
    #     optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
    #     scheduler_cfg={"name": "StepLR", "step_size": 28, "gamma": 0.09},
    #     # optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
    #     # scheduler_cfg={"name": "StepLR", "step_size": 30, "gamma": 0.01},
    #     loss_fn=CustomLosses("sqrt_mse_with_l1", weight_decay=1, weights=None)
    #     )


    # work_model_inference_main.model.main_model = work_model_inference_extractor.model

    # inf_model = work_model_inference_main_extra.inference(lit_model.trainer.checkpoint_callback.best_model_path)

    # work_model_wide.setup(
    #     model_name="resnet_with_wide_correction",
    #     model_cfg={"main_model": inf_model.model, "in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # # work_model.setup(
    # #     model_name="simple_opt",
    # #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    # #     dm_codec=codec
    # # )


    # lit_model = work_model_wide.train(
    #     optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
    #     scheduler_cfg={"name": "StepLR", "step_size": 28, "gamma": 0.09},
    #     # optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
    #     # scheduler_cfg={"name": "StepLR", "step_size": 40, "gamma": 0.01},
    #     loss_fn=CustomLosses("sqrt_mse_with_l1", weight_decay=1, weights=None)
    #     )

    # Загружаем лучшую модель
    # checkpoint_path="saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=12-val_loss=0.01266-train_loss=0.01224.ckpt",
    # checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=49-train_loss=0.03379-val_loss=0.03352-val_r2=0.84208-val_acc=0.25946-val_mae=0.05526-batch_size=32-dataset_size=100000.ckpt",
    # "saved_models/ERV-KuIMUXT1-BPFC1/best-epoch=22-train_loss=0.04863-val_loss=0.05762-val_r2=0.82785-val_mse=0.01366-val_mae=0.04395-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt"
    # inference_model = work_model.inference("saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=29-train_loss=0.04166-val_loss=0.04450-val_r2=0.92560-val_mse=0.00588-val_mae=0.03862-batch_size=32-base_dataset_size=1500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=29-train_loss=0.03546-val_loss=0.03841-val_r2=0.94190-val_mse=0.00459-val_mae=0.03381-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=34-train_loss=0.02388-val_loss=0.02641-val_r2=0.96637-val_mse=0.00265-val_mae=0.02376-batch_size=32-base_dataset_size=600000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=25-train_loss=0.02530-val_loss=0.02793-val_r2=0.96251-val_mse=0.00296-val_mae=0.02496-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\ERV-KuIMUXT1-BPFC1\\best-epoch=78-train_loss=0.01672-val_loss=0.02663-val_r2=0.96754-val_mse=0.00258-val_mae=0.02406-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\ERV-KuIMUXT1-BPFC1\\best-epoch=25-train_loss=0.01518-val_loss=0.01534-val_r2=0.89565-val_mse=0.00116-val_mae=0.01418-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_STD.ckpt")
    inference_model = work_model.inference("saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=26-train_loss=0.08889-val_loss=0.08248-val_r2=0.99727-val_mse=0.00020-val_mae=0.00647-batch_size=32-base_dataset_size=300000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models/EAMU4-KuIMUXT2-BPFC2/best-epoch=27-train_loss=0.08527-val_loss=0.07506-val_r2=0.99731-val_mse=0.00021-val_mae=0.00531-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference(lit_model.trainer.checkpoint_callback.best_model_path)


    # work_model_widenet = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE+100000, SamplerTypes.SAMPLER_SOBOL)
    # work_model_widenet.setup(
    #     model_name="resnet_with_wide_correction",
    #     model_cfg={"main_model": inference_model.model,
    #                "in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    #
    # lit_model_widenet = work_model_widenet.train(
    #     optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
    #     scheduler_cfg={"name": "StepLR", "step_size": 28, "gamma": 0.09},
    #     # optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
    #     # scheduler_cfg={"name": "StepLR", "step_size": 24, "gamma": 0.01},
    #     loss_fn=CustomLosses("mse_with_l1", weight_decay=1, weights=None)
    # )
    # inference_model = work_model_widenet.inference(lit_model_widenet.trainer.checkpoint_callback.best_model_path)

    # Предсказываем фильтр из тестового датасета
    # orig_fil, pred_fil = work_model.predict(inference_model, work_model.dm, idx=0)
    # Возьмем для примера первый touchstone-файл из тестового набора данных
    # test_tds = work_model.dm.get_dataset(split="test", meta=True)
    # for i in range(10):
    #     # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
    #     y_t, x_t, meta = test_tds[i]  # Используем первый файл набора данных]
    #
    #     # Декодируем данные
    #     orig_prms = work_model.dm.codec.decode_x(x_t)  # Создаем словарь параметров
    #     net = work_model.dm.codec.decode_s(y_t, meta)  # Создаем объект skrf.Network
    #
    #     # Предсказанные S-параметры
    #     pred_prms = inference_model.predict_x(net)
    #
    #     print(f"Исходные параметры: {orig_prms}")
    #     print(f"Предсказанные параметры: {pred_prms}")
    #     orig_fil = MWFilter.from_touchstone_dataset_item(({**meta['params'], **orig_prms}, net))
    #     pred_fil = work_model.create_filter_from_prediction(orig_fil, work_model.orig_filter, pred_prms, work_model.codec)
    #     ps_shifts = [pred_prms["a11"], pred_prms["a22"], pred_prms["b11"], pred_prms["b22"]]
    #     S11_corrected, S22_corrected, S21_corrected = phase.correct_phase_shift(pred_fil.f_norm, orig_fil.s[:, 0, 0],
    #                                                                             orig_fil.s[:, 1, 1], orig_fil.s[:, 1, 0],
    #                                                                             ps_shifts,
    #                                                                             plot=False)
    #     inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    #     plt.figure()
    #     plt.plot(pred_fil.f_norm, S11_corrected.real, pred_fil.f_norm, S11_corrected.imag)
    #     plt.plot(pred_fil.f_norm, pred_fil.s[:, 0, 0].real, pred_fil.f_norm, pred_fil.s[:, 0, 0].imag, linestyle=':')
    #     plt.legend(["s11 orig corr real", "s11 orig corr imag", "s11 pred real", "s11 pred imag"])
    #     plt.figure()
    #     plt.plot(pred_fil.f_norm, torch.angle(S11_corrected), pred_fil.f_norm,
    #              torch.angle(torch.tensor(pred_fil.s[:, 0, 0], dtype=torch.complex64)))
    #     plt.legend(["s11 orig corr phase", "s11 pred phase"])


        # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
        # optim_matrix, ps_opt = optimize_cm(pred_fil, orig_fil, fnorm_list=work_model.orig_filter.f_norm)
        # error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
        # error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
        #
        # error_matrix_optim.plot_matrix(title="Optimized de-tuned matrix errors", cmap="YlOrBr")
        # optim_matrix.plot_matrix(title="Optimized de-tuned matrix")
        # error_matrix_pred.plot_matrix(title="Predict de-tuned matrix errors", cmap="YlOrBr")
        # pred_fil.coupling_matrix.plot_matrix(title="Predict de-tuned matrix")
        # orig_fil.coupling_matrix.plot_matrix(title="Origin de-tuned matrix")


    # optim_matrix = optimize_cm(pred_fil, orig_fil)
    # error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    # error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    # error_matrix_optim.plot_matrix(title="Optimized tuned matrix errors for inverted model", cmap="YlOrBr")
    # error_matrix_pred.plot_matrix(title="Predict tuned matrix errors for inverted model", cmap="YlOrBr")
    # orig_fil.coupling_matrix.plot_matrix(title="Origin tuned matrix for inverted model")
    # pred_fil.coupling_matrix.plot_matrix(title="Predict tuned matrix for inverted model")
    # optim_matrix.plot_matrix(title="Optimized tuned matrix for inverted model")


    cst_tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/cst",
                            s_tf=S_Resample(301))
    # TODO: РЕФАКТОРИНГ МЕТОДОВ WORK_MODEL!!!!!
    phase_extractor = phase.PhaseLoadingExtractor(inference_model, work_model, work_model.orig_filter)
    for i in range(4, 7):
        w = work_model.orig_filter.f_norm
        orig_fil = tds[i][1]
        cst_fil = cst_tds[i][1]

        b11 = 2.5
        b22 = 0.1
        orig_fil.s[:, 0, 0] *= np.array(torch.exp(-1j*(w*b11)), dtype=np.complex64)
        orig_fil.s[:, 0, 1] *= np.array(torch.exp(-1j*0.5*(w*(b11+b22))), dtype=np.complex64)
        orig_fil.s[:, 1, 0] *= np.array(torch.exp(-1j*0.5*(w*(b11+b22))), dtype=np.complex64)
        orig_fil.s[:, 1, 1] *= np.array(torch.exp(-1j*(w*b22)), dtype=np.complex64)

        orig_fil_to_nn = copy.deepcopy(orig_fil)


        # plt.figure()
        # gvz = np.gradient(np.angle(orig_fil.s[:, 0, 0]), w)
        # plt.plot(w, gvz)
        # # plt.plot(w, min(gvz[0], gvz[-1])*np.ones_like(w))
        # # plt.title("ГВЗ S11")
        # plt.title(f"{tds.backend.paths[i].parts[-1]}. Значения на краях: {gvz[0]:.3f}, {gvz[-1]:.3f}")
        # print(f"For gvz: {gvz[0]:.3f}, {gvz[-1]:.3f}")

        phase_shifts, S_def = phase.fit_phase_edges_curvefit(
            w, orig_fil_to_nn,
            center_points=None,
            plot=False,
            verbose=False
        )
        orig_fil_to_nn.s = S_def
        orig_fil_to_nn_copy = copy.deepcopy(orig_fil_to_nn)
        phase_loadings = optimize_phase_loading(
            ntw_orig=orig_fil_to_nn_copy,
            inference_model=inference_model,
            work_model=work_model,
            reference_filter=work_model.orig_filter,
            w_norm=work_model.orig_filter.f_norm
        )
        print("Оптимальные фазовые параметры:", phase_loadings.x)
        b11, b22 = phase_loadings.x
        # b11 = 0.054
        # b22 = 0.052
        orig_fil_to_nn_copy.s[:, 0, 0] *= np.array(torch.exp(1j * (w * b11)), dtype=np.complex64)
        orig_fil_to_nn_copy.s[:, 0, 1] *= np.array(torch.exp(1j * 0.5 * (w * (b11 + b22))), dtype=np.complex64)
        orig_fil_to_nn_copy.s[:, 1, 0] *= np.array(torch.exp(1j * 0.5 * (w * (b11 + b22))), dtype=np.complex64)
        orig_fil_to_nn_copy.s[:, 1, 1] *= np.array(torch.exp(1j * (w * b22)), dtype=np.complex64)
        ntw_de = orig_fil_to_nn_copy
        # res = phase_extractor.extract_all(orig_fil_to_nn, w_norm=w)
        # ntw_de = res['ntw_deembedded']

        # # # start_time = time.time()
        pred_prms = inference_model.predict_x(ntw_de)
        # # stop_time = time.time()
        # # print(f"Predict time: {stop_time - start_time:.3f} sec")
        print(f"Предсказанные параметры: {pred_prms}")
        pred_fil = work_model.create_filter_from_prediction(orig_fil, work_model.orig_filter, pred_prms, work_model.codec)

        # plt.figure()
        # plt.plot(f_new, MWFilter.to_db(torch.tensor(s11_ext, dtype=torch.complex64)), label='S11 dB extr')
        # cst_fil.plot_s_db(m=0, n=0, label='S11 dB CST')
        # orig_fil.plot_s_db(m=0, n=0, label='S11 dB origin', ls='--')
        #
        # plt.figure()
        # plt.plot(f_new, MWFilter.to_db(torch.tensor(s22_ext, dtype=torch.complex64)), label='S22 dB extr')
        # cst_fil.plot_s_db(m=1, n=1, label='S22 dB CST')
        # orig_fil.plot_s_db(m=1, n=1, label='S22 dB origin', ls='--')
        # # ntw_de.plot_s_re(m=0, n=0, label='S11 Re corr', ls=':')


        plt.figure()
        cst_fil.plot_s_re(m=0, n=0, label='S11 Re CST')
        cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST')
        # orig_fil.plot_s_re(m=0, n=0, label='S11 Re origin', ls='--')
        ntw_de.plot_s_re(m=0, n=0, label='S11 Re corr', ls=':')
        ntw_de.plot_s_im(m=0, n=0, label='S11 Im corr', ls=':')
        # plt.plot(f_new, np.real(S11_ext), linestyle=':', label='S11 Re extr')

        plt.figure()
        cst_fil.plot_s_deg(m=0, n=0, label='S11 phase CST')
        # orig_fil.plot_s_deg(m=0, n=0, label='S11 phase origin', ls='--')
        ntw_de.plot_s_deg(m=0, n=0, label='S11 phase corr', ls=':')
        # ntw_de.plot_s_im(m=0, n=0, label='S11 Im corr')
        # plt.plot(f_new, np.degrees(np.angle(S11_ext)), linestyle=':', label='S11 phase extr')
        # cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST', ls='--')

        # plt.figure()
        # orig_fil_to_nn_copy.plot_s_re(m=1, n=1, label='S22 Re corr')
        # pred_fil.plot_s_re(m=1, n=1, label='S22 Re predict', ls=':')
        # orig_fil_to_nn_copy.plot_s_im(m=1, n=1, label='S22 Im corr')
        # pred_fil.plot_s_im(m=1, n=1, label='S22 Im predict', ls='--')

        ps_shifts = [pred_prms["a11"], pred_prms["a22"], pred_prms["b11"], pred_prms["b22"]]
        a11_final = phase_shifts['S11']['phi_c']+pred_prms["a11"]
        a22_final = phase_shifts['S22']['phi_c']+pred_prms["a22"]
        b11_final = phase_loadings.x[0]*0.5+pred_prms["b11"]
        b22_final = phase_loadings.x[1]*0.5+pred_prms["b22"]
        print(f"Финальный фазовый сдвиг, после корректировки ИИ: a11={a11_final:.6f} рад ({np.degrees(a11_final):.2f}°), a22={a22_final:.6f} рад ({np.degrees(a22_final):.2f}°)"
              f" b11 = {b11_final:.6f} рад ({np.degrees(b11_final):.2f}°), b22 = {b22_final:.6f} рад ({np.degrees(b22_final):.2f}°)")
        pred_fil.s[:, 0, 0] *= np.array(torch.exp(-1j*2*(a11_final + w*b11_final)), dtype=np.complex64)
        pred_fil.s[:, 0, 1] *= np.array(-torch.exp(-1j*(a11_final+a22_final + w*(b11_final+b22_final))), dtype=np.complex64)
        pred_fil.s[:, 1, 0] *= np.array(-torch.exp(-1j*(a11_final+a22_final + w*(b11_final+b22_final))), dtype=np.complex64)
        pred_fil.s[:, 1, 1] *= np.array(torch.exp(-1j*2*(a22_final + w*b22_final)), dtype=np.complex64)

        plt.figure(figsize=(4, 3))
        orig_fil.plot_s_re(m=0, n=0, label='S11 Re origin')
        pred_fil.plot_s_re(m=0, n=0, label='S11 Re predict', ls=':')
        orig_fil.plot_s_im(m=0, n=0, label='S11 Im origin')
        pred_fil.plot_s_im(m=0, n=0, label='S11 Im predict', ls='--')
        plt.title(tds.backend.paths[i].parts[-1])

        plt.figure(figsize=(4, 3))
        orig_fil.plot_s_deg(m=0, n=0, label='Phase S11 origin')
        pred_fil.plot_s_deg(m=0, n=0, label='Phase S11 predict')
        plt.title(tds.backend.paths[i].parts[-1])

        inference_model.plot_origin_vs_prediction(orig_fil, pred_fil, title=tds.backend.paths[i].parts[-1])
        optim_matrix, Q_opt, phase_opt = optimize_cm(pred_fil, orig_fil, phase_init=(0, 0, 0, 0))
        print(f"Оптимизированные параметры: {optim_matrix.factors}. Добротность: {Q_opt}. Фаза: {phase_opt}")
        # optim_matrix, Q, phase_opt = optimize_cm(pred_fil, orig_fil, phase_init=(a11_final, a22_final, b11_final, b22_final))
        # plt.title(tds.backend.paths[i].parts[-1])
        # print(f"Оптимизированные параметры: {optim_matrix.factors}. Добротность: {pred_fil.Q}. Фаза: {np.degrees(phase_opt)}")
        # optim_matrix.plot_matrix()

    # # Предсказываем эталонный фильтр
    # orig_fil = work_model.ds_gen.origin_filter
    # pred_prms = inference_model.predict_x(orig_fil)
    # pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, work_model.meta)
    # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    # optim_matrix = optimize_cm(pred_fil, orig_fil)
    # pred_fil.coupling_matrix.plot_matrix(title="Predict tuned matrix")
    # optim_matrix.plot_matrix(title="Optimized tuned matrix")
    # error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    # error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    # error_matrix_optim.plot_matrix(title="Optimized tuned matrix errors")
    # error_matrix_pred.plot_matrix(title="Predict tuned matrix errors")
    # orig_fil.coupling_matrix.plot_matrix(title="Origin tuned matrix")
    plt.show()


def plot_pl_csv_wide(csv_path, outdir="artifacts_opt3"):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Берём последнюю запись каждой эпохи
    by_epoch = df.groupby("epoch").last(numeric_only=True)

    epochs = by_epoch.index.values
    train_loss = by_epoch["train_loss_epoch"]
    val_loss = by_epoch["val_loss"]
    val_r2 = by_epoch["val_r2"]

    fig, ax1 = plt.subplots()

    # Первая ось — потери
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue", linestyle="--")
    ax1.plot(epochs, val_loss, label="Val Loss", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Вторая ось — R²
    ax2 = ax1.twinx()
    ax2.set_ylabel("R²", color="tab:green")
    ax2.plot(epochs, val_r2, label="Val R²", color="tab:green", linestyle=":")
    ax2.tick_params(axis='y', labelcolor="tab:green")

    # Легенда
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", frameon=True, fontsize=9)

    # plt.title("Метрики")
    fig.tight_layout()

    # plt.savefig(os.path.join(outdir, "loss_r2_curves.png"), dpi=200)
    # plt.savefig(os.path.join(outdir, "loss_r2_curves.svg"))
    plt.show()


if __name__ == "__main__":
    # csv_path = "lightning_logs/simple_opt_csv/version_0/metrics.csv"
    # plot_pl_csv_wide(csv_path)
    main()
    # online_correct()
    # inherence_correct()
    plt.show()
