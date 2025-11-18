import copy
import os
import random
import time
from pathlib import Path

import numpy as np

import cauchy_method
import phase

import pandas as pd

from cm_extract_api import inference_model
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
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



import numpy as np
import matplotlib.pyplot as plt

# ---------- вспомогательные ----------
def _poly_design_no_affine(w, deg=4):
    w = np.asarray(w, float)
    if deg < 2:
        return np.zeros((w.size, 0), float)
    wc = (w - w.mean()) / (np.max(np.abs(w - w.mean())) + 1e-12)
    cols = [wc**k for k in range(2, deg+1)]
    return np.column_stack(cols)

def _unwrap_phase(z):
    return np.unwrap(np.angle(z))

def _center_series(y, w, center_at):
    if center_at is None:
        return y
    if center_at == 'mid':
        idx = len(w)//2
    else:
        idx = int(np.argmin(np.abs(w - float(center_at))))
    return y - y[idx]

# ---------- Оценка фаз + построение аппроксимации и метрик ----------
def phase_fit_quality(w, Sii, deg=4, center_at='mid', name='Sii'):
    """
    Строит МНК-аппроксимацию фазы Sii: φ ≈ -2(a + b w) + sum_{k>=2} c_k w^k
    Возвращает словарь с (a,b,coeffs,phi_fit,metrics) и рисует фазу/остатки.
    """
    w = np.asarray(w, float).ravel()
    phi = _unwrap_phase(Sii)                                 # рад
    F = w.size
    X_ab = np.column_stack([np.ones(F), w])                  # [F,2] для (-2a, -2b)
    X_hi = _poly_design_no_affine(w, deg=deg)                # [F,deg-1]
    X    = np.column_stack([X_ab, X_hi])                     # полный дизайн
    beta, *_ = np.linalg.lstsq(X, phi, rcond=None)           # МНК
    a = -0.5*beta[0]; b = -0.5*beta[1]
    coeffs = beta[2:]                                        # кв. и выше
    phi_fit = X @ beta                                       # рад

    # метрики в градусах
    err = (phi - phi_fit)                      # рад
    err_deg = np.degrees(err)
    mae = float(np.mean(np.abs(err_deg)))
    rmse = float(np.sqrt(np.mean(err_deg**2)))
    # R^2 по рад
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((phi - np.mean(phi))**2)) + 1e-12
    r2 = 1.0 - ss_res/ss_tot

    # плот «фаза+аппроксимация» и «остатки»
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), dpi=120, sharex=True)
    phi_c     = _center_series(phi, w, center_at)
    phi_fit_c = _center_series(phi_fit, w, center_at)
    ax1.plot(w, np.degrees(phi_c), label='фаза (unwrapped)', lw=1.6)
    ax1.plot(w, np.degrees(phi_fit_c), '--', label=f'аппроксимация (deg={deg})', lw=1.6)
    ax1.set_ylabel('φ (°)')
    ax1.set_title(f'{name}: фаза vs аппроксимация  |  MAE={mae:.2f}°, RMSE={rmse:.2f}°, R²={r2:.4f}')
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.plot(w, err_deg, lw=1.2)
    ax2.axhline(0, color='k', alpha=0.3)
    ax2.set_xlabel('Нормированная частота ω')
    ax2.set_ylabel('Остаток φ (°)')
    ax2.set_title(f'{name}: остатки аппроксимации')
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()

    return {
        'a': a, 'b': b, 'coeffs': coeffs, 'phi_fit': phi_fit,
        'metrics': {'MAE_deg': mae, 'RMSE_deg': rmse, 'R2': r2}
    }

# ---------- прежние функции деэмбединга (без весов) ----------
def estimate_port_phase_poly(w, S11, S22, deg=4):
    w = np.asarray(w, float).ravel()
    phi11 = _unwrap_phase(S11)
    phi22 = _unwrap_phase(S22)
    F = w.size
    X_ab = np.column_stack([np.ones(F), w])
    X_hi = _poly_design_no_affine(w, deg=deg)

    def fit_one(phi):
        X = np.column_stack([X_ab, X_hi])
        beta, *_ = np.linalg.lstsq(X, phi, rcond=None)
        return (-0.5*beta[0], -0.5*beta[1])  # (a,b)

    a1, b1 = fit_one(phi11)
    a2, b2 = fit_one(phi22)
    return a1, a2, b1, b2

def apply_deembedding(w, S, a1, a2, b1, b2):
    w = np.asarray(w, float).ravel()
    phi1 = a1 + b1*w
    phi2 = a2 + b2*w
    f11 = np.exp(1j*2*phi1)
    f22 = np.exp(1j*2*phi2)
    f21 = np.exp(1j*(phi1+phi2))
    Sout = S.copy()
    Sout[:,0,0] *= f11
    Sout[:,1,1] *= f22
    Sout[:,0,1] *= f21
    Sout[:,1,0] *= f21
    return Sout

# ---------- обновление: добавляем сравнение аппроксимации внутрь общего пайплайна ----------
def plot_deembedding_phases(w, S_meas, deg=4,
                            center_at=None,
                            show_unwrapped=True,     # <— НОВОЕ: вкл/выкл unwrapped-фигуры
                            show_group_delay=True,
                            show_wrapped=False,
                            show_real_imag=True,
                            show_fit_quality=True):
    """
    w      : [F] нормированная частота ω
    S_meas : [F,2,2] complex — исходные S-параметры
    deg    : степень полинома для «внутренней» формы (>=2)
    center_at : None | 'mid' | float(ω) — вычитание константы фазы (для наглядности)
    show_unwrapped   : рисовать ли unwrapped-фазы S11/S21/S22 (до/после)
    show_group_delay : рисовать τ для S21 (норм.) = -dφ/dω
    show_wrapped     : дополнительно показать wrapped-фазы
    show_real_imag   : показать графики Re/Im до/после
    show_fit_quality : показать качество полиномиальной аппроксимации фаз S11/S22
    Возвращает: (a1, a2, b1, b2, S_def, fit_info)
      fit_info = {'S11': {...}, 'S22': {...}} если show_fit_quality=True, иначе {}
    """
    S_meas = np.asarray(S_meas)
    w = np.asarray(w, float).ravel()
    assert S_meas.shape[-2:] == (2,2), "Ожидается S:[F,2,2]"

    # --- извлечь фазы, при необходимости показать качество подгонки ---
    S11 = S_meas[:,0,0]
    S22 = S_meas[:,1,1]

    fit_info = {}
    if show_fit_quality:
        fit11 = phase_fit_quality(w, S11, deg=deg, center_at=center_at, name='S11')
        fit22 = phase_fit_quality(w, S22, deg=deg, center_at=center_at, name='S22')
        fit_info['S11'] = fit11
        fit_info['S22'] = fit22
        a1, b1 = fit11['a'], fit11['b']
        a2, b2 = fit22['a'], fit22['b']
    else:
        a1, a2, b1, b2 = estimate_port_phase_poly(w, S11, S22, deg=deg)

    # --- деэмбеддинг ---
    S_def = apply_deembedding(w, S_meas, a1, a2, b1, b2)

    # --- утилиты отображения ---
    def _center_series(y):
        if center_at is None:
            return y
        if center_at == 'mid':
            idx = len(w)//2
        else:
            idx = int(np.argmin(np.abs(w - float(center_at))))
        return y - y[idx]

    def _unwrap_phase_arr(z):
        return np.unwrap(np.angle(z))

    # --- (опционально) unwrapped фазы: S11, S21, S22 ---
    if show_unwrapped:
        def _plot_phase(ax, z_before, z_after, title):
            ph_b = _center_series(_unwrap_phase_arr(z_before))
            ph_a = _center_series(_unwrap_phase_arr(z_after))
            ax.plot(w, np.degrees(ph_b), label='до', lw=1.6)
            ax.plot(w, np.degrees(ph_a), '--', label='после', lw=1.6)
            ax.set_ylabel('φ (°)'); ax.set_title(title); ax.grid(True, alpha=0.25)

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), dpi=120, sharex=True)
        _plot_phase(axes[0], S_meas[:,0,0], S_def[:,0,0], 'Фаза S11 (unwrapped)')
        _plot_phase(axes[1], S_meas[:,0,1], S_def[:,0,1], 'Фаза S21 (unwrapped)')
        _plot_phase(axes[2], S_meas[:,1,1], S_def[:,1,1], 'Фаза S22 (unwrapped)')
        axes[-1].set_xlabel('Нормированная частота ω'); axes[0].legend()
        fig.suptitle('До/после деэмбединга фазы', y=0.98); fig.tight_layout()

    # --- (опц.) group delay (норм.) для S21 ---
    if show_group_delay:
        phi_b = _unwrap_phase_arr(S_meas[:,0,1]); phi_a = _unwrap_phase_arr(S_def[:,0,1])
        tau_b = -np.gradient(phi_b, w);           tau_a = -np.gradient(phi_a, w)
        fig2, ax2 = plt.subplots(figsize=(9, 4), dpi=120)
        ax2.plot(w, tau_b, label='до', lw=1.6)
        ax2.plot(w, tau_a, '--', label='после', lw=1.6)
        ax2.set_xlabel('Нормированная частота ω'); ax2.set_ylabel('Групповая задержка (норм.)')
        ax2.set_title('Групповая задержка S21 (до/после)')
        ax2.grid(True, alpha=0.25); ax2.legend(); fig2.tight_layout()

    # --- (опц.) wrapped фазы ---
    if show_wrapped:
        def _wr(z):
            ph = np.angle(z)
            if center_at is not None:
                idx = len(w)//2 if center_at == 'mid' else int(np.argmin(np.abs(w - float(center_at))))
                ph = np.angle(np.exp(1j*(ph - ph[idx])))
            return np.degrees(ph)
        fig3, axes3 = plt.subplots(3, 1, figsize=(9, 8), dpi=120, sharex=True)
        axes3[0].plot(w, _wr(S_meas[:,0,0]), label='до'); axes3[0].plot(w, _wr(S_def[:,0,0]), '--', label='после'); axes3[0].set_title('S11 (wrapped)')
        axes3[1].plot(w, _wr(S_meas[:,0,1]), label='до'); axes3[1].plot(w, _wr(S_def[:,0,1]), '--', label='после'); axes3[1].set_title('S21 (wrapped)')
        axes3[2].plot(w, _wr(S_meas[:,1,1]), label='до'); axes3[2].plot(w, _wr(S_def[:,1,1]), '--', label='после'); axes3[2].set_xlabel('Нормированная частота ω'); axes3[2].set_title('S22 (wrapped)')
        axes3[0].legend(); fig3.tight_layout()

    # --- (опц.) Re/Im до/после ---
    if show_real_imag:
        fig4, axes4 = plt.subplots(3, 2, figsize=(11, 8), dpi=120, sharex=True)
        names = [('S11', (0,0)), ('S21', (0,1)), ('S22', (1,1))]
        for r, (nm, ij) in enumerate(names):
            zb = S_meas[:, ij[0], ij[1]]; za = S_def[:, ij[0], ij[1]]
            ax = axes4[r, 0]; ax.plot(w, zb.real, label='до', lw=1.6); ax.plot(w, za.real, '--', label='после', lw=1.6)
            ax.set_ylabel(f'{nm} — Re'); ax.grid(True, alpha=0.25);
            if r == 0: ax.set_title('Действительная часть')
            ax = axes4[r, 1]; ax.plot(w, zb.imag, label='до', lw=1.6); ax.plot(w, za.imag, '--', label='после', lw=1.6)
            ax.grid(True, alpha=0.25);
            if r == 0: ax.set_title('Мнимая часть')
        axes4[-1, 0].set_xlabel('Нормированная частота ω'); axes4[-1, 1].set_xlabel('Нормированная частота ω')
        axes4[0, 0].legend(loc='best'); fig4.suptitle('Re/Im S-параметров: до и после деэмбединга', y=0.98); fig4.tight_layout()

    return a1, a2, b1, b2, S_def, fit_info



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
    inference_model = work_model.inference("saved_models/EAMU4-KuIMUXT2-BPFC4/best-epoch=28-train_loss=0.08384-val_loss=0.06965-val_r2=0.99855-val_mse=0.00011-val_mae=0.00464-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
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


    for i in range(0, len(tds)):
        # i = random.randint(0, len(tds))
        w = work_model.orig_filter.f_norm
        orig_fil = tds[i][1]
        orig_fil_to_nn = copy.deepcopy(orig_fil)
        w_ext, s11_ext, s21_ext = cauchy_method.extract_coeffs(freq=orig_fil.f / 1e6, Q=work_model.orig_filter.Q, f0=work_model.orig_filter.f0,
                                     s11=orig_fil_to_nn.s[:, 0, 0], s21=-orig_fil_to_nn.s[:, 1, 0],
                                     N=work_model.orig_filter.order,
                                     nz=8, bw=work_model.orig_filter.bw)
        # n_points = len(w_ext)
        # s_new = np.zeros((n_points, 2, 2), dtype=complex)
        #
        # s_new[:, 0, 0] = s11_ext
        # s_new[:, 1, 0] = s21_ext
        # s_new[:, 0, 1] = s21_ext  # если считаешь сеть взаимной
        # s_new[:, 1, 1] = s11_ext  # либо возьми из аппроксимации/старой сети
        #
        # w = w_ext
        # orig_fil_to_nn.s = s_new
        # orig_fil_to_nn.f = MWFilter.nfreq_to_freq(w, work_model.orig_filter.f0, work_model.orig_filter.bw)*1e6

        eps_mu = 1/3e8
        phase.estimate_phi0_dl_wrapped_from_vectors(
            f_hz=orig_fil_to_nn.f,
            S_vec=orig_fil_to_nn.s[:,1,1],
            w_norm=w,
            w0=1.42,
            eps_mu=eps_mu
        )
        # phi0_11, dl_11, phi_corr_11, S11_corr = phase.estimate_phi0_dl_wrapped_from_vectors(
        #     1.42, orig_fil_to_nn.f, orig_fil_to_nn.s[:, 0, 0], eps_mu, label='S11', plot=True
        # )
        # 
        # phi0_22, dl_22, phi_corr_22, S22_corr = phase.estimate_phi0_dl_wrapped_from_vectors(
        #     1.42, orig_fil_to_nn.f, orig_fil_to_nn.s[:, 1, 1], eps_mu, label='S22', plot=True
        # )

        # res, S_def = phase.fit_phase_edges_curvefit(
        #     w, orig_fil_to_nn,
        #     # q=0.35,
        #     # center_points=(-1.42, 1.42),
        #     center_points=(-3, 3),
        #     correct_freq_dependence=True,
        #     fit_on_extrapolated=True,
        #     plot=True
        # )
        # orig_fil_to_nn.s = S_def


        # # # start_time = time.time()
        # pred_prms = inference_model.predict_x(orig_fil_to_nn)
        # # stop_time = time.time()
        # # print(f"Predict time: {stop_time - start_time:.3f} sec")
        # print(f"Предсказанные параметры: {pred_prms}")
        # pred_fil = work_model.create_filter_from_prediction(orig_fil, work_model.orig_filter, pred_prms, work_model.codec)
        # ps_shifts = [pred_prms["a11"], pred_prms["a22"], pred_prms["b11"], pred_prms["b22"]]
        # a11_final = res['S11']['phi_c']+pred_prms["a11"]
        # a22_final = res['S22']['phi_c']+pred_prms["a22"]
        # b11_final = pred_prms["b11"]+res['S11']['phi_b']
        # b22_final = pred_prms["b22"]+res['S22']['phi_b']
        # print(f"Финальный фазовый сдвиг, после корректировки ИИ: a11={a11_final:.6f} рад ({np.degrees(a11_final):.2f}°), a22={a22_final:.6f} рад ({np.degrees(a22_final):.2f}°)"
        #       f" b11 = {b11_final:.6f} рад ({np.degrees(b11_final):.2f}°), b22 = {b22_final:.6f} рад ({np.degrees(b22_final):.2f}°)")
        # pred_fil.s[:, 0, 0] *= np.array(torch.exp(-1j*2*(a11_final + w*b11_final)), dtype=np.complex64)
        # pred_fil.s[:, 0, 1] *= np.array(-torch.exp(-1j*(a11_final+a22_final + w*(b11_final+b22_final))), dtype=np.complex64)
        # pred_fil.s[:, 1, 0] *= np.array(-torch.exp(-1j*(a11_final+a22_final + w*(b11_final+b22_final))), dtype=np.complex64)
        # pred_fil.s[:, 1, 1] *= np.array(torch.exp(-1j*2*(a22_final + w*b22_final)), dtype=np.complex64)

        # plt.figure()
        # orig_fil.plot_s_re(m=0, n=0, label='S11 Re origin')
        # pred_fil.plot_s_re(m=0, n=0, label='S11 Re predict', ls=':')
        # orig_fil.plot_s_im(m=0, n=0, label='S11 Im origin')
        # pred_fil.plot_s_im(m=0, n=0, label='S11 Im predict', ls='--')
        # plt.title(tds.backend.paths[i].parts[-1])

        # plt.figure()
        # orig_fil.plot_s_deg(m=0, n=0, label='Phase S11 origin')
        # pred_fil.plot_s_deg(m=0, n=0, label='Phase S11 predict')
        # plt.title(tds.backend.paths[i].parts[-1])
        # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil, title=tds.backend.paths[i].parts[-1])
        # optim_matrix = optimize_cm(pred_fil, orig_fil)
        # print(f"Оптимизированные параметры: {optim_matrix.factors}. Добротность: {pred_fil.Q}. Фаза:")
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
