import copy
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader

import cauchy_method
import cm_extract_api
import phase
import skrf as rf
import pandas as pd
# from utils import make_network
# from cm_extract_api import inference_model, work_model
from common import WorkModel
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
import configs as cfg
import common
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.nn.scalers import MinMaxScaler
from phase import PhaseLoadingExtractor
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")

def build_base_path(filter_name: str):
    base_path = f"filters/FilterData/{filter_name}/conf"
    return base_path

def build_work_model(*config_paths: str | os.PathLike, is_inference: bool):
    configs = cfg.Configs(*config_paths)
    wm = common.WorkModel(configs=configs, is_inference=is_inference)
    wm.setup(
        model_name=configs.APP_CONFIG.model.model_name,
        codec_type=configs.APP_CONFIG.model.codec,
    )
    return wm

def load_model(*config_paths: str | os.PathLike):
    wm = build_work_model(*config_paths, is_inference=True)
    inf_model = wm.inference(wm.configs.MODEL_CHECKPOINT_PATH)
    return wm, inf_model

def load_cm_extractor(filter_name: str):
    print(f"Загрузка cm_extractor")
    base_path = build_base_path(filter_name)
    return load_model(f"{base_path}/base.yml",
                          f"{base_path}/models/cm_extractor.yml")

def load_phase_freq_dep_extractor(filter_name: str):
    print(f"Загрузка phase_freq_dep_extractor")
    base_path = build_base_path(filter_name)
    return load_model(f"{base_path}/base.yml",
                          f"{base_path}/models/phase_freq_dep_extractor.yml")

def load_phase_const_extractor(filter_name: str):
    print(f"Загрузка phase_const_extractor")
    base_path = build_base_path(filter_name)
    return load_model(f"{base_path}/base.yml",
                          f"{base_path}/models/phase_const_extractor.yml")


def train_model(*config_paths: str | os.PathLike, optimizer_cfg: dict|None=None, scheduler_cfg: dict|None=None, loss_fn: nn.Module|None=None):
    wm = build_work_model(*config_paths, is_inference=False)
    optimizer_cfg = optimizer_cfg or {"name": "AdamW", "lr": 9.4e-4, "weight_decay": 1e-5}
    scheduler_cfg = scheduler_cfg or {"name": "StepLR", "step_size": 25, "gamma": 0.09}
    loss_fn = loss_fn or CustomLosses("sqrt_mse_with_l1", weight_decay=1, weights=None)

    lit_model = wm.train(
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        loss_fn=loss_fn,
        strategy_type=wm.configs.APP_CONFIG.train.strategy_type,
    )
    return wm, lit_model

def train_cm_extractor(filter_name: str):
    print(f"Обучение cm_extractor")
    base_path = build_base_path(filter_name)
    return train_model(f"{base_path}/base.yml",
                          f"{base_path}/models/cm_extractor.yml")

def train_phase_freq_dep_extractor(filter_name: str):
    print(f"Обучение phase_freq_dep_extractor")
    base_path = build_base_path(filter_name)
    return train_model(f"{base_path}/base.yml",
                          f"{base_path}/models/phase_freq_dep_extractor.yml")

def train_phase_const_extractor(filter_name: str):
    print(f"Обучение phase_const_extractor")
    base_path = build_base_path(filter_name)
    return train_model(f"{base_path}/base.yml",
                          f"{base_path}/models/phase_const_extractor.yml")



def main():
    FILTER_NAME = "EAMU4-KuIMUXT2-BPFC2"
    # 'S1_1.real', 'S1_2.real', 'S2_1.real', 'S2_2.real', 'S1_1.imag', 'S1_2.imag', 'S2_1.imag', 'S2_2.imag'
    # lit_model = cm_extract_api.train_model(os.path.join(configs.APP_CONFIG.base_dir, "manifest.yml"))

    # train_phase_freq_dep_extractor(FILTER_NAME)
    # train_phase_const_extractor(FILTER_NAME)

    wm_phase_const, inf_model_const = load_phase_const_extractor(FILTER_NAME)
    wm_phase_freq_dep, inf_model_freq_dep = load_phase_freq_dep_extractor(FILTER_NAME)
    wm_cm, inf_cm = load_cm_extractor(FILTER_NAME)

    tds = TouchstoneDataset(f"filters/FilterData/{FILTER_NAME}/measure/narrowband", s_tf=S_Resample(301))
    # tds = TouchstoneDataset(f"filters/FilterData/{FILTER_NAME}/measure/24.10.25/non-shifted", s_tf=S_Resample(301))
    # tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/19.02.26", s_tf=S_Resample(301))
    cst_tds = TouchstoneDataset(f"filters/FilterData/{FILTER_NAME}/measure/cst",
                                s_tf=S_Resample(301))
    # TODO: РЕФАКТОРИНГ МЕТОДОВ WORK_MODEL!!!!!
    # phase_extractor = phase.PhaseLoadingExtractor(inference_model_phase_coarse, api_phase_coarse.work_model, api_phase_coarse.work_model.orig_filter)
    #
    # codec = api_phase_coarse.codec
    # loss = nn.L1Loss()
    # codec_db = copy.deepcopy(codec)
    # codec_db.y_channels = ['S1_1.db', 'S2_1.db', 'S2_2.db']
    #
    # losses = []

    # for i in range(0, 5):
    #     inference_model_phase_coarse.predict_for(api_phase_fine.work_model.dm, i)

    for i in range(0, len(tds)):
        w = wm_phase_const.orig_filter.f_norm
        orig_fil = tds[i][1]
        # cst_fil = cst_tds[i][1]

        # Тестовое добавление фазового сдвига
        orig_fil = PhaseLoadingExtractor.add_phase_from_coeffs(orig_fil, a11=0, a22=0, b11=0, b22=0, w=w)

        # plt.figure(figsize=(5, 3))
        # orig_fil.plot_s_deg(m=0, n=0, label='S11 phase')
        # plt.title("Смещенная на pi фаза")

        orig_fil_to_nn = copy.deepcopy(orig_fil)
        orig_fil_to_nn.s[:, 1, 0] *= -1
        orig_fil_to_nn.s[:, 0, 1] *= -1

        pred_prms = inf_model_freq_dep.predict_x(orig_fil_to_nn)
        total_results = {'a11' : 0, 'a22': 0, 'b11': 0, 'b22': 0}
        total_results['b11'] += pred_prms['b11']
        total_results['b22'] += pred_prms['b22']
        pred_prms.update({'a11':0, 'a22':0})
        net_de = PhaseLoadingExtractor.remove_phase_from_coeffs(orig_fil_to_nn, w, **pred_prms)
        # print(f"[{i}] Предсказание частотно-зависимой составляющей: {pred_prms}")
        for _ in range(0, 10):
            pred_prms = inf_model_freq_dep.predict_x(net_de)
            total_results['b11'] += pred_prms['b11']
            total_results['b22'] += pred_prms['b22']
            pred_prms.update({'a11': 0, 'a22': 0})
            net_de = PhaseLoadingExtractor.remove_phase_from_coeffs(orig_fil_to_nn, w, **pred_prms)
            # print(f"[{i}] Предсказание частотно-зависимой составляющей: {pred_prms}")
        # print(f"[{i}] Предсказание частотно-зависимой составляющей: {total_results}")

        pred_prms = inf_model_const.predict_x(net_de)
        pred_prms.update({'b11': 0, 'b22': 0})
        total_results['a11'] += pred_prms['a11']
        total_results['a22'] += pred_prms['a22']
        # print(f"[{i}] Предсказание постоянной составляющей: {pred_prms}")
        # net_de = PhaseLoadingExtractor.remove_phase_from_coeffs(net_de, w, **pred_prms)
        for _ in range(0, 10):
            pred_prms = inf_model_const.predict_x(net_de)
            total_results['a11'] += pred_prms['a11']
            total_results['a22'] += pred_prms['a22']
            pred_prms.update({'b11': 0, 'b22': 0})
            # total_results['a11'] += tuned_prms['a11']
            # total_results['a22'] += tuned_prms['a22']
            net_de = PhaseLoadingExtractor.remove_phase_from_coeffs(net_de, w, **pred_prms)
        print(f"[{i}] Предсказание фазовых коэффициентов: {total_results}")

        # tuned_prms['b11'] += pred_prms['b11']
        # tuned_prms['b22'] += pred_prms['b22']
        # print(f"Итоговые параметры: {tuned_prms}")

        # plt.figure(figsize=(5, 3))
        # cst_fil.plot_s_re(m=0, n=0, label='S11 Re CST')
        # cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST')
        # net_de.plot_s_re(m=0, n=0, label='S11 Re corr', ls=':')
        # net_de.plot_s_im(m=0, n=0, label='S11 Im corr', ls=':')
        # plt.title(tds.backend.paths[i].parts[-1])

        # plt.figure(figsize=(5, 3))
        # cst_fil.plot_s_re(m=1, n=0, label='S21 Re CST')
        # cst_fil.plot_s_im(m=1, n=0, label='S21 Im CST')
        # net_de.plot_s_re(m=1, n=0, label='S21 Re corr', ls=':')
        # net_de.plot_s_im(m=1, n=0, label='S21 Im corr', ls=':')

        # # plt.figure()
        # # gvz = np.gradient(np.angle(orig_fil.s[:, 0, 0]), w)
        # # plt.plot(w, gvz)
        # # # plt.plot(w, min(gvz[0], gvz[-1])*np.ones_like(w))
        # # # plt.title("ГВЗ S11")
        # # plt.title(f"{tds.backend.paths[i].parts[-1]}. Значения на краях: {gvz[0]:.3f}, {gvz[-1]:.3f}")
        # # print(f"For gvz: {gvz[0]:.3f}, {gvz[-1]:.3f}")
        #
        # # phase_shifts, S_def = phase.fit_phase_edges_curvefit(
        # #     w, orig_fil_to_nn,
        # #     center_points=None,
        # #     plot=False,
        # #     verbose=False
        # # )
        # # orig_fil_to_nn.s = S_def
        # # orig_fil_to_nn_copy = copy.deepcopy(orig_fil_to_nn)
        # # phase_loadings = optimize_phase_loading(
        # #     ntw_orig=orig_fil_to_nn_copy,
        # #     inference_model=inference_model,
        # #     work_model=work_model,
        # #     reference_filter=work_model.orig_filter,
        # #     w_norm=work_model.orig_filter.f_norm
        # # )
        # # print("Оптимальные фазовые параметры:", phase_loadings.x)
        # # b11, b22 = phase_loadings.x
        # # # b11 = 0.054
        # # # b22 = 0.052
        # # orig_fil_to_nn_copy.s[:, 0, 0] *= np.array(torch.exp(1j * (w * b11)), dtype=np.complex64)
        # # orig_fil_to_nn_copy.s[:, 0, 1] *= np.array(torch.exp(1j * 0.5 * (w * (b11 + b22))), dtype=np.complex64)
        # # orig_fil_to_nn_copy.s[:, 1, 0] *= np.array(torch.exp(1j * 0.5 * (w * (b11 + b22))), dtype=np.complex64)
        # # orig_fil_to_nn_copy.s[:, 1, 1] *= np.array(torch.exp(1j * (w * b22)), dtype=np.complex64)
        # # ntw_de = orig_fil_to_nn_copy
        # res = phase_extractor.extract_all(orig_fil_to_nn, w_norm=w, verbose=True, plot_edges=False)
        # ntw_de = res['ntw_deembedded']
        #
        # ts = TouchstoneData(ntw_de)
        # s = codec.encode(ts)[1].unsqueeze(0)
        # s_db = codec_db.encode(ts)[1].unsqueeze(0)
        #
        # # # # start_time = time.time()
        pred_prms = inf_cm.predict_x(net_de)
        # # # stop_time = time.time()
        # # # print(f"Predict time: {stop_time - start_time:.3f} sec")
        print(f"Предсказанные параметры: {pred_prms}")
        pred_fil = wm_cm.create_filter_from_prediction(orig_fil, wm_cm.orig_filter, pred_prms)
        # # pred_fil.coupling_matrix.plot_matrix()
        # l = loss(s_db, codec_db.encode(TouchstoneData(pred_fil))[1].unsqueeze(0))
        # losses.append(l)
        # print(f"[{i}] Recover S-parameters loss: {l}")
        #
        #
        # # plt.figure()
        # # cst_fil.plot_s_re(m=0, n=0, label='S11 Re CST')
        # # cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST')
        # # orig_fil.plot_s_re(m=0, n=0, label='S11 Re origin', ls='--')
        # # ntw_de.plot_s_re(m=0, n=0, label='S11 Re corr', ls=':')
        # # ntw_de.plot_s_im(m=0, n=0, label='S11 Im corr', ls=':')
        # # plt.plot(f_new, np.real(S11_ext), linestyle=':', label='S11 Re extr')
        #
        # # plt.figure()
        # # cst_fil.plot_s_deg(m=0, n=0, label='S11 phase CST')
        # # orig_fil.plot_s_deg(m=0, n=0, label='S11 phase origin', ls='--')
        # # ntw_de.plot_s_re(m=0, n=0, label='S11 Re corr', ls=':')
        # # ntw_de.plot_s_im(m=0, n=0, label='S11 Im corr')
        # # plt.plot(f_new, np.degrees(np.angle(S11_ext)), linestyle=':', label='S11 phase extr')
        # # cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST', ls='--')
        #
        # # plt.figure()
        # # orig_fil_to_nn_copy.plot_s_re(m=1, n=1, label='S22 Re corr')
        # # pred_fil.plot_s_re(m=1, n=1, label='S22 Re predict', ls=':')
        # # orig_fil_to_nn_copy.plot_s_im(m=1, n=1, label='S22 Im corr')
        # # pred_fil.plot_s_im(m=1, n=1, label='S22 Im predict', ls='--')
        #
        # # ps_shifts = [pred_prms["a11"], pred_prms["a22"], pred_prms["b11"], pred_prms["b22"]]
        # a11_final = res['phi1_c'] + pred_prms["a11"]
        # a22_final = res['phi2_c'] + pred_prms["a22"]
        # b11_final = pred_prms["b11"] + res['b11_opt']
        # b22_final = pred_prms["b22"] + res['b22_opt']
        # print(
        #     f"Финальный фазовый сдвиг, после корректировки ИИ: a11={a11_final:.6f} рад ({np.degrees(a11_final):.2f}°), a22={a22_final:.6f} рад ({np.degrees(a22_final):.2f}°)"
        #     f" b11 = {b11_final:.6f} рад ({np.degrees(b11_final):.2f}°), b22 = {b22_final:.6f} рад ({np.degrees(b22_final):.2f}°)")
        # pred_fil.s[:, 0, 0] *= np.array(torch.exp(-1j * 2 * (a11_final + w * b11_final)), dtype=np.complex64)
        # pred_fil.s[:, 0, 1] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))),
        #                                 dtype=np.complex64)
        # pred_fil.s[:, 1, 0] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))),
        #                                 dtype=np.complex64)
        # pred_fil.s[:, 1, 1] *= np.array(torch.exp(-1j * 2 * (a22_final + w * b22_final)), dtype=np.complex64)
        # #
        # # plt.figure(figsize=(4, 3))
        # # orig_fil.plot_s_re(m=0, n=0, label='S11 Re origin')
        # # pred_fil.plot_s_re(m=0, n=0, label='S11 Re predict', ls=':')
        # # orig_fil.plot_s_im(m=0, n=0, label='S11 Im origin')
        # # pred_fil.plot_s_im(m=0, n=0, label='S11 Im predict', ls='--')
        # # plt.title(tds.backend.paths[i].parts[-1])
        # #
        # # plt.figure(figsize=(4, 3))
        # # orig_fil.plot_s_deg(m=0, n=0, label='Phase S11 origin')
        # # pred_fil.plot_s_deg(m=0, n=0, label='Phase S11 predict')
        # # plt.title(tds.backend.paths[i].parts[-1])
        #
        inf_cm.plot_origin_vs_prediction(orig_fil, pred_fil, title=f"Origin tune: {tds.backend.paths[i].parts[-1]}")
        # # fine_tuned_model.plot_origin_vs_prediction(orig_fil, tuned_fil, title=f"Fine tune: {tds.backend.paths[i].parts[-1]}")
        #
        # optim_filter, phase_opt = optimize_cm(pred_fil, orig_fil, phase_init=(a11_final, a22_final, b11_final, b22_final))
        # plt.title(tds.backend.paths[i].parts[-1])
        # print(f"Оптимизированные параметры: {optim_filter.coupling_matrix.factors}. Добротность: {pred_fil.Q}. Фаза: {np.degrees(phase_opt)}")
        # # optim_filter.coupling_matrix.plot_matrix()


    # Предсказываем эталонный фильтр
    # orig_fil = work_model.ds_gen.origin_filter
    # w = orig_fil.f_norm
    # res = phase_extractor.extract_all(orig_fil, w_norm=w, verbose=True)
    # ntw_de = res['ntw_deembedded']
    #
    # pred_prms = inference_model.predict_x(ntw_de)
    # print(f"Предсказанные параметры: {pred_prms}")
    # pred_fil = work_model.create_filter_from_prediction(orig_fil, work_model.orig_filter, pred_prms)
    # a11_final = res['phi1_c'] + pred_prms["a11"]
    # a22_final = res['phi2_c'] + pred_prms["a22"]
    # b11_final = pred_prms["b11"] + res['b11_opt']
    # b22_final = pred_prms["b22"] + res['b22_opt']
    # print(
    #     f"Финальный фазовый сдвиг, после корректировки ИИ: a11={a11_final:.6f} рад ({np.degrees(a11_final):.2f}°), a22={a22_final:.6f} рад ({np.degrees(a22_final):.2f}°)"
    #     f" b11 = {b11_final:.6f} рад ({np.degrees(b11_final):.2f}°), b22 = {b22_final:.6f} рад ({np.degrees(b22_final):.2f}°)")
    # pred_fil.s[:, 0, 0] *= np.array(torch.exp(-1j * 2 * (a11_final + w * b11_final)), dtype=np.complex64)
    # pred_fil.s[:, 0, 1] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))),
    #                                 dtype=np.complex64)
    # pred_fil.s[:, 1, 0] *= np.array(-torch.exp(-1j * (a11_final + a22_final + w * (b11_final + b22_final))),
    #                                 dtype=np.complex64)
    # pred_fil.s[:, 1, 1] *= np.array(torch.exp(-1j * 2 * (a22_final + w * b22_final)), dtype=np.complex64)
    #
    # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil,
    #                                           title=f"Origin tune: Origin Filter")
    # optim_filter, phase_opt = optimize_cm(pred_fil, orig_fil, phase_init=(a11_final, a22_final, b11_final, b22_final))
    # plt.title("Optimized: Origin Filter")
    # print(
    #     f"Оптимизированные параметры: {optim_filter.coupling_matrix.factors}. Добротность: {pred_fil.Q}. Фаза: {np.degrees(phase_opt)}")
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
    plt.show()
