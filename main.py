import copy
import os
import random
import time
from pathlib import Path

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

torch.set_float32_matmul_precision("medium")


class CorrectionNet(nn.Module):
    def __init__(self, s_shape, m_dim, hidden_dim=512):
        super().__init__()
        s_dim = s_shape[0] * s_shape[1]  # 301 * 8 = 2408 or 301 * 4 = 1204
        self.fc1 = nn.Linear(s_dim + m_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, m_dim)

    def forward(self, s_params, matrix_pred):
        s_params = s_params.view(s_params.size(0), -1)  # B x 2408 or B x 1204
        x = torch.cat([s_params, matrix_pred], dim=1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        delta = self.fc3(x)
        return matrix_pred + delta


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
    work_model_inference = common.WorkModel(configs.ENV_DATASET_PATH, 50000, SamplerTypes.SAMPLER_SOBOL)
    work_model_inference.setup(
        model_name="resnet",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )
    work_model_inference.inference(
        "saved_models\\ERV-KuIMUXT1-BPFC1\\best-epoch=29-train_loss=0.07467-val_loss=0.08013-val_r2=0.79334-val_mse=0.01666-val_mae=0.06346-batch_size=32-base_dataset_size=50000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    work_model.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )
    work_model.model.main_model = work_model_inference.model

    corr_model = CorrectionNet(s_shape=(8, 301), m_dim=len(codec.x_keys))
    optim = torch.optim.AdamW(params=corr_model.parameters(), lr=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=40, gamma=0.1)

    # lit_model = work_model.train(
    #     # optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
    #     # scheduler_cfg={"name": "StepLR", "step_size": 28, "gamma": 0.09},
    #     optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
    #     scheduler_cfg={"name": "StepLR", "step_size": 50, "gamma": 0.01},
    #     loss_fn=CustomLosses("mse_with_l1", weight_decay=1, weights=None)
    #     )

    # Загружаем лучшую модель
    # checkpoint_path="saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=12-val_loss=0.01266-train_loss=0.01224.ckpt",
    # checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=49-train_loss=0.03379-val_loss=0.03352-val_r2=0.84208-val_acc=0.25946-val_mae=0.05526-batch_size=32-dataset_size=100000.ckpt",
    # "saved_models/ERV-KuIMUXT1-BPFC1/best-epoch=22-train_loss=0.04863-val_loss=0.05762-val_r2=0.82785-val_mse=0.01366-val_mae=0.04395-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt"
    # inference_model = work_model.inference("saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=29-train_loss=0.04166-val_loss=0.04450-val_r2=0.92560-val_mse=0.00588-val_mae=0.03862-batch_size=32-base_dataset_size=1500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=29-train_loss=0.03546-val_loss=0.03841-val_r2=0.94190-val_mse=0.00459-val_mae=0.03381-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=34-train_loss=0.02388-val_loss=0.02641-val_r2=0.96637-val_mse=0.00265-val_mae=0.02376-batch_size=32-base_dataset_size=600000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=25-train_loss=0.02530-val_loss=0.02793-val_r2=0.96251-val_mse=0.00296-val_mae=0.02496-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    inference_model = work_model.inference(
        "saved_models\\ERV-KuIMUXT1-BPFC1\\best-epoch=78-train_loss=0.01672-val_loss=0.02663-val_r2=0.96754-val_mse=0.00258-val_mae=0.02406-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference(lit_model.trainer.checkpoint_callback.best_model_path)

    tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/modeling",
                            s_tf=TComposite(
                                [S_Crop(f_start=work_model.orig_filter.f[0], f_stop=work_model.orig_filter.f[-1]),
                                 S_Resample(len(work_model.orig_filter.f))]))

    fast_calc = FastMN2toSParamCalculation(matrix_order=work_model.orig_filter.coupling_matrix.matrix_order,
                                           wlist=work_model.orig_filter.f_norm,
                                           Q=work_model.orig_filter.Q,
                                           fbw=work_model.orig_filter.fbw)
    loss = nn.L1Loss()
    codec_db = copy.deepcopy(codec)
    codec_db.y_channels = ['S1_1.db', 'S1_2.db', 'S2_1.db', 'S2_2.db']
    for i in range(20):
        # i = random.randint(0, len(tds))
        orig_fil = tds[i][1]
        # start_time = time.time()
        pred_prms = inference_model.predict_x(orig_fil)
        # stop_time = time.time()
        # print(f"Predict time: {stop_time - start_time:.3f} sec")
        pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, work_model.meta)
        inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
        keys = pred_prms.keys()
        ts = TouchstoneData(orig_fil)
        preds = pred_fil.coupling_matrix.factors.unsqueeze(0)
        s = codec.encode(ts)[1].unsqueeze(0)
        s_db = codec_db.encode(ts)[1].unsqueeze(0)
        print(f"Initial loss: {loss(s_db, codec_db.encode(TouchstoneData(pred_fil))[1].unsqueeze(0))}")
        corr_model.train()
        start_time = time.time()
        for i in range(50):
            correction = corr_model(s, preds)
            total_pred = correction  # скорректированная матрица
            M = CouplingMatrix.from_factors(total_pred, pred_fil.coupling_matrix.links, pred_fil.coupling_matrix.matrix_order)
            _, s11_pred, s21_pred, s22_pred = fast_calc.RespM2(M, with_s22=True)
            s_db_corr = torch.stack([
                MWFilter.to_db(s11_pred),
                MWFilter.to_db(s21_pred),  # замените на S12, если у вас именно S1_2.db
                MWFilter.to_db(s21_pred),  # либо правильно распакуйте, если RespM2 возвращает все 4
                MWFilter.to_db(s22_pred),
            ]).unsqueeze(0)  # [B, 4, L] — подгоните под ваш codec_db
            # with torch.no_grad():
            #     total_pred = total_pred.squeeze(0)
            #     total_pred_prms = dict(zip(keys, total_pred))
            #     correct_pred_fil = MWFilterBaseLMWithMetrics.create_filter_from_prediction(orig_fil, total_pred_prms, work_model.meta)
            #     s_db_corr = codec_db.encode(TouchstoneData(correct_pred_fil))[1].unsqueeze(0)
            err = loss(s_db_corr, s_db)
            print(f"[{i}] Error: {err.item()}")
            optim.zero_grad()
            err.backward()
            optim.step()
            sch.step(i)
        stop_time = time.time()
        print(f"Tuning time: {stop_time-start_time}")

        with torch.no_grad():
            correction = corr_model(s, preds)
            total_pred = correction  # скорректированная матрица
            total_pred = total_pred.squeeze(0)
            total_pred_prms = dict(zip(keys, total_pred))
            print(f"Origin parameters: {pred_prms}")
            print(f"Tuned parameters: {total_pred_prms}")
            correct_pred_fil = MWFilterBaseLMWithMetrics.create_filter_from_prediction(orig_fil, total_pred_prms, work_model.meta)
            inference_model.plot_origin_vs_prediction(orig_fil, correct_pred_fil)




def main():
    work_model = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE, SamplerTypes.SAMPLER_SOBOL)
    # sensivity.run(work_model.orig_filter)
    # common.plot_distribution(work_model.ds, num_params=len(work_model.ds_gen.origin_filter.coupling_matrix.links))
    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 # keys_for_analysis=[f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links]+["Q"])
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links])
    codec = codec
    work_model_inference_extractor = common.WorkModel(configs.ENV_DATASET_PATH, 50000, SamplerTypes.SAMPLER_SOBOL)
    work_model_inference_extractor.setup(
        model_name="resnet",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )
    work_model_inference_extractor.inference("saved_models/EAMU4-KuIMUXT2-BPFC2/best-epoch=49-train_loss=0.28419-val_loss=0.30947-val_r2=0.75481-val_mse=0.01966-val_mae=0.07616-batch_size=32-base_dataset_size=50000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")

    work_model_inference_main = common.WorkModel(configs.ENV_DATASET_PATH, 100000, SamplerTypes.SAMPLER_SOBOL)
    work_model_inference_main.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )
    work_model_inference_main.model.main_model = work_model_inference_extractor.model
    work_model_inference_main.inference("saved_models/EAMU4-KuIMUXT2-BPFC2/best-epoch=98-train_loss=0.14037-val_loss=0.23625-val_r2=0.89381-val_mse=0.00844-val_mae=0.04743-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")

    work_model.setup(
        model_name="resnet_with_wide_correction",
        model_cfg={"main_model": work_model_inference_main.model, "in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )
    # work_model.setup(
    #     model_name="simple_opt",
    #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )


    # lit_model = work_model.train(
    #     # optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
    #     # scheduler_cfg={"name": "StepLR", "step_size": 28, "gamma": 0.09},
    #     optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
    #     scheduler_cfg={"name": "StepLR", "step_size": 50, "gamma": 0.01},
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
    inference_model = work_model.inference("saved_models/EAMU4-KuIMUXT2-BPFC2/best-epoch=71-train_loss=0.11403-val_loss=0.13005-val_r2=0.98534-val_mse=0.00116-val_mae=0.01580-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
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
    orig_fil, pred_fil = inference_model.predict(work_model.dm, idx=0)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    optim_matrix = optimize_cm(pred_fil, orig_fil)
    error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)

    error_matrix_optim.plot_matrix(title="Optimized de-tuned matrix errors", cmap="YlOrBr")
    optim_matrix.plot_matrix(title="Optimized de-tuned matrix")
    error_matrix_pred.plot_matrix(title="Predict de-tuned matrix errors", cmap="YlOrBr")
    pred_fil.coupling_matrix.plot_matrix(title="Predict de-tuned matrix")
    orig_fil.coupling_matrix.plot_matrix(title="Origin de-tuned matrix")


    # optim_matrix = optimize_cm(pred_fil, orig_fil)
    # error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    # error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    # error_matrix_optim.plot_matrix(title="Optimized tuned matrix errors for inverted model", cmap="YlOrBr")
    # error_matrix_pred.plot_matrix(title="Predict tuned matrix errors for inverted model", cmap="YlOrBr")
    # orig_fil.coupling_matrix.plot_matrix(title="Origin tuned matrix for inverted model")
    # pred_fil.coupling_matrix.plot_matrix(title="Predict tuned matrix for inverted model")
    # optim_matrix.plot_matrix(title="Optimized tuned matrix for inverted model")


    # tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/modeling/detuned",
    #                         s_tf=TComposite(
    #                             [S_Crop(f_start=work_model.orig_filter.f[0], f_stop=work_model.orig_filter.f[-1]),
    #                              S_Resample(len(work_model.orig_filter.f))]))
    # for i in range(len(tds)):
    #     # i = random.randint(0, len(tds))
    #     orig_fil = tds[i][1]
    #     # start_time = time.time()
    #     pred_prms = inference_model.predict_x(orig_fil)
    #     # stop_time = time.time()
    #     # print(f"Predict time: {stop_time - start_time:.3f} sec")
    #     print(f"Предсказанные параметры: {pred_prms}")
    #     pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, work_model.meta)
    #     inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    #     optim_matrix = optimize_cm(pred_fil, orig_fil)
    #     # optim_matrix.plot_matrix(title="AI-extracted matrix (Res3)")
    #     # cst_matrix = CouplingMatrix.from_file("filters/FilterData/EAMU4T1-BPFC2/measure/AMU4_T1_Ch2_Measur_Res3_var_extr_matrix.txt")
    #     # cst_matrix.plot_matrix(title="CST-extracted matrix (Res3)")
    #     print(f"Оптимизированные параметры: {optim_matrix.factors}. Добротность: {pred_fil.Q}")
    #
    #     # plt.figure()
    #     # plt.plot(pred_fil.f_norm, orig_fil.s_db[:, 0, 0], label='S11 origin')
    #     # plt.plot(pred_fil.f_norm, orig_fil.s_db[:, 1, 0], label='S21 origin')
    #     # plt.plot(pred_fil.f_norm, orig_fil.s_db[:, 1, 1], label='S22 origin')
    #     #
    #     # fast_calc = FastMN2toSParamCalculation(matrix_order=pred_fil.coupling_matrix.matrix_order,
    #     #                                        wlist=pred_fil.f_norm,
    #     #                                        Q=pred_fil.Q,
    #     #                                        fbw=pred_fil.fbw)

    # Предсказываем эталонный фильтр
    # orig_fil = work_model.ds_gen.origin_filter
    # pred_prms = inference_model.predict_x(orig_fil)
    # corrected = predict_with_corrector(np.array(list(pred_prms.values())).reshape(1, -1), "saved_models\\EAMU4-KuIMUXT3-BPFC1\\ml-correctors")
    # corr_prms = dict(zip(pred_prms.keys(), list(corrected.reshape(-1))))
    # pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, work_model.meta)
    # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)


    # Предсказываем эталонный фильтр
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
    # online_correct()
    main()
    plt.show()
