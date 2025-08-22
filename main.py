import os
import random
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from torch.nn import MSELoss, L1Loss

from common import check_metrics
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
from mwlab.nn.scalers import MinMaxScaler, StdScaler
from mwlab import TouchstoneDataset, TouchstoneLDataModule, TouchstoneDatasetAnalyzer

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

import sklearn
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import joblib
from tqdm import tqdm
import sensivity
from torchvision import models as t_models

from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample

torch.set_float32_matmul_precision("medium")

# ---------- 2. Сбор предсказаний ----------
def collect_nn_predictions(model, dataloader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="NN prediction"):
            x, y = batch
            if hasattr(model, 'to'):  # на случай, если это LightningModule
                x = x.to(model.device)
            y_pred = model(x).cpu()
            preds.append(y_pred.numpy())
            targets.append(y.numpy())

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def fit_minmax_scalers(preds, targets):
    pred_scaler = sklearn.preprocessing.MinMaxScaler()
    target_scaler = sklearn.preprocessing.MinMaxScaler()
    pred_scaled = pred_scaler.fit_transform(preds)
    target_scaled = target_scaler.fit_transform(targets)
    return pred_scaler, target_scaler, pred_scaled, target_scaled


def train_boosting_model(X_train_scaled, y_train_scaled, params=None):
    if params is None:
        params = dict(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            verbose=1
        )

    base_model = GradientBoostingRegressor(**params)
    model = MultiOutputRegressor(base_model)

    print("➡️ Обучение модели...")
    model.fit(X_train_scaled, y_train_scaled)

    print("✅ Обучение завершено. Вычисляю метрики на обучающей выборке...\n")

    y_pred = model.predict(X_train_scaled)

    mse = sklearn.metrics.mean_squared_error(y_train_scaled, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_train_scaled, y_pred)
    r2 = sklearn.metrics.r2_score(y_train_scaled, y_pred)


    print("\n📈 Mетрики:")
    print(f"  MSE = {mse:.4f}")
    print(f"  R²  = {r2:.4f}")
    print(f"  MAE = {mae:.4f}")

    return model


def save_boosting_pipeline(path: str | Path, model, pred_scaler, target_scaler):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, path / "gb_model.joblib")
    joblib.dump(pred_scaler, path / "scaler_pred.joblib")
    joblib.dump(target_scaler, path / "scaler_target.joblib")


def train_corrector(lightning_model, datamodule, output_dir="./gb_corrector"):
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()

    # Получаем предсказания нейросети
    print("Get AI-prediction")
    preds, targets = collect_nn_predictions(lightning_model, train_loader)

    # Нормируем
    print("Normalize data")
    pred_scaler, target_scaler, X_scaled, y_scaled = fit_minmax_scalers(preds, targets)

    # Обучаем градиентный бустинг
    print("Start train ml-model")
    model = train_boosting_model(X_scaled, y_scaled)

    # Сохраняем модель и скейлеры
    save_boosting_pipeline(output_dir, model, pred_scaler, target_scaler)

    print(f"✅ Обучение завершено. Модель и скейлеры сохранены в: {output_dir}")


def predict_with_corrector(nn_preds: np.ndarray, corrector_dir: str | Path) -> np.ndarray:
    """
    Корректирует предсказания нейросети с помощью обученной ML-модели и скейлеров.

    :param nn_preds: numpy-массив предсказаний нейросети (размер [N, D])
    :param corrector_dir: путь к директории с моделью и скейлерами
    :return: откорректированные предсказания (в исходном масштабе)
    """
    corrector_dir = Path(corrector_dir)

    # Загрузка модели и скейлеров
    model = joblib.load(corrector_dir / "gb_model.joblib")
    pred_scaler = joblib.load(corrector_dir / "scaler_pred.joblib")
    target_scaler = joblib.load(corrector_dir / "scaler_target.joblib")

    # Нормализация входных предсказаний
    nn_preds_scaled = pred_scaler.transform(nn_preds)

    # Получение откорректированных предсказаний
    corrected_scaled = model.predict(nn_preds_scaled)

    # Обратное преобразование в исходный масштаб
    corrected = target_scaler.inverse_transform(corrected_scaled)
    return corrected

def fine_tune_model(inference_model: nn.Module, target_input: MWFilter, meta: dict, device='cuda', epochs=10, lr=1e-4):
    """
    model: обученная модель
    generate_nearby_dataset: функция, создающая датасет около предсказания
    target_input: вход, вблизи которого хотим дообучить модель
    """
    # 🔁 Сгенерировать небольшой датасет около предсказания
    orig_fil = common.create_origin_filter(configs.ENV_ORIGIN_DATA_PATH)
    for i in range(5):
        print(f"Iteration: {i}")
        pred_prms = inference_model.predict_x(target_input)
        pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, meta)
        sampler_configs = {
            "pss_origin": PSShift(phi11=1e-12, phi21=1e-12, theta11=1e-12, theta21=1e-12),
            "pss_shifts_delta": PSShift(phi11=1e-12, phi21=1e-12, theta11=1e-12, theta21=1e-12),
            "cm_shifts_delta": CMShifts(self_coupling=0.1, mainline_coupling=0.1, cross_coupling=0.0001),
            "samplers_size": 1000,
        }
        samplers = CMTheoreticalDatasetGeneratorSamplers.create_samplers(pred_fil,
                                                                                    samplers_type=SamplerTypes.SAMPLER_SOBOL(
                                                                                        one_param=False),
                                                                                        **sampler_configs
                                                                                    )
        ds_gen = CMTheoreticalDatasetGenerator(
            path_to_save_dataset=os.path.join(configs.ENV_TUNE_DATASET_PATH, samplers.cms.type.name, f"{len(samplers.cms)}"),
            backend_type='ram',
            orig_filter=pred_fil,
            filename="Dataset",
            rewrite=True
        )
        ds_gen.generate(samplers)

        ds = TouchstoneDataset(source=ds_gen.backend, in_memory=True)
        # common.plot_distribution(ds, num_params=len(ds_gen.origin_filter.coupling_matrix.links))
        # plt.show()
        codec = MWFilterTouchstoneCodec.from_dataset(ds=ds,
                                                     keys_for_analysis=[f"m_{r}_{c}" for r, c in orig_fil.coupling_matrix.links])
        dm = TouchstoneLDataModule(
            source=ds_gen.backend,  # Путь к датасету
            codec=codec,  # Кодек для преобразования TouchstoneData → (x, y)
            batch_size=configs.BATCH_SIZE,  # Размер батча
            val_ratio=0,  # Доля валидационного набора
            test_ratio=0,  # Доля тестового набора
            cache_size=None,
            scaler_in=MinMaxScaler(dim=(0, 2), feature_range=(0, 1)),  # Скейлер для входных данных
            scaler_out=MinMaxScaler(dim=0, feature_range=(-0.5, 0.5)),  # Скейлер для выходных данных
            swap_xy=True,
            num_workers=0,
            # Параметры базового датасета:
            base_ds_kwargs={
                "in_memory": True
            }
        )
        dm.setup("fit")
        dataloader = dm.train_dataloader()
        inference_model = inference_model.to(device)
        inference_model.train()

        # 🔧 Оптимизатор и функция потерь
        trainer = L.Trainer(
            deterministic=True,
            max_epochs=epochs,  # Максимальное количество эпох обучения
            accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
            log_every_n_steps=100,  # Частота логирования в процессе обучения
        )
        trainer.fit(inference_model, dm)

    return inference_model


def main_ae():
    sampler_configs = {
        "pss_origin": PSShift(phi11=0, phi21=0, theta11=0, theta21=0),
        "pss_shifts_delta": PSShift(phi11=0.00000001, phi21=0.00000001, theta11=0.00000001, theta21=0.00000001),
        "cm_shifts_delta": CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=5e-3,
                                    parasitic_coupling=5e-3),
        "samplers_size": configs.BASE_DATASET_SIZE,
    }
    work_model = common.AEWorkModel(configs.ENV_DATASET_PATH, sampler_configs, SamplerTypes.SAMPLER_SOBOL)
    # common.plot_distribution(work_model.ds, num_params=len(work_model.ds_gen.origin_filter.coupling_matrix.links))
    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                    work_model.orig_filter.coupling_matrix.links])
    codec.y_channels = ['S1_1.db', 'S1_2.db', 'S2_1.db', 'S2_2.db']
    codec = codec
    work_model.setup(
        # model_name="cae",
        # model_cfg={"in_ch":len(codec.y_channels), "z_dim":work_model.orig_filter.order*6},
        model_name="imp_cae",
        model_cfg={"in_ch":len(codec.y_channels), "z_dim":work_model.orig_filter.order*6},
        # model_name="mlp",
        # model_cfg={},
        dm_codec=codec
    )

    lit_model = work_model.train(
        optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-2},
        scheduler_cfg={"name": "StepLR", "step_size": 24, "gamma": 0.01},
        loss_fn=L1Loss()
    )

    # Загружаем лучшую модель
    inference_model = work_model.inference(lit_model.trainer.checkpoint_callback.best_model_path)
    # inference_model = work_model.inference("saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=24-train_loss=0.00163-val_loss=0.00146-val_r2=0.99352-val_mse=0.00002-val_mae=0.00146-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")

    check_metrics(trainer=work_model.trainer, model=inference_model, dm=work_model.dm)

    # Предсказываем фильтр из тестового датасета
    orig_fil, pred_fil = inference_model.predict(work_model.dm, idx=0)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    plt.figure()
    plt.plot(orig_fil.f, np.real(orig_fil.s[:, 0, 0]), orig_fil.f, np.real(pred_fil.s[:, 0, 0]),
             orig_fil.f, np.imag(orig_fil.s[:, 0, 0]), orig_fil.f, np.imag(pred_fil.s[:, 0, 0]))
    plt.legend(["real S11 orig", "real S11 pred", "imag S11 orig", "imag S11 pred"])
    plt.title("S11")

    plt.figure()
    plt.plot(orig_fil.f, np.real(orig_fil.s[:, 1, 0]), orig_fil.f, np.real(pred_fil.s[:, 1, 0]),
             orig_fil.f, np.imag(orig_fil.s[:, 1, 0]), orig_fil.f, np.imag(pred_fil.s[:, 1, 0]))
    plt.legend(["real S21 orig", "real S21 pred", "imag S21 orig", "imag S21 pred"])
    plt.title("S21")


def main_extract_matrix():
    sampler_configs = {
        "pss_origin": PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        "pss_shifts_delta": PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        # "cm_shifts_delta": CMShifts(self_coupling=1.8, mainline_coupling=0.3, cross_coupling=9e-2, parasitic_coupling=5e-3), # Параметры для реального фильтра
        "cm_shifts_delta": CMShifts(self_coupling=0.5, mainline_coupling=0.1, cross_coupling=5e-3,
                                    parasitic_coupling=5e-3),
        "samplers_size": configs.BASE_DATASET_SIZE,
    }
    work_model = common.WorkModel(configs.ENV_DATASET_PATH, sampler_configs, SamplerTypes.SAMPLER_STD)
    # sensivity.run(work_model.orig_filter)
    # common.plot_distribution(work_model.ds, num_params=len(work_model.ds_gen.origin_filter.coupling_matrix.links))
    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links])
    codec = codec
    # work_model_inference = common.WorkModel(configs.ENV_DATASET_PATH, 1000, SamplerTypes.SAMPLER_SOBOL)
    # work_model_inference.setup(
    #     model_name="resnet_with_correction",
    #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # work_model_inference.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=25-train_loss=0.03897-val_loss=0.04207-val_r2=0.92781-val_mse=0.00573-val_mae=0.03634-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # work_model.setup(
    #     model_name="resnet_with_wide_correction",
    #     model_cfg={"main_model": work_model_inference.model,
    #                "in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    # work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=25-train_loss=0.02530-val_loss=0.02793-val_r2=0.96251-val_mse=0.00296-val_mae=0.02496-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    #
    # work_model.setup(
    #     model_name="resnet_with_correction",
    #     model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
    #     dm_codec=codec
    # )
    work_model.setup(
        model_name="cae",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )

    lit_model = work_model.train(
        # optimizer_cfg={"name": "AdamW", "lr": 0.0009400000000000001, "weight_decay": 1e-5},
        # scheduler_cfg={"name": "StepLR", "step_size": 28, "gamma": 0.09},
        optimizer_cfg={"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5},
        scheduler_cfg={"name": "StepLR", "step_size": 24, "gamma": 0.01},
        loss_fn=CustomLosses("mse_with_l1", weight_decay=1, weights=None)
        )

    # Загружаем лучшую модель
    # checkpoint_path="saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=12-val_loss=0.01266-train_loss=0.01224.ckpt",
    # checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=49-train_loss=0.03379-val_loss=0.03352-val_r2=0.84208-val_acc=0.25946-val_mae=0.05526-batch_size=32-dataset_size=100000.ckpt",
    # "saved_models/ERV-KuIMUXT1-BPFC1/best-epoch=22-train_loss=0.04863-val_loss=0.05762-val_r2=0.82785-val_mse=0.01366-val_mae=0.04395-batch_size=32-base_dataset_size=100000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt"
    # inference_model = work_model.inference("saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=29-train_loss=0.04166-val_loss=0.04450-val_r2=0.92560-val_mse=0.00588-val_mae=0.03862-batch_size=32-base_dataset_size=1500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=29-train_loss=0.03546-val_loss=0.03841-val_r2=0.94190-val_mse=0.00459-val_mae=0.03381-batch_size=32-base_dataset_size=500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=34-train_loss=0.02388-val_loss=0.02641-val_r2=0.96637-val_mse=0.00265-val_mae=0.02376-batch_size=32-base_dataset_size=600000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # inference_model = work_model.inference("saved_models\\EAMU4T1-BPFC2\\best-epoch=25-train_loss=0.02530-val_loss=0.02793-val_r2=0.96251-val_mse=0.00296-val_mae=0.02496-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    inference_model = work_model.inference(lit_model.trainer.checkpoint_callback.best_model_path)

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
    # orig_fil, pred_fil = inference_model.predict(work_model.dm, idx=0)
    # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    # optim_matrix = optimize_cm(pred_fil, orig_fil)
    # error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    # error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    #
    # error_matrix_optim.plot_matrix(title="Optimized de-tuned matrix errors", cmap="YlOrBr")
    # optim_matrix.plot_matrix(title="Optimized de-tuned matrix")
    # error_matrix_pred.plot_matrix(title="Predict de-tuned matrix errors", cmap="YlOrBr")
    # pred_fil.coupling_matrix.plot_matrix(title="Predict de-tuned matrix")
    # orig_fil.coupling_matrix.plot_matrix(title="Origin de-tuned matrix")


    #
    # optim_matrix = optimize_cm(pred_fil, orig_fil)
    # error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    # error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    # error_matrix_optim.plot_matrix(title="Optimized tuned matrix errors for inverted model", cmap="YlOrBr")
    # error_matrix_pred.plot_matrix(title="Predict tuned matrix errors for inverted model", cmap="YlOrBr")
    # orig_fil.coupling_matrix.plot_matrix(title="Origin tuned matrix for inverted model")
    # pred_fil.coupling_matrix.plot_matrix(title="Predict tuned matrix for inverted model")
    # optim_matrix.plot_matrix(title="Optimized tuned matrix for inverted model")


    tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/modeling",
                            s_tf=TComposite(
                                [S_Crop(f_start=work_model.orig_filter.f[0], f_stop=work_model.orig_filter.f[-1]),
                                 S_Resample(len(work_model.orig_filter.f))]))
    for i in range(1):
        # i = random.randint(0, len(tds))
        orig_fil = tds[2][1]
        # start_time = time.time()
        pred_prms = inference_model.predict_x(orig_fil)
        # stop_time = time.time()
        # print(f"Predict time: {stop_time - start_time:.3f} sec")
        print(f"Предсказанные параметры: {pred_prms}")
        pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, work_model.meta)
        inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
        optim_matrix = optimize_cm(pred_fil, orig_fil)
        # optim_matrix.plot_matrix(title="AI-extracted matrix (Res3)")
        # cst_matrix = CouplingMatrix.from_file("filters/FilterData/EAMU4T1-BPFC2/measure/AMU4_T1_Ch2_Measur_Res3_var_extr_matrix.txt")
        # cst_matrix.plot_matrix(title="CST-extracted matrix (Res3)")
        print(f"Оптимизированные параметры: {optim_matrix.factors}")

        # plt.figure()
        # plt.plot(pred_fil.f_norm, orig_fil.s_db[:, 0, 0], label='S11 origin')
        # plt.plot(pred_fil.f_norm, orig_fil.s_db[:, 1, 0], label='S21 origin')
        # plt.plot(pred_fil.f_norm, orig_fil.s_db[:, 1, 1], label='S22 origin')


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
    # csv_path = "lightning_logs/cae_csv/version_71/metrics.csv"
    # plot_pl_csv_wide(csv_path)
    main_ae()
    plt.show()
