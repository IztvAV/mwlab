import os
import time

from mwlab.nn.scalers import MinMaxScaler
from mwlab import TouchstoneDataset, TouchstoneLDataModule, TouchstoneDatasetAnalyzer
from mwlab.io.backends import RAMBackend
from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample

from filters import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers, SamplerTypes, MWFilter
from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLModule, MWFilterBaseLMWithMetrics

from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift

import matplotlib.pyplot as plt
import lightning as L

from torch import nn
import models
import torch
from filters.mwfilter_optim.bfgs import optimize_cm

torch.set_float32_matmul_precision("medium")

BATCH_SIZE = 64
BASE_DATASET_SIZE = 250_000
FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "datasets_data")


def plot_distribution(ds: TouchstoneDataset, num_params: int, batch: int = 6):
    analyzer = TouchstoneDatasetAnalyzer(ds)
    varying = analyzer.get_varying_keys()
    r = int((num_params / batch) + num_params % batch)
    for i in range(r):
        analyzer.plot_param_distributions(varying[batch * i:batch * (i + 1)])


def create_origin_filter(path_orig_filter: str, f_start=None, f_stop=None, f_unit=None, resample_scale=301):
    tds = TouchstoneDataset(source=path_orig_filter)
    origin_filter = MWFilter.from_touchstone_dataset_item(tds[0])
    f0 = origin_filter.f0
    bw = origin_filter.bw
    if f_start is None:
        f_start = f0 - 1.2 * bw
    if f_stop is None:
        f_stop = f0 + 1.2 * bw
    if f_unit is None:
        f_unit = "MHz"
    y_transform = TComposite([
        S_Crop(f_start=f_start, f_stop=f_stop, unit=f_unit),
        S_Resample(resample_scale)
    ])
    tds_transformed = TouchstoneDataset(source=path_orig_filter, s_tf=y_transform)
    origin_filter = MWFilter.from_touchstone_dataset_item(tds_transformed[0])
    return origin_filter


def create_samplers(orig_filter: MWFilter):
    sampler_configs = {
        "pss_origin": PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        "pss_shifts_delta": PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        "cm_shifts_delta": CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=0.005),
        "samplers_size": BASE_DATASET_SIZE,
    }
    samplers_lhs_all_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                    samplers_type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE(one_param=False),
                                                                                    **sampler_configs)
    samplers_lhs_all_params_shuffle_cms = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_lhs_all_params.cms.shuffle(ratio=1),
        pss=samplers_lhs_all_params.pss)
    samplers_lhs_all_params_shuffle_pss = CMTheoreticalDatasetGeneratorSamplers(cms=samplers_lhs_all_params.cms,
                                                                                pss=samplers_lhs_all_params.pss.shuffle(
                                                                                    ratio=1))
    samplers_lhs_all_params_shuffle_all = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_lhs_all_params.cms.shuffle(ratio=1),
        pss=samplers_lhs_all_params.pss.shuffle(ratio=1))

    sampler_configs["samplers_size"] = int(BASE_DATASET_SIZE/100)
    samplers_lhs_with_one_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                         samplers_type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE(one_param=True),
                                                                                         **sampler_configs)
    total_samplers = CMTheoreticalDatasetGeneratorSamplers.concat(
            (samplers_lhs_all_params_shuffle_cms, samplers_lhs_all_params_shuffle_pss,
             samplers_lhs_all_params_shuffle_all, samplers_lhs_all_params)
    )
    return total_samplers


def main():
    print("Создаем фильтр")
    orig_filter = create_origin_filter(ENV_ORIGIN_DATA_PATH)
    print("Создаем сэмплеры")
    samplers = create_samplers(orig_filter)
    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_save_dataset=os.path.join(ENV_DATASET_PATH, samplers.cms.type.name, f"{len(samplers.cms)}"),
        backend_type='ram',
        orig_filter=orig_filter,
        filename="Dataset",
    )
    ds_gen.generate(samplers)

    ds = TouchstoneDataset(source=ds_gen.backend, in_memory=True)
    plot_distribution(ds, num_params=len(ds_gen.origin_filter.coupling_matrix.links))

    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds)
    codec.exclude_keys(["f0", "bw", "N", "Q"])
    print(codec)
    # Исключаем из анализа ненужные x-параметры
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    dm = TouchstoneLDataModule(
        source=ds_gen.backend,  # Путь к датасету
        codec=codec,  # Кодек для преобразования TouchstoneData → (x, y)
        batch_size=BATCH_SIZE,  # Размер батча
        val_ratio=0.2,  # Доля валидационного набора
        test_ratio=0.05,  # Доля тестового набора
        cache_size=0,
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

    # Декодирование
    _, __, meta = dm.get_dataset(split="train", meta=True)[0]
    ts_rec = codec.decode(dm.train_ds.dataset[0][0], meta)
    plt.figure()
    ts_rec.network.plot_s_db(m=0, n=0, label='S11 from dataset')
    ts_rec.network.plot_s_db(m=1, n=0, label='S21 from dataset')

    # Печатаем размеры полученных наборов
    print(f"Размер тренировочного набора: {len(dm.train_ds)}")
    print(f"Размер валидационного набора: {len(dm.val_ds)}")
    print(f"Размер тестового набора: {len(dm.test_ds)}")

    model = models.ResNet1DFlexible(
        in_channels=len(codec.y_channels),
        out_channels=len(ds_gen.origin_filter.coupling_matrix.links),
        num_blocks=[1, 1, 6, 3],
        layer_channels=[128, 256, 512, 512],
        first_conv_kernel=11,
        first_conv_channels=128,
    )

    lit_model = MWFilterBaseLMWithMetrics(
        model=model,  # Наша нейросетевая модель
        swap_xy=True,
        scaler_in=dm.scaler_in,  # Скейлер для входных данных
        scaler_out=dm.scaler_out,  # Скейлер для выходных данных
        codec=codec,  # Кодек для преобразования данных
        optimizer_cfg={"name": "Adam", "lr": 0.0017552306729777972},
        scheduler_cfg={"name": "StepLR", "step_size": 10, "gamma": 0.1},
        loss_fn=nn.MSELoss()
    )

    stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=15, mode="min", min_delta=0.00001)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", dirpath="saved_models/" + FILTER_NAME,
                                                     filename="best-{epoch}-{val_loss:.5f}-{train_loss:.5f}",
                                                     mode="min",
                                                     save_top_k=1,  # Сохраняем только одну лучшую
                                                     save_weights_only=False,
                                                     # Сохранять всю модель (включая структуру)
                                                     verbose=False  # Отключаем логирование сохранения)
                                                     )

    # Обучение модели с помощью PyTorch Lightning
    trainer = L.Trainer(
        max_epochs=150,  # Максимальное количество эпох обучения
        accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
        log_every_n_steps=100,  # Частота логирования в процессе обучения
        callbacks=[
            stoping,
            checkpoint
        ]
    )
    # # Загружаем лучшую модель
    # lit_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
    #     checkpoint_path="saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=10-val_loss=0.01374-train_loss=0.01136.ckpt",
    #     model=model
    # ).to(lit_model.device)


    # Запуск процесса обучения
    trainer.fit(lit_model, dm)
    print(f"Best model saved into: {checkpoint.best_model_path}")

    # Загружаем лучшую модель
    inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
        checkpoint_path=checkpoint.best_model_path,
        model=model
    ).to(lit_model.device)
    orig_fil, pred_fil = inference_model.predict(dm, idx=0)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    optimize_cm(pred_fil, orig_fil)

    # Предсказываем эталонный фильтр
    orig_fil = ds_gen.origin_filter
    pred_prms = inference_model.predict_x(orig_fil)
    pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, meta)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    optimize_cm(pred_fil, orig_fil)
    plt.show()


if __name__ == "__main__":
    main()
    plt.show()
