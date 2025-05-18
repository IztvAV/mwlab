import os
from mwlab.nn.scalers import MinMaxScaler
from mwlab import TouchstoneDataset, TouchstoneLDataModule
from mwlab.io.backends import RAMBackend

from filters import CMTheoreticalDatasetGenerator
from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLModule, MWFilterBaseLMWithMetrics

from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift

import matplotlib.pyplot as plt
import lightning as L

from torch import nn
import models
import torch

torch.set_float32_matmul_precision("medium")


BATCH_SIZE = 64
DATASET_SIZE = 500_000
FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "FilterData", FILTER_NAME, "datasets_data")


def main():
    backend = RAMBackend([])
    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_origin_filter=ENV_ORIGIN_DATA_PATH,
        path_to_save_dataset=ENV_DATASET_PATH,
        pss_origin=PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        pss_shifts_delta=PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        cm_shifts_delta=CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=0.005),
        samplers_size=DATASET_SIZE,
        backend=backend
    )
    ds_gen.generate()

    codec = MWFilterTouchstoneCodec.from_dataset(TouchstoneDataset(source=backend, in_memory=True))
    codec.exclude_keys(["f0", "bw", "N", "Q"])
    print(codec)
    # codec.y_channels = ['S1_1.real', 'S2_1.real', 'S2_2.real', 'S1_1.imag', 'S2_1.imag', 'S2_2.imag']
    # codec.y_channels = ['S1_1.db', 'S1_2.db', 'S2_1.db', 'S2_2.db']

    # Исключаем из анализа ненужные x-параметры
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    dm = TouchstoneLDataModule(
        source=backend,         # Путь к датасету
        codec=codec,                   # Кодек для преобразования TouchstoneData → (x, y)
        batch_size=BATCH_SIZE,                 # Размер батча
        val_ratio=0.2,                 # Доля валидационного набора
        test_ratio=0.05,                # Доля тестового набора
        cache_size=0,
        scaler_in=MinMaxScaler(dim=(0, 2), feature_range=(0, 1)),                          # Скейлер для входных данных
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

    # model = models.Simple_Opt_3(in_channels=len(codec.y_channels),
    #                      out_channels=len(ds_gen.origin_filter.coupling_matrix.links))
    model = models.ResNet1D(in_channels=len(codec.y_channels),
                         out_channels=len(ds_gen.origin_filter.coupling_matrix.links))
    # model = models.ImprovedResNet1D(in_channels=len(codec.y_channels),
    #                                 out_channels=len(ds_gen.origin_filter.coupling_matrix.links),
    #                                 layer_blocks=[1, 2, 4, 1],
    #                                 dilation_factors=[1, 1, 1],
    #                                 base_channels=64)
    # model = models.ResNet1DBiRNN(in_channels=len(codec.y_channels),
    #                              out_channels=len(ds_gen.origin_filter.coupling_matrix.links),
    #                              resnet_out_channels=256,
    #                              hidden_size=256,
    #                              num_layers=2,
    #                              dropout=0.0,
    #                              rnn_type='gru'
    #                              )
    # model = models.DenseNet1D(in_channels=len(codec.y_channels),
    #                           growth_rate=48,
    #                           num_classes=len(ds_gen.origin_filter.coupling_matrix.links))
    # model = models.BiRNN(in_channels=301,
    #                      num_layers=5,
    #                      out_channels=len(ds_gen.origin_filter.coupling_matrix.links),
    #                      hidden_size=512,
    #                      droupout=0.0,
    #                      rnn_type='lstm')

    lit_model = MWFilterBaseLMWithMetrics(
        model=model,  # Наша нейросетевая модель
        swap_xy=True,
        scaler_in=dm.scaler_in,  # Скейлер для входных данных
        scaler_out=dm.scaler_out,  # Скейлер для выходных данных
        codec=codec,  # Кодек для преобразования данных
        optimizer_cfg={"name": "Adam", "lr": 1e-2},  # Конфигурация оптимизатора
        scheduler_cfg={"name": "StepLR", "step_size": 20, "gamma": 0.5},
        # optimizer_cfg={"name": "SGD", "lr": 0.1, "momentum": 0.99, "nesterov": True},
        # scheduler_cfg={"name": "CosineAnnealingWarmRestarts", "T_0": 4, "T_mult": 2, "eta_min": 1e-5},
        # optimizer_cfg={"name": "AdamW", "lr": 0.01},
        # scheduler_cfg={"name": "OneCycleLR", "max_lr": 1e-2, "epochs": 5, "steps_per_epoch": len(dm.train_ds)},
        loss_fn=nn.MSELoss()
    )


    stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min", min_delta=0.00001)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", dirpath="saved_models/"+FILTER_NAME,
                                                     filename="best-{epoch}-{val_loss:.5f}",
                                                     mode="min",
                                                     save_top_k=1,                                  # Сохраняем только одну лучшую
                                                     save_weights_only=False,                       # Сохранять всю модель (включая структуру)
                                                     verbose=False                                  # Отключаем логирование сохранения)
                                                     )

    # Обучение модели с помощью PyTorch Lightning
    trainer = L.Trainer(
        max_epochs=500,  # Максимальное количество эпох обучения
        accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
        log_every_n_steps=100,  # Частота логирования в процессе обучения
        callbacks=[
            stoping,
            checkpoint
        ]
    )

    # Запуск процесса обучения
    trainer.fit(lit_model, dm)
    trainer.save_checkpoint("saved_models/last.ckpt")
    print(f"Best model saved into: {checkpoint.best_model_path}")

    # Загружаем лучшую модель
    inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
        checkpoint_path=checkpoint.best_model_path,
        model=model
    ).to(lit_model.device)
    orig_fil, pred_fil = inference_model.predict(dm, idx=0)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)

    # Загружаем последнюю модель
    last_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
        checkpoint_path="saved_models/last.ckpt",
        model=model
    ).to(lit_model.device)
    orig_fil, pred_fil = last_model.predict(dm, idx=0)
    last_model.plot_origin_vs_prediction(orig_fil, pred_fil)

    orig_fil, pred_fil = lit_model.predict(dm, idx=0)
    lit_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    plt.show()


if __name__ == "__main__":
    main()
    plt.show()
