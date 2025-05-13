import os
import mwlab
from mwlab.nn.scalers import MinMaxScaler

from filters import TouchstoneMWFilterDataset, MWFilter, CouplingMatrix
from filters import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers
from filters import Sampler, SamplerTypes
from filters.codecs import MWFilterTouchstoneCodec

from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift

import matplotlib.pyplot as plt
import numpy as np
import lightning as L
from sympy.diffgeom.rn import theta

from torch import nn
from dataclasses import dataclass

DATASET_SIZE = 1_000
FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
ENV_ORIGIN_DATA_PATH = os.getcwd() + f"\\Data\\{FILTER_NAME}\\origins_data"
ENV_DATASET_PATH = os.getcwd() + f"\\Data\\{FILTER_NAME}\\datasets_data"


class Simple_Opt_3(nn.Module):
    def __init__(self, N):
        super(Simple_Opt_3, self).__init__()
        # Количество выходных аргументов
        self.nargout = 1
        # Количество выходных каналов
        self.in_channels=6

        # --------------------------  1 conv-слой ---------------------------
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=1, padding='same')
        # --------------------------  2 conv-слой ---------------------------
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        # --------------------------  3 conv-слой ---------------------------
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        # --------------------------  4 conv-слой ---------------------------
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        self.seq_conv = nn.Sequential(
            # --------------------------  1 conv-слой ---------------------------
            self.conv1,
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3, padding=1),
            # --------------------------  2 conv-слой ---------------------------
            self.conv2,
            nn.ReLU(),
            self.conv2,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # --------------------------  3 conv-слой ---------------------------
            self.conv3,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # --------------------------  4 conv-слой ---------------------------
            self.conv4,
        )
        self.seq_fc = nn.Sequential(
            # --------------------------  fc-слои ---------------------------
            nn.Linear(64 * 26, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, N), # N - количество ненулевых элементов матрицы связи
            nn.Tanh()
        )

    def encode(self, x):
        conv_x = self.seq_conv(x)
        conv_x_reshaped = conv_x.view(conv_x.size(0), -1)
        fc_x = self.seq_fc(conv_x_reshaped)
        return fc_x

    def forward(self, x):
        encoded = self.encode(x)
        return encoded


def main():
    # tds = mwlab.TouchstoneDataset(source=ENV_ORIGIN_DATA_PATH)
    # print(f"Загружено файлов: {len(tds)}")
    # print("Пример параметров из первого файла:")
    # print(tds[0][0], "\n")
    # origin_filter: MWFilter = MWFilter.from_touchstone_dataset_item(tds[0])
    # y_transform = TComposite([
    #     S_Crop(f_start=origin_filter.f0-origin_filter.bw*1.2, f_stop=origin_filter.f0+origin_filter.bw*1.2, unit='MHz'),
    #     S_Resample(301)
    # ])
    # tds_transformed = mwlab.TouchstoneDataset(source=ENV_ORIGIN_DATA_PATH, s_tf=y_transform)
    #
    # # Пример кодирования и декодирования
    # origin_filter = MWFilter.from_touchstone_dataset_item(tds_transformed[0])
    # prms, _ = tds_transformed[0]  # Используем первый файл набора
    #
    # m_min, m_max = create_min_max_matrices(origin_matrix=origin_filter.coupling_matrix, deltas=np.array([1.5, 0.1, 0.005]))
    # phase_shifts_min, phase_shifts_max = create_min_max_phase_shifts(origin_shifts=np.array([0.547, -1.0, 0.01685, 0.017]),
    #                                                                  deltas=np.array([0.02, 0.02, 0.005, 0.005]))
    # samplers = CMTheoreticalDatasetGeneratorSamplers(
    #     cm_shifts=Sampler.lhs(start=m_min, stop=m_max, num=DATASET_SIZE),
    #     ps_shifts=Sampler.lhs(start=phase_shifts_min, stop=phase_shifts_max, num=DATASET_SIZE)
    # )

    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_origin_filter=ENV_ORIGIN_DATA_PATH,
        path_to_save_dataset=ENV_DATASET_PATH,
        pss_origin=PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        pss_shifts_delta=PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        cm_shifts_delta=CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=0.005),
        samplers_size=DATASET_SIZE,
    )
    ds_gen.generate()

    codec = MWFilterTouchstoneCodec.from_dataset(mwlab.TouchstoneDataset(source=ds_gen.path_to_dataset, in_memory=True, s_tf=ds_gen.y_transform))
    codec.exclude_keys(["f0", "bw", "N", "Q"])
    print(codec)
    codec.y_channels = ['S1_1.real', 'S2_1.real', 'S2_2.real', 'S1_1.imag', 'S2_1.imag', 'S2_2.imag']

    # Исключаем из анализа ненужные x-параметры
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    dm = mwlab.TouchstoneLDataModule(
        source=ds_gen.path_to_dataset,         # Путь к датасету
        codec=codec,                   # Кодек для преобразования TouchstoneData → (x, y)
        batch_size=64,                 # Размер батча
        val_ratio=0.2,                 # Доля валидационного набора
        test_ratio=0.05,                # Доля тестового набора
        cache_size=None,
        scaler_in=MinMaxScaler(dim=(0, 2)),                          # Скейлер для входных данных
        scaler_out=MinMaxScaler(dim=0, feature_range=(-0.5, 0.5)),  # Скейлер для выходных данных
        swap_xy=True,
        # Параметры базового датасета:
        base_ds_kwargs={
            # "x_tf": x_transform,       # Предобработка входных данных
            "s_tf": ds_gen.y_transform,        # Предобработка выходных данных
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

    model = Simple_Opt_3(len(ds_gen.origin_filter.coupling_matrix.links))
    lit_model = mwlab.BaseLModule(
        model=model,  # Наша нейросетевая модель
        swap_xy=True,
        scaler_in=dm.scaler_in,  # Скейлер для входных данных
        scaler_out=dm.scaler_out,  # Скейлер для выходных данных
        codec=codec,  # Кодек для преобразования данных
        optimizer_cfg={"name": "Adam", "lr": 2e-3},  # Конфигурация оптимизатора
        scheduler_cfg={"name": "StepLR", "step_size": 20, "gamma": 0.5},
        loss_fn=nn.MSELoss()
    )
    stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.00001)

    # Обучение модели с помощью PyTorch Lightning
    trainer = L.Trainer(
        max_epochs=100,  # Максимальное количество эпох обучения
        accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
        log_every_n_steps=100,  # Частота логирования в процессе обучения
        callbacks=[
            stoping
        ]
    )

    # Запуск процесса обучения
    trainer.fit(lit_model, dm)

    # Возьмем для примера первый touchstone-файл из тестового набора данных
    test_tds = dm.get_dataset(split="test", meta=True)
    # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
    y_t, x_t, meta = test_tds[0]  # Используем первый файл набора данных

    # Декодируем данные
    orig_prms = dm.codec.decode_x(x_t)  # Создаем словарь параметров
    net = dm.codec.decode_s(y_t, meta)  # Создаем объект skrf.Network

    # Предсказанные S-параметры
    pred_prms = lit_model.predict_x(net)

    print(f"Исходные параметры: {orig_prms}")
    print(f"Предсказанные параметры: {pred_prms}")

    orig_cm = MWFilter.matrix_from_touchstone_data_parameters(orig_prms)
    pred_cm = MWFilter.matrix_from_touchstone_data_parameters(pred_prms)
    plt.show()


if __name__ == "__main__":
    main()
    plt.show()
