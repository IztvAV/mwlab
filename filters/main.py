import os
import mwlab
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.transforms import TComposite
from mwlab.nn.scalers import MinMaxScaler

from filters import TouchstoneMWFilterDataset, MWFilter, CouplingMatrix
from filters import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers
from filters import Sampler, SamplerTypes

import matplotlib.pyplot as plt
import numpy as np
import lightning as L

from torch import nn

ENV_ORIGIN_DATA_PATH = os.getcwd() + "\\Data\\origins_data"
ENV_DATASET_PATH = os.getcwd() + "\\Data\\datasets_data"


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


def create_min_max_matrices(origin_matrix: CouplingMatrix, deltas: np.array):
    Mmin = origin_matrix.matrix
    Mmax = origin_matrix.matrix

    for (i, j) in origin_matrix.links:
        if i == j:
            Mmin[i][j] -= deltas[0]
            Mmax[i][j] += deltas[0]
        elif j == (i + 1):
            Mmin[i][j] -= deltas[1]
            Mmin[j][i] -= deltas[1]
            Mmax[i][j] += deltas[1]
            Mmax[j][i] += deltas[1]
        else:
            Mmin[i][j] -= deltas[2]
            Mmin[j][i] -= deltas[2]
            Mmax[i][j] += deltas[2]
            Mmax[j][i] += deltas[2]
    return Mmin, Mmax


def create_min_max_phase_shifts(origin_shifts: np.array, deltas):
    phase_shifts_min = origin_shifts - deltas
    phase_shifts_max = origin_shifts + deltas
    return phase_shifts_min, phase_shifts_max


def main():
    # tds = TouchstoneMWFilterDataset(source=ENV_ORIGIN_DATA_PATH)
    tds = mwlab.TouchstoneDataset(source=ENV_ORIGIN_DATA_PATH)
    print(f"Загружено файлов: {len(tds)}")
    print("Пример параметров из первого файла:")
    print(tds[0][0], "\n")
    origin_filter: MWFilter = MWFilter.from_touchstone_dataset_item(tds[0])
    y_transform = TComposite([
        S_Crop(f_start=origin_filter.f0-origin_filter.bw*1.2, f_stop=origin_filter.f0+origin_filter.bw*1.2, unit='MHz'),
        S_Resample(301)
    ])
    tds_transformed = TouchstoneMWFilterDataset(source=ENV_ORIGIN_DATA_PATH, s_tf=y_transform)
    codec = mwlab.TouchstoneCodec.from_dataset(tds_transformed)
    print(codec)
    codec.y_channels = ['S1_1.real', 'S2_1.real', 'S2_2.real', 'S1_1.imag', 'S2_1.imag', 'S2_2.imag']

    # Исключаем из анализа ненужные x-параметры
    keys_to_exclude = ["f0", "bw", "N", "Q"]  # Q исключаю сейчас для простоты
    codec.x_keys = list(filter(lambda x: x not in keys_to_exclude, codec.x_keys))
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    # Пример кодирования и декодирования
    prms, origin_filter = tds_transformed[0]  # Используем первый файл набора
    ts = mwlab.TouchstoneData(origin_filter, prms)  # Создаем объект TouchstoneData

    # Кодирование
    x, y, meta = codec.encode(ts)
    print("x-параметры:\n", x)
    print("y-параметры:\n", y)
    print("Метаданные:\n" + str(meta))

    m_min, m_max = create_min_max_matrices(origin_matrix=origin_filter.coupling_matrix, deltas=np.array([1.5, 0.1, 0.005]))
    phase_shifts_min, phase_shifts_max = create_min_max_phase_shifts(origin_shifts=np.array([0.547, -1.0, 0.01685, 0.017]),
                                                                     deltas=np.array([0.02, 0.02, 0.005, 0.005]))
    samplers = CMTheoreticalDatasetGeneratorSamplers(
        cm_shifts=Sampler.lhs(start=m_min, stop=m_max, num=10000),
        ps_shifts=Sampler.lhs(start=phase_shifts_min, stop=phase_shifts_max, num=10000)
    )

    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_save_dataset=ENV_DATASET_PATH,
        origin_filter=origin_filter,
        samplers=samplers
    )
    plt.figure()
    # ds_gen.generate()

    dm = mwlab.TouchstoneLDataModule(
        source=ENV_DATASET_PATH,         # Путь к датасету
        codec=codec,                   # Кодек для преобразования TouchstoneData → (x, y)
        batch_size=64,                 # Размер батча
        val_ratio=0.2,                 # Доля валидационного набора
        test_ratio=0.1,                # Доля тестового набора
        cache_size=None,
        scaler_in=MinMaxScaler(dim=(0, 2)),                          # Скейлер для входных данных
        scaler_out=MinMaxScaler(dim=0, feature_range=(-0.5, 0.5)),  # Скейлер для выходных данных
        swap_xy=True,
        # Параметры базового датасета:
        base_ds_kwargs={
            # "x_tf": x_transform,       # Предобработка входных данных
            "s_tf": y_transform        # Предобработка выходных данных
        }
    )

    dm.setup("fit")

    # Декодирование
    ts_rec = codec.decode(dm.train_ds.dataset[0][0], meta)
    plt.figure()
    ts_rec.network.plot_s_db(m=0, n=0, label='S11 from dataset')
    ts_rec.network.plot_s_db(m=1, n=0, label='S21 from dataset')

    # Печатаем размеры полученных наборов
    print(f"Размер тренировочного набора: {len(dm.train_ds)}")
    print(f"Размер валидационного набора: {len(dm.val_ds)}")
    print(f"Размер тестового набора: {len(dm.test_ds)}")

    model = Simple_Opt_3(len(origin_filter.coupling_matrix.links))
    lit_model = mwlab.BaseLModule(
        model=model,  # Наша нейросетевая модель
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
    dm.setup("predict")
    prediction = trainer.predict(lit_model, datamodule=dm)

    # Извлекаем предсказания для первого файла
    pred_ts = prediction[0][0]

    # Оригинальные и предсказанные S-параметры
    orig = codec.decode(dm.predict_ds[0][0], meta=dm.predict_ds[0][-1]).network
    pred = pred_ts.network

    # Устанавливаем единицы измерения частоты в ГГц
    orig.frequency.unit = "GHz"
    pred.frequency.unit = "GHz"

    # Построение графиков для сравнения S11 и S21
    plt.figure()

    # S11: оригинал и предсказание
    orig.plot_s_db(m=0, n=0, color='r', label='S11 original')
    pred.plot_s_db(m=0, n=0, color='b', ls=':', label='S11 predicted')

    # S21: оригинал и предсказание
    orig.plot_s_db(m=0, n=1, color='m', label='S21 original')
    pred.plot_s_db(m=0, n=1, color='k', ls=':', label='S21 predicted')

    plt.legend()
    plt.title("Сравнение оригинальных и предсказанных S-параметров")
    plt.xlabel("Частота (ГГц)")
    plt.ylabel("Амплитуда (дБ)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
    plt.show()
