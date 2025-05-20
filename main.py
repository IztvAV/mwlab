import os
import time

from mwlab.nn.scalers import MinMaxScaler
from mwlab import TouchstoneDataset, TouchstoneLDataModule
from mwlab.io.backends import RAMBackend

from filters import CMTheoreticalDatasetGenerator, CouplingMatrix
from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLModule, MWFilterBaseLMWithMetrics
from filters.filter import MWFilter

from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift

import matplotlib.pyplot as plt
import lightning as L

from torch import nn
import models
import torch
from scipy.optimize import minimize

torch.set_float32_matmul_precision("medium")

to_db = lambda x: 20 * torch.log10(abs(x))

BATCH_SIZE = 64
DATASET_SIZE = 300_000
FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "FilterData", FILTER_NAME, "datasets_data")


class FastMN2toSParamCalculation:
    def __init__(self, matrix_order, w_min, w_max, w_num, fbw, Q):
        self.matrix_order = matrix_order
        self.w = torch.linspace(w_min, w_max, w_num)
        self.R = torch.zeros(matrix_order, matrix_order, dtype=torch.complex64)
        self.R[0, 0] = 1j
        self.R[-1, -1] = 1j
        self.I = torch.eye(matrix_order, matrix_order, dtype=torch.complex64)
        self.I[0, 0] = 0
        self.I[-1, -1] = 0
        self.G = 1j*1 * torch.eye(matrix_order, matrix_order, dtype=torch.complex64)
        self.G[0, 0] = 0
        self.G[-1, -1] = 0
        for res in range(1, matrix_order - 1):
            self.G[res, res] = 1 / (fbw * Q)
        self.S11 = torch.zeros(w_num, dtype=torch.complex64)
        self.S21 = torch.zeros(w_num, dtype=torch.complex64)
        self.S22 = torch.zeros(w_num, dtype=torch.complex64)

    def RespM2_gpu(self, M):
        # Батчевое создание матриц A
        w = self.w.view(-1, 1, 1)  # (B, 1, 1)
        MR = torch.tensor(M) - self.R
        A = MR + w * self.I - self.G

        # Обратные матрицы
        Ainv = torch.linalg.inv(A)  # (B, N, N)

        # Расчет S-параметров
        A00 = Ainv[:, 0, 0]
        # ANN = Ainv[:, -1, -1]
        AN0 = Ainv[:, -1, 0]

        S11 = 1 + 2j * 1 * A00
        # S22 = 1 + 2j * Rl * ANN
        S21 = -2j * torch.sqrt(torch.tensor(1 * 1, dtype=torch.float32)) * AN0

        return self.w, S11, S21


def optimize_cm(pred_filter:MWFilter, orig_filter: MWFilter):
    def cost(x, *args):
        """ х - элементы матрицы связи (сначала главная диагональ D, потом D+1, потом побочная d, потом d+1"""
        fast_calc, orig_filter, s11_origin, s21_origin = args
        matrix = CouplingMatrix.from_factors(x, orig_filter.coupling_matrix.links, orig_filter.coupling_matrix.matrix_order)
        _, s11_pred, s21_pred = fast_calc.RespM2_gpu(matrix)
        cost = torch.sum(torch.abs(s21_origin - to_db(s21_pred))) + torch.sum(torch.abs(s11_origin - to_db(s11_pred)))
        return cost.item()

    x0_real = pred_filter.coupling_matrix.factors
    x0 = x0_real
    x0 = torch.round(torch.tensor(x0), decimals=5)
    print("Start optimize")
    fast_calc = FastMN2toSParamCalculation(matrix_order=orig_filter.coupling_matrix.matrix_order, w_min=-1.2, w_max=1.2, w_num=301, Q=orig_filter.Q, fbw=orig_filter.fbw)
    _, s11_origin, s21_origin = fast_calc.RespM2_gpu(orig_filter.coupling_matrix.matrix)
    s11_origin_db = to_db(s11_origin)
    s21_origin_db = to_db(s21_origin)
    start_time = time.time_ns()
    prev_cost = 0
    for _ in range(15):
        optim_res = minimize(fun=cost, x0=x0, jac="2-points", method="BFGS",
                             args=(fast_calc, orig_filter, s11_origin_db, s21_origin_db),
                             options={"disp": True, "maxiter": 50})
        x0 = optim_res.x

        if optim_res.nit == 0:
            print("Number of iteration is 0. Break loop")
            break
        elif abs(optim_res.fun - prev_cost) < 1e-2:
            print("Different between cost function values less than 1e-2. Break loop")
            break
        elif abs(optim_res.fun) < 1:
            print("Cost function value less than 1. Break loop")
            break
        prev_cost = optim_res.fun
    stop_time = time.time_ns()
    print(f"Optimize time: {(stop_time - start_time) / 1e9} sec")

    optim_matrix = CouplingMatrix.from_factors(optim_res.x, orig_filter.coupling_matrix.links,
                                               orig_filter.coupling_matrix.matrix_order)
    w, s11_optim_resp, s21_optim_resp = fast_calc.RespM2_gpu(optim_matrix)
    s11_optim_db = to_db(s11_optim_resp)
    s21_optim_db = to_db(s21_optim_resp)

    plt.figure()
    plt.title("S11")
    plt.plot(w, s11_origin_db, w, s11_optim_db)
    plt.legend(["Origin", "Optimized"])

    plt.figure()
    plt.title("S21")
    plt.plot(w, s21_origin_db, w, s21_optim_db)
    plt.legend(["Origin", "Optimized"])

    return optim_res.x



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
    # model = models.ResNet1DFlexible(
    #     in_channels=8,  # По вашему исходному коду
    #     out_channels=30,  # По вашему исходному коду
    #     first_conv_channels=64,
    #     first_conv_kernel=10,
    #     layer_channels=[128, 64, 64, 128],
    #     num_blocks=[5, 1, 3, 2],
    # )
    # model = models.ResNetRNN1D(in_channels=len(codec.y_channels),
    #                            out_channels=len(ds_gen.origin_filter.coupling_matrix.links),
    #                            resnet_hidden_size=len(ds_gen.origin_filter.coupling_matrix.links),
    #                            rnn_hidden_size=32,
    #                            rnn_type='lstm',
    #                            rnn_layers=1,
    #                            bidirectional=False)
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
        optimizer_cfg={"name": "Adam", "lr": 0.01},  # Конфигурация оптимизатора
        scheduler_cfg={"name": "StepLR", "step_size": 20, "gamma": 0.5},
        # optimizer_cfg={"name": "SGD", "lr": 0.1, "momentum": 0.99, "nesterov": True},
        # scheduler_cfg={"name": "CosineAnnealingWarmRestarts", "T_0": 4, "T_mult": 2, "eta_min": 1e-5},
        # optimizer_cfg={"name": "AdamW", "lr": 3e-4},
        # scheduler_cfg={"name": "OneCycleLR", "max_lr": 3e-3, "epochs": 50, "steps_per_epoch": len(dm.train_ds), "pct_start": 0.2},
        loss_fn=nn.MSELoss()
    )


    stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.00001)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", dirpath="saved_models/"+FILTER_NAME,
                                                     filename="best-{epoch}-{val_loss:.5f}",
                                                     mode="min",
                                                     save_top_k=1,                                  # Сохраняем только одну лучшую
                                                     save_weights_only=False,                       # Сохранять всю модель (включая структуру)
                                                     verbose=False                                  # Отключаем логирование сохранения)
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

    # Запуск процесса обучения
    # trainer.fit(lit_model, dm)
    # print(f"Best model saved into: {checkpoint.best_model_path}")

    # Загружаем лучшую модель
    inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
        checkpoint_path="saved_models/SCYA501-KuIMUXT5-BPFC3/best-epoch=23-val_loss=0.01887.ckpt",
        model=model
    ).to(lit_model.device)
    orig_fil, pred_fil = inference_model.predict(dm, idx=0)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)


    optim_factors = optimize_cm(pred_fil, orig_fil)
    plt.show()


if __name__ == "__main__":
    main()
    plt.show()
