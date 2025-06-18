import os
import time

from mwlab.nn.scalers import MinMaxScaler, StdScaler
from mwlab import TouchstoneDataset, TouchstoneLDataModule, TouchstoneDatasetAnalyzer
from mwlab.io.backends import RAMBackend
from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample

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

torch.set_float32_matmul_precision("medium")



def main():
    L.seed_everything(0)
    print("Создаем фильтр")
    orig_filter = common.create_origin_filter(configs.ENV_ORIGIN_DATA_PATH)
    print("Создаем сэмплеры")
    samplers = common.create_lhs_samplers(orig_filter)
    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_save_dataset=os.path.join(configs.ENV_DATASET_PATH, samplers.cms.type.name, f"{len(samplers.cms)}"),
        backend_type='ram',
        orig_filter=orig_filter,
        filename="Dataset",
    )
    ds_gen.generate(samplers)

    ds = TouchstoneDataset(source=ds_gen.backend, in_memory=True)
    # plot_distribution(ds, num_params=len(ds_gen.origin_filter.coupling_matrix.links))
    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds=ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in orig_filter.coupling_matrix.links])

    codec_main_coupling = MWFilterTouchstoneCodec.from_dataset(ds=ds,
                                                 keys_for_analysis=["m_0_1", "m_1_2", "m_2_3", "m_3_4", "m_4_5", "m_5_6",
                                                                    "m_6_7", "m_7_8", "m_8_9", "m_9_10", "m_10_11",
                                                                    "m_11_12", "m_12_13"])
    codec_self_coupling = MWFilterTouchstoneCodec.from_dataset(ds=ds,
                                                 keys_for_analysis=["m_1_1", "m_2_2", "m_3_3", "m_4_4", "m_5_5", "m_6_6",
                                                                    "m_7_7", "m_8_8", "m_9_9", "m_10_10", "m_11_11",
                                                                    "m_12_12"])
    codec_cross_coupling = MWFilterTouchstoneCodec.from_dataset(ds=ds,
                                                               keys_for_analysis=["m_2_11", "m_3_10",
                                                                                  "m_4_9",
                                                                                  "m_5_8"])
    codecs = [codec_main_coupling, codec_self_coupling, codec_cross_coupling]
    codec_test = MWFilterTouchstoneCodec.from_dataset(ds=ds, keys_for_analysis=["m_0_1", "m_1_2", "m_2_3", "m_3_4", "m_4_5", "m_5_6",
                                                                    "m_6_7", "m_7_8", "m_8_9", "m_9_10", "m_10_11",
                                                                    "m_11_12", "m_12_13", "m_1_1", "m_2_2", "m_3_3", "m_4_4", "m_5_5", "m_6_6",
                                                                    "m_7_7", "m_8_8", "m_9_9", "m_10_10", "m_11_11",
                                                                    "m_12_12"])
    codec = codec

    stats = ""

    dm = TouchstoneLDataModule(
        source=ds_gen.backend,  # Путь к датасету
        codec=codec,  # Кодек для преобразования TouchstoneData → (x, y)
        batch_size=configs.BATCH_SIZE,  # Размер батча
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

    print(codec)
    print("Каналы Y:", codec.y_channels)
    print("Каналы X:", codec.x_keys)
    print("Количество каналов:", len(codec.y_channels))


    main = models.ResNet1DFlexible(
        in_channels=len(codec.y_channels),
        out_channels=len(codec.x_keys),
        num_blocks=[1, 4, 3, 6],
        layer_channels=[64, 64, 128, 256],
        first_conv_kernel=8,
        first_conv_channels=64,
        first_maxpool_kernel=2,
        activation_in='sigmoid',
        activation_block='swish',
        use_se=False,
        se_reduction=32
    )

    mlp = models.CorrectionMLP(
        input_dim=len(codec.x_keys),
        output_dim=len(codec.x_keys),
        hidden_dims=[8, 64, 4096],
        activation_fun='soft_sign'
    )

    model = models.ModelWithCorrection(
        main_model=main,
        correction_model=mlp,
    )

    dm.setup("fit")

    # Декодирование
    _, __, meta = dm.get_dataset(split="train", meta=True)[0]

    # Печатаем размеры полученных наборов
    print(f"Размер тренировочного набора: {len(dm.train_ds)}")
    print(f"Размер валидационного набора: {len(dm.val_ds)}")
    print(f"Размер тестового набора: {len(dm.test_ds)}")

    lit_model = MWFilterBaseLMWithMetrics(
        model=model,  # Наша нейросетевая модель
        swap_xy=True,
        scaler_in=dm.scaler_in,  # Скейлер для входных данных
        scaler_out=dm.scaler_out,  # Скейлер для выходных данных
        codec=codec,  # Кодек для преобразования данных
        optimizer_cfg={"name": "Adam", "lr": 0.0007526812333573349},
        scheduler_cfg={"name": "StepLR", "step_size": 14, "gamma": 0.15},
        # loss_fn=CustomLosses("error"),
        loss_fn=nn.MSELoss()
    )

    stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_mse", patience=20, mode="min", min_delta=0.00001)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="val_mse", dirpath="saved_models/" + configs.FILTER_NAME,
                                                     filename="best-{epoch}-{val_loss:.5f}-{train_loss:.5f}-{val_r2:.5f}",
                                                     mode="min",
                                                     save_top_k=1,  # Сохраняем только одну лучшую
                                                     save_weights_only=False,
                                                     # Сохранять всю модель (включая структуру)
                                                     verbose=False  # Отключаем логирование сохранения)
                                                     )

    # Обучение модели с помощью PyTorch Lightning
    trainer = L.Trainer(
        deterministic=True,
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
    #     checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=47-val_loss=0.01193-train_loss=0.01149-val_r2=0.85343.ckpt",
    #     model=model
    # ).to(lit_model.device)

    # Запуск процесса обучения
    trainer.fit(lit_model, dm)
    print(f"Best model saved into: {checkpoint.best_model_path}")

    # Загружаем лучшую модель
    inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
        # checkpoint_path="saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=12-val_loss=0.01266-train_loss=0.01224.ckpt",
        # checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=47-val_loss=0.01193-train_loss=0.01149-val_r2=0.85343.ckpt",
        checkpoint_path=checkpoint.best_model_path,
        model=model
    ).to(lit_model.device)
    orig_fil, pred_fil = inference_model.predict(dm, idx=0)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    optim_matrix = optimize_cm(pred_fil, orig_fil)
    error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)

    error_matrix_optim.plot_matrix(title="Optimized de-tuned matrix errors", cmap="YlOrBr")
    optim_matrix.plot_matrix(title="Optimized de-tuned matrix")
    error_matrix_pred.plot_matrix(title="Predict de-tuned matrix errors", cmap="YlOrBr")
    pred_fil.coupling_matrix.plot_matrix(title="Predict de-tuned matrix")
    orig_fil.coupling_matrix.plot_matrix(title="Origin de-tuned matrix")

    # Предсказываем эталонный фильтр
    orig_fil = ds_gen.origin_filter
    pred_prms = inference_model.predict_x(orig_fil)
    pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, meta)
    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    optim_matrix = optimize_cm(pred_fil, orig_fil)
    error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    error_matrix_optim.plot_matrix(title="Optimized tuned matrix errors")
    error_matrix_pred.plot_matrix(title="Predict tuned matrix errors")
    orig_fil.coupling_matrix.plot_matrix(title="Origin tuned matrix")

    # start_time = time.time()
    # predict_for_test_dataset(codec_main_coupling=codec_main_coupling, codec_self_coupling=codec_self_coupling,
    #                          codec_cross_coupling=codec_cross_coupling,
    #                          path_to_best_model_for_main_coupling="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=54-val_loss=0.00078-train_loss=0.00076-val_r2=0.98951.ckpt",
    #                          path_to_best_model_for_self_coupling="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=11-val_loss=0.00006-train_loss=0.00006-val_r2=0.99925.ckpt",
    #                          path_to_best_model_for_cross_coupling="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=23-val_loss=0.04211-train_loss=0.03836-val_r2=0.43820.ckpt",
    #                          base_fil=ds_gen.origin_filter, dm=dm)
    # stop_time = time.time()
    # print(f"Prediction time: {stop_time - start_time:.3f} sec")
    #
    # dm.setup("fit")
    # _, __, meta = dm.get_dataset(split="test", meta=True)[0]
    # start_time = time.time()
    # predict_for_filter(codec_main_coupling=codec_main_coupling, codec_self_coupling=codec_self_coupling,
    #                    codec_cross_coupling=codec_cross_coupling,
    #                    path_to_best_model_for_main_coupling="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=54-val_loss=0.00078-train_loss=0.00076-val_r2=0.98951.ckpt",
    #                    path_to_best_model_for_self_coupling="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=11-val_loss=0.00006-train_loss=0.00006-val_r2=0.99925.ckpt",
    #                    path_to_best_model_for_cross_coupling="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=23-val_loss=0.04211-train_loss=0.03836-val_r2=0.43820.ckpt",
    #                    base_fil=ds_gen.origin_filter, meta=meta, orig_fil=ds_gen.origin_filter,
    #                    dm=dm)
    # stop_time = time.time()
    # print(f"Prediction time: {stop_time - start_time:.3f} sec")

    # paths = [
    #     "saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=54-val_loss=0.00078-train_loss=0.00076-val_r2=0.98951.ckpt",
    #     "saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=11-val_loss=0.00006-train_loss=0.00006-val_r2=0.99925.ckpt",
    #     "saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=23-val_loss=0.04211-train_loss=0.03836-val_r2=0.43820.ckpt"
    # ]
    #
    # for codec, path in zip(codecs, paths):
    #     main = models.ResNet1DFlexible(
    #         in_channels=len(codec.y_channels),
    #         out_channels=len(codec.x_keys),
    #         num_blocks=[1, 4, 3, 5],
    #         layer_channels=[64, 64, 128, 256],
    #         first_conv_kernel=8,
    #         first_conv_channels=64,
    #         first_maxpool_kernel=2,
    #         activation_in='sigmoid',
    #         activation_block='swish',
    #         use_se=False,
    #         se_reduction=1
    #     )
    #
    #     mlp = models.CorrectionMLP(
    #         input_dim=len(codec.x_keys),
    #         output_dim=len(codec.x_keys),
    #         hidden_dims=[128, 256, 512],
    #         activation_fun='swish'
    #     )
    #
    #     model = models.ModelWithCorrection(
    #         main_model=main,
    #         correction_model=mlp,
    #     )
    #
    #     # inference_model = inference_for_codec(path, codec)
    #
    #     print(codec)
    #     print("Каналы Y:", codec.y_channels)
    #     print("Каналы X:", codec.x_keys)
    #     print("Количество каналов:", len(codec.y_channels))
    #
    #     dm = TouchstoneLDataModule(
    #         source=ds_gen.backend,  # Путь к датасету
    #         codec=codec,  # Кодек для преобразования TouchstoneData → (x, y)
    #         batch_size=BATCH_SIZE,  # Размер батча
    #         val_ratio=0.2,  # Доля валидационного набора
    #         test_ratio=0.05,  # Доля тестового набора
    #         cache_size=0,
    #         scaler_in=MinMaxScaler(dim=(0, 2), feature_range=(0, 1)),  # Скейлер для входных данных
    #         scaler_out=MinMaxScaler(dim=0, feature_range=(-0.5, 0.5)),  # Скейлер для выходных данных
    #         swap_xy=True,
    #         num_workers=0,
    #         # Параметры базового датасета:
    #         base_ds_kwargs={
    #             "in_memory": True
    #         }
    #     )
    #     dm.setup("fit")
    #
    #     # Декодирование
    #     _, __, meta = dm.get_dataset(split="train", meta=True)[0]
    #
    #     # Печатаем размеры полученных наборов
    #     print(f"Размер тренировочного набора: {len(dm.train_ds)}")
    #     print(f"Размер валидационного набора: {len(dm.val_ds)}")
    #     print(f"Размер тестового набора: {len(dm.test_ds)}")
    #
    #     lit_model = MWFilterBaseLMWithMetrics(
    #         model=model,  # Наша нейросетевая модель
    #         swap_xy=True,
    #         scaler_in=dm.scaler_in,  # Скейлер для входных данных
    #         scaler_out=dm.scaler_out,  # Скейлер для выходных данных
    #         codec=codec,  # Кодек для преобразования данных
    #         optimizer_cfg={"name": "Adam", "lr": 0.0006859984857331174},
    #         scheduler_cfg={"name": "StepLR", "step_size": 15, "gamma": 0.1},
    #         loss_fn=CustomLosses("error"),
    #         # loss_fn=nn.MSELoss()
    #     )
    #
    #     stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_mse", patience=20, mode="min", min_delta=0.00001)
    #     checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="val_mse", dirpath="saved_models/" + FILTER_NAME,
    #                                                      filename="best-{epoch}-{val_loss:.5f}-{train_loss:.5f}-{val_r2:.5f}",
    #                                                      mode="min",
    #                                                      save_top_k=1,  # Сохраняем только одну лучшую
    #                                                      save_weights_only=False,
    #                                                      # Сохранять всю модель (включая структуру)
    #                                                      verbose=False  # Отключаем логирование сохранения)
    #                                                      )
    #
    #     # Обучение модели с помощью PyTorch Lightning
    #     trainer = L.Trainer(
    #         deterministic=True,
    #         max_epochs=150,  # Максимальное количество эпох обучения
    #         accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
    #         log_every_n_steps=100,  # Частота логирования в процессе обучения
    #         callbacks=[
    #             stoping,
    #             checkpoint
    #         ]
    #     )
    #     # # Загружаем лучшую модель
    #     # lit_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
    #     #     checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=47-val_loss=0.01193-train_loss=0.01149-val_r2=0.85343.ckpt",
    #     #     model=model
    #     # ).to(lit_model.device)
    #
    #
    #     # Запуск процесса обучения
    #     trainer.fit(lit_model, dm)
    #     print(f"Best model saved into: {checkpoint.best_model_path}")
    #     stats += checkpoint.best_model_path+"\n"
    #
    # print(f"Total stats: \n{stats}")

    # # Загружаем лучшую модель
    # inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
    #     # checkpoint_path="saved_models\\SCYA501-KuIMUXT5-BPFC3\\best-epoch=12-val_loss=0.01266-train_loss=0.01224.ckpt",
    #     # checkpoint_path="saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=47-val_loss=0.01193-train_loss=0.01149-val_r2=0.85343.ckpt",
    #     checkpoint_path=checkpoint.best_model_path,
    #     model=model
    # ).to(lit_model.device)
    # orig_fil, pred_fil = inference_model.predict(dm, idx=0)
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
    # # Предсказываем эталонный фильтр
    # orig_fil = ds_gen.origin_filter
    # pred_prms = inference_model.predict_x(orig_fil)
    # pred_fil = inference_model.create_filter_from_prediction(orig_fil, pred_prms, meta)
    # inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    # optim_matrix = optimize_cm(pred_fil, orig_fil)
    # error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    # error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    # error_matrix_optim.plot_matrix(title="Optimized tuned matrix errors")
    # error_matrix_pred.plot_matrix(title="Predict tuned matrix errors")
    # orig_fil.coupling_matrix.plot_matrix(title="Origin tuned matrix")
    plt.show()


if __name__ == "__main__":
    main()
    plt.show()
