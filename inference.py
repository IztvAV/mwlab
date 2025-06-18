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


def inference_for_codec(model_name: str, path_to_model: str, codec: MWFilterTouchstoneCodec):
    model = common.get_model(model_name)
    inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
        checkpoint_path=path_to_model,
        model=model
    )
    return inference_model


def predict_for_codec(path_to_inference_model: str, codec: MWFilterTouchstoneCodec, dm: TouchstoneLDataModule):
    dm.codec = codec
    dm.setup("fit")
    test_tds = dm.get_dataset(split="test", meta=True)
    # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
    y_t, x_t, meta = test_tds[-1]  # Используем первый файл набора данных]

    # Декодируем данные
    orig_prms = dm.codec.decode_x(x_t)  # Создаем словарь параметров
    net = dm.codec.decode_s(y_t, meta)  # Создаем объект skrf.Network

    # Предсказанные S-параметры
    inference_model = inference_for_codec(path_to_inference_model, codec)
    pred_prms = inference_model.predict_x(net)
    return orig_prms, pred_prms, meta


def predict_for_test_dataset(codec_main_coupling: MWFilterTouchstoneCodec, codec_self_coupling: MWFilterTouchstoneCodec,
                             codec_cross_coupling: MWFilterTouchstoneCodec, base_fil: MWFilter, dm: TouchstoneLDataModule,
                             path_to_best_model_for_main_coupling="saved_models/EAMU4-KuIMUXT3-BPFC1/best-epoch=62-val_loss=0.00234-train_loss=0.00272-val_r2=0.97111.ckpt",
                             path_to_best_model_for_self_coupling="saved_models/EAMU4-KuIMUXT3-BPFC1/best-epoch=11-val_loss=0.00014-train_loss=0.00011-val_r2=0.99824.ckpt",
                             path_to_best_model_for_cross_coupling="saved_models/EAMU4-KuIMUXT3-BPFC1/best-epoch=16-val_loss=0.04654-train_loss=0.04326-val_r2=0.43075.ckpt"):

    orig_main_couplings, pred_main_couplings, meta = predict_for_codec(path_to_best_model_for_main_coupling, codec_main_coupling,
                                                                       dm)
    print(f"Исходные main couplings параметры: {orig_main_couplings}")
    print(f"Предсказанные main couplings параметры: {pred_main_couplings}")

    orig_self_couplings, pred_self_couplings, meta = predict_for_codec(path_to_best_model_for_self_coupling, codec_self_coupling,
                                                                 dm)
    print(f"Исходные self couplings параметры: {orig_self_couplings}")
    print(f"Предсказанные self couplings параметры: {pred_self_couplings}")


    orig_cross_couplings, pred_cross_couplings, meta = predict_for_codec(path_to_best_model_for_cross_coupling, codec_cross_coupling,
                                                                        dm)
    print(f"Исходные cross couplings параметры: {orig_cross_couplings}")
    print(f"Предсказанные cross couplings параметры: {pred_cross_couplings}")

    orig_couplings = {**orig_main_couplings, **orig_self_couplings, **orig_cross_couplings}
    pred_couplings = {**pred_main_couplings, **pred_self_couplings, **pred_cross_couplings}

    inference_model = inference_for_codec(
        path_to_model=path_to_best_model_for_main_coupling,
        codec=codec_main_coupling
    )

    pred_fil = inference_model.create_filter_from_prediction(base_fil, pred_couplings, meta)
    orig_fil = inference_model.create_filter_from_prediction(base_fil, orig_couplings, meta)

    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)

    optim_matrix = optimize_cm(pred_fil, orig_fil)
    error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    error_matrix_pred.plot_matrix(title="Predict de-tuned matrix errors", cmap="YlOrBr")
    error_matrix_optim.plot_matrix(title="Optim de-tuned matrix errors", cmap="YlOrBr")
    pred_fil.coupling_matrix.plot_matrix(title="Predict de-tuned matrix")
    optim_matrix.plot_matrix(title="Optimized de-tuned matrix")
    orig_fil.coupling_matrix.plot_matrix(title="Origin de-tuned matrix")


def predict_for_filter(codec_main_coupling: MWFilterTouchstoneCodec, codec_self_coupling: MWFilterTouchstoneCodec,
                       codec_cross_coupling: MWFilterTouchstoneCodec, base_fil: MWFilter, dm: TouchstoneLDataModule,
                       meta: dict, orig_fil: MWFilter,
                       path_to_best_model_for_main_coupling="saved_models/EAMU4-KuIMUXT3-BPFC1/best-epoch=62-val_loss=0.00234-train_loss=0.00272-val_r2=0.97111.ckpt",
                       path_to_best_model_for_self_coupling="saved_models/EAMU4-KuIMUXT3-BPFC1/best-epoch=11-val_loss=0.00014-train_loss=0.00011-val_r2=0.99824.ckpt",
                       path_to_best_model_for_cross_coupling="saved_models/EAMU4-KuIMUXT3-BPFC1/best-epoch=16-val_loss=0.04654-train_loss=0.04326-val_r2=0.43075.ckpt"):
    dm.codec = codec_main_coupling
    dm.setup("fit")
    inference_model = inference_for_codec(path_to_best_model_for_main_coupling, codec_main_coupling)
    main_couplings = inference_model.predict_x(orig_fil)

    dm.codec = codec_self_coupling
    dm.setup("fit")
    inference_model = inference_for_codec(path_to_best_model_for_self_coupling, codec_self_coupling)
    self_couplings = inference_model.predict_x(orig_fil)

    dm.codec = codec_cross_coupling
    dm.setup("fit")
    inference_model = inference_for_codec(path_to_best_model_for_cross_coupling, codec_cross_coupling)
    cross_couplings = inference_model.predict_x(orig_fil)

    pred_couplings = {**main_couplings, **self_couplings, **cross_couplings}
    pred_fil = inference_model.create_filter_from_prediction(base_fil, pred_couplings, meta)

    inference_model.plot_origin_vs_prediction(orig_fil, pred_fil)
    optim_matrix = optimize_cm(pred_fil, orig_fil)
    error_matrix_pred = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, pred_fil.coupling_matrix)
    error_matrix_optim = CouplingMatrix.error_matrix(orig_fil.coupling_matrix, optim_matrix)
    error_matrix_pred.plot_matrix(title="Predict tuned matrix errors", cmap="YlOrBr")
    error_matrix_optim.plot_matrix(title="Optim tuned matrix errors", cmap="YlOrBr")
    pred_fil.coupling_matrix.plot_matrix(title="Predict tuned matrix")
    optim_matrix.plot_matrix(title="Optimized tuned matrix")
    orig_fil.coupling_matrix.plot_matrix(title="Origin tuned matrix")

