import os
import random
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from common import WorkModel
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
import skrf as rf


work_model: WorkModel|None = None
inference_model = None


def load_model(fil_name: str="EAMU4T1-BPFC2"):
    """ Пока в таком формате. В дальнейшем надо будет стандартизировать названия моделей """
    try:
        global work_model
        work_model = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE, SamplerTypes.SAMPLER_STD)
        codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                     keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                        work_model.orig_filter.coupling_matrix.links])
        codec = codec
        work_model.setup(
            model_name="simple_opt",
            model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
            dm_codec=codec
        )
        global inference_model
        inference_model = work_model.inference(f"saved_models\\{fil_name}\\best-epoch=25-train_loss=0.02530-val_loss=0.02793-val_r2=0.96251-val_mse=0.00296-val_mae=0.02496-batch_size=32-base_dataset_size=1000000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e


def predict(fil: rf.Network):
    try:
        global inference_model
        pred_prms = inference_model.predict_x(fil)
        print(f"Предсказанные параметры: {pred_prms}")
        pred_fil = inference_model.create_filter_from_prediction(fil, pred_prms, work_model.meta)
        optim_matrix = optimize_cm(pred_fil, pred_fil)
        print(f"Оптимизированные параметры: {optim_matrix.factors}")
        return optim_matrix.matrix.numpy()
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e


def model_info():
    try:
        global work_model
        info = work_model.info()
        return info
    except Exception as e:
        raise ValueError(f"На сервере возникла ошибка: {e}") from e


def calc_s_params(M:np.array, f0:float, bw:float, Q:float, frange:list or np.array):
    fbw = bw/f0
    S = MWFilter.response_from_coupling_matrix(M=M, f0=f0, FBW=fbw, Q=Q, frange=frange)
    S11 = S[:, 0, 0]
    S22 = S[:, 1, 1]
    S12 = S[:, 0, 1]
    S21 = S[:, 1, 0]
    return S11, S12, S21, S22

