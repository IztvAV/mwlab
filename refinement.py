import os
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from common import WorkModel
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
from mwlab.nn.scalers import MinMaxScaler, StdScaler
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

import sklearn
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import joblib
from tqdm import tqdm
import sensivity
from torchvision import models as t_models

from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class UnrolledOptimizer(nn.Module):
    def __init__(self, s_dim, m_dim, hidden_dim=128, num_steps=5, shared_weights=True):
        super().__init__()
        self.num_steps = num_steps
        self.shared_weights = shared_weights

        def make_block():
            return nn.Sequential(
                nn.Linear(s_dim + m_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, m_dim)
            )

        if shared_weights:
            self.block = make_block()
        else:
            self.blocks = nn.ModuleList([make_block() for _ in range(num_steps)])

    def forward(self, s_params, m_init):
        m = m_init
        for i in range(self.num_steps):
            inp = torch.cat([s_params, m], dim=1)
            if self.shared_weights:
                delta = self.block(inp)
            else:
                delta = self.blocks[i](inp)
            m = m + delta
        return m


class CorrectionNet(nn.Module):
    def __init__(self, s_shape, m_dim, hidden_dim=256):
        super().__init__()
        s_dim = s_shape[0] * s_shape[1]  # 301 * 8 = 2408
        self.fc1 = nn.Linear(s_dim + m_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, m_dim)

    def forward(self, s_params, matrix_pred):

        s_params = s_params.view(s_params.size(0), -1)  # B x 2408
        x = torch.cat([s_params, matrix_pred], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta = self.fc3(x)
        return matrix_pred + delta


class CorrectionDataset(Dataset):
    def __init__(self, npz_file, normalize=True, norm_type="minmax"):
        data = np.load(npz_file)
        self.s_params = data["s_params"].astype(np.float32)
        self.matrix_pred = data["matrix_pred"].astype(np.float32)
        self.matrix_target = data["matrix_target"].astype(np.float32)

        self.normalize = normalize
        self.norm_type = norm_type

        if normalize:
            self._compute_stats()
            self._apply_normalization()

    def _compute_stats(self):
        if self.norm_type == "zscore":
            self.s_mean = self.s_params.mean(axis=0)
            self.s_std = self.s_params.std(axis=0) + 1e-8
            self.m_mean = self.matrix_pred.mean(axis=0)
            self.m_std = self.matrix_pred.std(axis=0) + 1e-8
        elif self.norm_type == "minmax":
            self.s_min = self.s_params.min(axis=0)
            self.s_max = self.s_params.max(axis=0)
            self.m_min = self.matrix_pred.min(axis=0)
            self.m_max = self.matrix_pred.max(axis=0)
        else:
            raise ValueError("Unknown normalization type")

    def _apply_normalization(self):
        if self.norm_type == "zscore":
            self.s_params = (self.s_params - self.s_mean) / self.s_std
            self.matrix_pred = (self.matrix_pred - self.m_mean) / self.m_std
            self.matrix_target = (self.matrix_target - self.m_mean) / self.m_std
        elif self.norm_type == "minmax":
            self.s_params = (self.s_params - self.s_min) / (self.s_max - self.s_min + 1e-8)
            self.matrix_pred = (self.matrix_pred - self.m_min) / (self.m_max - self.m_min + 1e-8)
            self.matrix_target = (self.matrix_target - self.m_min) / (self.m_max - self.m_min + 1e-8)

    def __len__(self):
        return len(self.s_params)

    def __getitem__(self, idx):
        return {
            "s_params": torch.from_numpy(self.s_params[idx]),
            "matrix_pred": torch.from_numpy(self.matrix_pred[idx]),
            "matrix_target": torch.from_numpy(self.matrix_target[idx]),
        }

    def get_norm_params(self):
        """Позволяет сохранить параметры нормализации"""
        if not self.normalize:
            return None

        if self.norm_type == "zscore":
            return {
                "s_mean": self.s_mean, "s_std": self.s_std,
                "m_mean": self.m_mean, "m_std": self.m_std,
                "type": "zscore"
            }
        elif self.norm_type == "minmax":
            return {
                "s_min": self.s_min, "s_max": self.s_max,
                "m_min": self.m_min, "m_max": self.m_max,
                "type": "minmax"
            }


    @classmethod
    def generate(cls, inference_model, work_model: common.WorkModel, npz_file, normalize=True, norm_type="minmax"):
        ds = work_model.ds
        backend = ds.backend
        s_params = []
        matrix_pred = []
        matrix_target = []
        for i in tqdm(range(len(backend)), desc="NN prediction"):
            tsd = backend.read(i)
            s_params.append(work_model.codec.encode_s(tsd.network)[0])
            matrix_target.append([backend.read(0).params[x] for x in work_model.codec.x_keys])
            matrix_pred_ = inference_model.predict_x(tsd.network)
            matrix_pred.append([matrix_pred_[x] for x in work_model.codec.x_keys])
        os.makedirs(os.path.dirname(npz_file), exist_ok=True)
        np.savez(npz_file,
                 s_params=s_params,
                 matrix_pred=matrix_pred,
                 matrix_target=matrix_target
                 )
        return cls(npz_file, normalize, norm_type)

    def normalize_input(self, s_params, matrix_pred):
        if self.norm_type == "zscore":
            raise ValueError("Undefined")
            # s = (s_params - norm_params["s_mean"]) / norm_params["s_std"]
            # m = (matrix_pred - norm_params["m_mean"]) / norm_params["m_std"]
        elif self.norm_type == "minmax":
            s = (s_params - self.s_min) / (self.s_max - self.s_min + 1e-8)
            m = (matrix_pred - self.m_min) / (self.m_max - self.m_min + 1e-8)
        else:
            raise ValueError("Unknown normalization type")
        return s.astype(np.float32), m.astype(np.float32)

    def denormalize_matrix(self, matrix_norm):
        if self.norm_type == "zscore":
            raise ValueError("Undefined")
            # return matrix_norm * norm_params["m_std"] + norm_params["m_mean"]
        elif self.norm_type == "minmax":
            return matrix_norm * self.m_max - self.m_min + self.m_min
        else:
            raise ValueError("Unknown normalization type")

    # def save(self, path_prefix):
    #     """Сохраняет нормализованные данные и параметры нормализации"""
    #     os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    #
    #     # Сохраняем данные
    #     np.savez(f"{path_prefix}.npz",
    #              s_params=self.s_params,
    #              matrix_pred=self.matrix_pred,
    #              matrix_target=self.matrix_target)

        # # Сохраняем параметры нормализации (если есть)
        # norm_params = self.get_norm_params()
        # if norm_params is not None:
        #     # Преобразуем numpy в обычные списки для сериализации в JSON
        #     norm_params_serializable = {
        #         k: v.tolist() if isinstance(v, np.ndarray) else v
        #         for k, v in norm_params.items()
        #     }
        #     with open(f"{path_prefix}_norm.json", "w") as f:
        #         json.dump(norm_params_serializable, f, indent=2)





def refinement():
    work_model = common.WorkModel(configs.ENV_TUNE_DATASET_PATH,100000, SamplerTypes.SAMPLER_SOBOL)
    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                    work_model.orig_filter.coupling_matrix.links])
    work_model.setup(
        model_name="resnet_with_correction",
        model_cfg={"in_channels": len(codec.y_channels), "out_channels": len(codec.x_keys)},
        dm_codec=codec
    )
    inference_model = work_model.inference(
        "saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=29-train_loss=0.04166-val_loss=0.04450-val_r2=0.92560-val_mse=0.00588-val_mae=0.03862-batch_size=32-base_dataset_size=1500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
    # dataset = CorrectionDataset.generate(inference_model, work_model, os.path.join(configs.ENV_TUNE_DATASET_PATH, "refinement.npz"))
    dataset = CorrectionDataset(os.path.join(configs.ENV_TUNE_DATASET_PATH, "refinement.npz"))

    # Предположим, dataset уже создан
    val_fraction = 0.2
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # Инициализация
    s_dim = dataset[0]["s_params"].shape
    m_dim = dataset[0]["matrix_pred"].shape[0]
    model = CorrectionNet(s_dim, m_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    # Обучающий цикл
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in train_loader:
            s = batch["s_params"]
            mp = batch["matrix_pred"]
            mt = batch["matrix_target"]

            optimizer.zero_grad()
            mc = model(s, mp)
            loss = loss_fn(mc, mt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s.size(0)
        train_loss = total_loss / len(train_set)

            # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                s = batch["s_params"]
                mp = batch["matrix_pred"]
                mt = batch["matrix_target"]

                mc = model(s, mp)
                loss = loss_fn(mc, mt)
                val_loss += loss.item() * s.size(0)

        val_loss /= len(val_set)
        print(f"Epoch {epoch}: train_loss: {train_loss:.6f} val_loss = {val_loss:.6f}")

    s_params = []
    matrix_target = []
    matrix_pred = []
    backend = TouchstoneDataset(configs.ENV_ORIGIN_DATA_PATH).backend
    tsd = backend.read(0)
    s_params.append(work_model.codec.encode_s(tsd.network)[0])
    matrix_target.append([backend.read(0).params[x] for x in work_model.codec.x_keys])
    matrix_pred_ = inference_model.predict_x(tsd.network)
    matrix_pred.append([matrix_pred_[x] for x in work_model.codec.x_keys])
    s_norm, m_pred_norm = dataset.normalize_input(s_params, matrix_pred)
    s_tensor = torch.from_numpy(s_norm)  # (1, 301, 8)
    m_tensor = torch.from_numpy(m_pred_norm)  # (1, 37)
    # Прогон через модель
    model.eval()
    with torch.no_grad():
        m_corrected_norm = model(s_tensor, m_tensor)
    m_corrected = dataset.denormalize_matrix(m_corrected_norm.squeeze(0).numpy())
    print(f"Matrix target: {matrix_target}")
    print(f"Matrix corrected: {m_corrected}")

    # npz_file = os.path.join(configs.ENV_TUNE_DATASET_PATH, "orig_fil.npz")
    # os.makedirs(os.path.dirname(npz_file), exist_ok=True)
    # np.savez(npz_file,
    #          s_params=s_norm,
    #          matrix_pred=m_pred_norm,
    #          matrix_target=matrix_target
    #          )
    # dataset = CorrectionDataset(os.path.join(configs.ENV_TUNE_DATASET_PATH, "orig_fil.npz"))
    # loader = DataLoader(dataset, batch_size=64)
    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for batch in loader:
    #         s = batch["s_params"]
    #         mp = batch["matrix_pred"]
    #         mt = batch["matrix_target"]
    #
    #         mc = model(s, mp)
    #         loss = loss_fn(mc, mt)
    #         val_loss += loss.item() * s.size(0)
    #
    # val_loss /= len(val_set)
    # print(f"Test for real filter: val_loss = {val_loss:.6f}")





if __name__ == "__main__":
    refinement()