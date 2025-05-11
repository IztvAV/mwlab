import os
import numpy as np
from ..filter import MWFilter
from ..utils import Sampler
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class CMTheoreticalDatasetGeneratorSamplers:
    cm_shifts: Sampler
    ps_shifts: Sampler


GET_FILENAME_FOR_INDEX = lambda path, idx: path+f"\\Data_{idx:04d}.s2p"


class CMTheoreticalDatasetGenerator:
    def __init__(self,
                 path_to_save_dataset: str, # Путь к директориям с датасетами
                 origin_filter: MWFilter,
                 samplers: CMTheoreticalDatasetGeneratorSamplers
                 ):
        self._samplers = samplers
        self._origin_filter = origin_filter
        self._dataset_size = len(samplers.ps_shifts)
        self._path_to_save_dataset = path_to_save_dataset+f"_{self._dataset_size}"
        self._enable_generate = True
        print(f"Write data into directory: {self._path_to_save_dataset}")
        if not os.path.exists(self._path_to_save_dataset):
            os.makedirs(self._path_to_save_dataset)
        if os.path.exists(GET_FILENAME_FOR_INDEX(self._path_to_save_dataset, 0)):
            print(f"Directory already have dataset files!!!")
            self._enable_generate = False

    def generate(self):
        if not self._enable_generate:
            return
        if len(self._samplers.ps_shifts) != len(self._samplers.cm_shifts):
            raise ValueError(f"Размер сэмплера с фазовыми сдвигами (ps_shifts): {len(self._samplers.ps_shifts)}"
                             f" должен равняться размеру сэмплера со сдвигами элементов матрицы связи (cm_shifts): "
                             f"{len(self._samplers.cm_shifts)}")
        size = len(self._samplers.cm_shifts)
        for idx in tqdm(range(size), desc=f"Генерация датасета в путь: {self._path_to_save_dataset}"):
            new_matrix = self._samplers.cm_shifts[idx]
            ps_shifts = self._samplers.ps_shifts[idx]
            s_params = MWFilter.response_from_coupling_matrix(M=new_matrix, f0=self._origin_filter.f0,
                                                       FBW=self._origin_filter.fbw, Q=self._origin_filter.Q,
                                                       frange=self._origin_filter.f/1e6, PSs=ps_shifts)
            new_filter = MWFilter(f0=self._origin_filter.f0, order=self._origin_filter.order, bw=self._origin_filter.bw,
                     Q=self._origin_filter.Q, matrix=new_matrix, frequency=self._origin_filter.f, s=s_params, z0=50)
            # plt.figure()
            # new_filter.plot_s_db()
            new_filter.write_touchstone(GET_FILENAME_FOR_INDEX(self._path_to_save_dataset, idx))
            pass

