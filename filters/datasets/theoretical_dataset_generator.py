import os
import numpy as np
from filters import MWFilter
from ..utils import Sampler
from dataclasses import dataclass


@dataclass
class CMTheoreticalDatasetGeneratorSamplers:
    cm_shifts: Sampler
    ps_shifts: Sampler


class CMTheoreticalDatasetGenerator:
    def __init__(self,
                 path_to_save_dataset: str, # Путь к директориям с датасетами
                 origin_filter: MWFilter,
                 samplers: CMTheoreticalDatasetGeneratorSamplers
                 ):
        self._samplers = samplers
        self._origin_filter = origin_filter
        self._path_to_save_dataset = path_to_save_dataset
        print(f"Write data into directory: {path_to_save_dataset}")
        if not os.path.exists(path_to_save_dataset):
            os.makedirs(path_to_save_dataset)

    """TODO: Пока поддержана только свернутая топология. Сдвиги формируются как в старом методе"""
    def _get_cm_shift(self):
        m = self._origin_filter.coupling_matrix.matrix
        for idx in range(len(self._samplers.cm_shifts)):
            delta = self._samplers.cm_shifts[idx]
            for (k, (i, j)) in enumerate(self._origin_filter.coupling_matrix.links):
                if i == j:
                    m[i][j] += delta[0]
                elif j == (i + 1):
                    m[i][j] += delta[1]
                else:
                    m[i][j] += delta[2]
            yield m

    def _get_ps_shift(self):
        for i in range(len(self._samplers.ps_shifts)):
            yield self._samplers.ps_shifts[i]

    def generate(self):
        if len(self._samplers.ps_shifts) != len(self._samplers.cm_shifts):
            raise ValueError(f"Размер сэмплера с фазовыми сдвигами (ps_shifts): {len(self._samplers.ps_shifts)}"
                             f" должен равняться размеру сэмплера со сдвигами элементов матрицы связи (cm_shifts): "
                             f"{len(self._samplers.cm_shifts)}")
        size = len(self._samplers.cm_shifts)
        for _ in range(size):
            new_matrix = self._get_cm_shift()
            ps_shifts = self._get_ps_shift()
            # TODO: Проверить правильность возвращаемого значения
            s_params = MWFilter.response_from_coupling_matrix(M=new_matrix, f0=self._origin_filter.f0,
                                                       FBW=self._origin_filter.fbw, Q=self._origin_filter.Q,
                                                       frange=self._origin_filter.f, Pss=ps_shifts)
            new_filter = MWFilter(f0=self._origin_filter.f0, order=self._origin_filter.order, bw=self._origin_filter.bw,
                     Q=self._origin_filter.Q, matrix=new_matrix, frequency=frequencies, s=s_params, z0=50)
            # TODO: сделать сохранение в датасет S-параметров и матрицы связи

