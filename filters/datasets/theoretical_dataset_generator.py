import os
import numpy as np
from mwlab import TouchstoneData, TouchstoneDataset
from ..filter import MWFilter, CouplingMatrix
from ..utils import Sampler
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from mwlab.io.backends import HDF5Backend
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.transforms import TComposite



@dataclass
class CMShifts:
    self_coupling: float
    mainline_coupling: float
    cross_coupling: float


@dataclass
class PSShift:
    phi11: float
    phi21: float
    theta11: float
    theta21: float

    def array(self):
        return np.array([self.phi11, self.phi21, self.theta11, self.theta21])


@dataclass
class CMTheoreticalDatasetGeneratorSamplers:
    cm_shifts: Sampler
    ps_shifts: Sampler


GET_S2P_FILENAME_FOR_INDEX = lambda path, idx: path + f"\\Data_{idx:04d}.s2p"
GET_HDF5_FILENAME = lambda path: path + f"\\Dataset.h5"


class CMTheoreticalDatasetGenerator:
    def __init__(self,
                 path_to_origin_filter: str,  # Путь к директории где располагается s2p-файл с оригинальным фильтром
                 path_to_save_dataset: str,  # Путь к директориям с датасетами
                 cm_shifts_delta: CMShifts,
                 pss_origin: PSShift,
                 pss_shifts_delta: PSShift,
                 samplers_size: int,
                 resample_scale: int = 301,  # Значение для ресемплинга
                 f_start: float|None = None,  # Полоса откуда обрезаем
                 f_stop: float|None = None,  # Полоса до которой обрезаем
                 f_unit: str|None = None
                 ):
        tds = TouchstoneDataset(source=path_to_origin_filter)
        origin_filter_ = MWFilter.from_touchstone_dataset_item(tds[0])
        f0 = origin_filter_.f0
        bw = origin_filter_.bw
        if f_start is None:
            f_start = f0-1.2*bw
        if f_stop is None:
            f_stop = f0+1.2*bw
        if f_unit is None:
            f_unit = "MHz"
        self._y_transform = TComposite([
            S_Crop(f_start=f_start, f_stop=f_stop, unit=f_unit),
            S_Resample(resample_scale)
        ])
        tds_transformed = TouchstoneDataset(source=path_to_origin_filter, s_tf=self._y_transform)
        self._origin_filter = MWFilter.from_touchstone_dataset_item(tds_transformed[0])
        m_min, m_max = self.create_min_max_matrices(origin_matrix=self.origin_filter.coupling_matrix,
                                               deltas=cm_shifts_delta)
        phase_shifts_min, phase_shifts_max = self.create_min_max_phase_shifts(
            origin_shifts=pss_origin, deltas=pss_shifts_delta)
        self._samplers = CMTheoreticalDatasetGeneratorSamplers(
            cm_shifts=Sampler.lhs(start=m_min, stop=m_max, num=samplers_size),
            ps_shifts=Sampler.lhs(start=phase_shifts_min, stop=phase_shifts_max, num=samplers_size)
        )

        self._dataset_size = samplers_size
        self._path_to_save_dataset = path_to_save_dataset+f"_{self._dataset_size}"
        self._enable_generate = True
        print(f"Write data into directory: {self._path_to_save_dataset}")
        if not os.path.exists(self._path_to_save_dataset):
            os.makedirs(self._path_to_save_dataset)
        if os.path.exists(GET_HDF5_FILENAME(self._path_to_save_dataset)):
            print(f"Directory already have dataset files!!!")
            self._enable_generate = False

    @staticmethod
    def create_min_max_matrices(origin_matrix: CouplingMatrix, deltas: CMShifts):
        Mmin = origin_matrix.matrix
        Mmax = origin_matrix.matrix

        for (i, j) in origin_matrix.links:
            if i == j:
                Mmin[i][j] -= deltas.self_coupling
                Mmax[i][j] += deltas.self_coupling
            elif j == (i + 1):
                Mmin[i][j] -= deltas.mainline_coupling
                Mmin[j][i] -= deltas.mainline_coupling
                Mmax[i][j] += deltas.mainline_coupling
                Mmax[j][i] += deltas.mainline_coupling
            else:
                Mmin[i][j] -= deltas.cross_coupling
                Mmin[j][i] -= deltas.cross_coupling
                Mmax[i][j] += deltas.cross_coupling
                Mmax[j][i] += deltas.cross_coupling
        return Mmin, Mmax

    @staticmethod
    def create_min_max_phase_shifts(origin_shifts: PSShift, deltas: PSShift):
        phase_shifts_min = origin_shifts.array() - deltas.array()
        phase_shifts_max = origin_shifts.array() + deltas.array()
        return phase_shifts_min, phase_shifts_max

    @property
    def path_to_dataset(self):
        return GET_HDF5_FILENAME(self._path_to_save_dataset)

    @property
    def origin_filter(self):
        return self._origin_filter

    @property
    def y_transform(self):
        return self._y_transform

    def generate(self):
        if not self._enable_generate:
            return
        if len(self._samplers.ps_shifts) != len(self._samplers.cm_shifts):
            raise ValueError(f"Размер сэмплера с фазовыми сдвигами (ps_shifts): {len(self._samplers.ps_shifts)}"
                             f" должен равняться размеру сэмплера со сдвигами элементов матрицы связи (cm_shifts): "
                             f"{len(self._samplers.cm_shifts)}")
        size = len(self._samplers.cm_shifts)
        with HDF5Backend(GET_HDF5_FILENAME(self._path_to_save_dataset), mode="w") as h5b:
            for idx in tqdm(range(size), desc=f"Генерация датасета в путь: {self._path_to_save_dataset}"):
                new_matrix = self._samplers.cm_shifts[idx]
                ps_shifts = self._samplers.ps_shifts[idx]
                s_params = MWFilter.response_from_coupling_matrix(M=new_matrix, f0=self._origin_filter.f0,
                                                           FBW=self._origin_filter.fbw, Q=self._origin_filter.Q,
                                                           frange=self._origin_filter.f/1e6, PSs=ps_shifts)
                new_filter = MWFilter(f0=self._origin_filter.f0, order=self._origin_filter.order, bw=self._origin_filter.bw,
                         Q=self._origin_filter.Q, matrix=new_matrix, frequency=self._origin_filter.f, s=s_params, z0=50)
                ts = new_filter.to_touchstone_data()
                new_filter.write_touchstone(GET_S2P_FILENAME_FOR_INDEX(self._path_to_save_dataset, idx))
                h5b.append(ts)
                pass

