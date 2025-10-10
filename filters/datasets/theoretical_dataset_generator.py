import copy
import os
import numpy as np
import torch
from torch import dtype

from mwlab import TouchstoneData, TouchstoneDataset
from ..filter import MWFilter, CouplingMatrix
from ..utils import Sampler, SamplerTypes
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from mwlab.io.backends import HDF5Backend, RAMBackend, StorageBackend, FileBackend
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.transforms import TComposite

import matplotlib.pyplot as plt


class DatasetMWFilter(MWFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_touchstone_data(self, path=None) -> TouchstoneData:
        params = {"f0": self.f0, "bw": self.bw, "Q": self.Q, "N": self.order}
        M = self.coupling_matrix.matrix.cpu().numpy() if torch.is_tensor(
            self.coupling_matrix.matrix) else self.coupling_matrix.matrix

        params.update({f"m_{row}_{col}": M[row, col] for row, col in self.coupling_matrix.links_for_analysis})
        td = TouchstoneData(network=self, params=params, path=path)
        return td



@dataclass
class CMShifts:
    self_coupling: float
    mainline_coupling: float
    cross_coupling: float
    parasitic_coupling: float
    absolute: bool = True


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
    cms: Sampler
    pss: Sampler
    qs: Sampler

    @classmethod
    def create_samplers(cls, origin_filter: MWFilter, samplers_type: SamplerTypes, samplers_size: int,
                         cm_shifts_delta, pss_origin, pss_shifts_delta, samplers_kwargs={}):
        m_min, m_max = CMTheoreticalDatasetGeneratorSamplers.create_min_max_matrices(origin_matrix=origin_filter.coupling_matrix,
                                                    deltas=cm_shifts_delta)
        phase_shifts_min, phase_shifts_max = CMTheoreticalDatasetGeneratorSamplers.create_min_max_phase_shifts(
            origin_shifts=pss_origin, deltas=pss_shifts_delta)

        cms_sampler = Sampler.for_type(
            **samplers_kwargs,
            type=samplers_type,
            start=CouplingMatrix(matrix=m_min).factors,
            stop=CouplingMatrix(matrix=m_max).factors,
            num=samplers_size)
        """ Нас интересует только МС, сдвиги можно оставить как есть """
        pss_sampler_type = copy.deepcopy(samplers_type)
        pss_sampler_type.one_param = False
        pss_sampler = Sampler.for_type(
            **samplers_kwargs,
            type=samplers_type,
            start=phase_shifts_min,
            stop=phase_shifts_max,
            num=len(cms_sampler))

        qs_sampler_type = copy.deepcopy(samplers_type)
        qs_sampler_type.one_param = False
        qs_sampler = Sampler.for_type(
            **samplers_kwargs,
            type=samplers_type,
            start=origin_filter.Q*0.6,
            stop=origin_filter.Q*1.4,
            num=len(cms_sampler)
        )

        # cm_sampler_space = np.zeros(shape=(len(cms_factors), *origin_filter.coupling_matrix.matrix.shape),
        #                             dtype=float)
        # for cm_factors, idx in tuple(zip(cms_factors, range(len(cms_factors)))):
        #     cm_sampler_space[idx] = CouplingMatrix.from_factors(cm_factors, origin_filter.coupling_matrix.links,
        #                                                         origin_filter.order + 2)
        return cls(
            cms=cms_sampler,
            pss=pss_sampler,
            qs=qs_sampler,
        )

    @classmethod
    def concat(cls, samplers: tuple):
        cms = Sampler.concat([cm.cms for cm in samplers])
        pss = Sampler.concat([ps.pss for ps in samplers])
        qs = Sampler.concat([qs.qs for qs in samplers])
        return cls(
            cms=cms,
            pss=pss,
            qs=qs
        )

    @staticmethod
    def create_min_max_matrices(origin_matrix: CouplingMatrix, deltas: CMShifts):
        Mmin = origin_matrix.matrix
        Mmax = origin_matrix.matrix
        N = origin_matrix.matrix_order
        antidiagonal = [(i, N - 1 - i) for i in range(N)]

        if not deltas.absolute:
            for (i, j) in origin_matrix.links:
                if i == j:
                    shift = deltas.self_coupling
                elif j == (i + 1):
                    shift = deltas.mainline_coupling
                elif (i, j) in antidiagonal:
                    shift = deltas.cross_coupling
                else:
                    shift = deltas.parasitic_coupling
                m_min_ = (1-shift)*Mmin[i][j]
                m_max_ = (1+shift)*Mmax[i][j]
                m_min = min(m_min_, m_max_)
                m_max = max(m_min_, m_max_)
                Mmin[i][j] = m_min
                Mmin[j][i] = m_min
                Mmax[i][j] = m_max
                Mmax[j][i] = m_max
        else:
            for (i, j) in origin_matrix.links:
                if i == j:
                    m_max_ = Mmax[i][j] + deltas.self_coupling
                    m_min_ = Mmin[i][j] - deltas.self_coupling
                elif j == (i + 1):
                    m_max_ = Mmax[i][j] + deltas.mainline_coupling
                    m_min_ = Mmin[i][j] - deltas.mainline_coupling
                elif (i, j) in antidiagonal:
                    m_max_ = Mmax[i][j] + deltas.cross_coupling
                    m_min_ = Mmin[i][j] - deltas.cross_coupling
                else:
                    m_max_ = Mmax[i][j] + deltas.parasitic_coupling
                    m_min_ = Mmin[i][j] - deltas.parasitic_coupling
                m_min = min(m_min_, m_max_)
                m_max = max(m_min_, m_max_)
                Mmin[i][j] = m_min
                Mmin[j][i] = m_min
                Mmax[i][j] = m_max
                Mmax[j][i] = m_max
        return Mmin, Mmax

    @staticmethod
    def create_min_max_phase_shifts(origin_shifts: PSShift, deltas: PSShift):
        phase_shifts_min = origin_shifts.array() - deltas.array()
        phase_shifts_max = origin_shifts.array() + deltas.array()
        return phase_shifts_min, phase_shifts_max



AVAILABLE_BACKEND_TYPES = ['s2p', 'hdf5', 'ram']


class CMTheoreticalDatasetGenerator:
    _BASE_FILENAME = "Dataset"
    _FILENAME_SUFFIX = {'s2p': ".s2p", 'hdf5': ".h5", "ram": ".pkl"}
    def __init__(self,
                 path_to_save_dataset: str,  # Путь к директориям с датасетами
                 filename: str,
                 orig_filter: MWFilter,
                 backend_type: str = 'ram',
                 backend_kwargs: dict = {},
                 rewrite: bool = False
                 ):
        self._path_to_save_dataset = path_to_save_dataset
        self._backend_type = backend_type
        self._filename = filename + self._FILENAME_SUFFIX[self._backend_type]
        self._backend = self.create_backend(backend_type, backend_kwargs, rewrite)
        self._origin_filter = orig_filter

    def _full_dataset_path(self):
        return os.path.join(self._path_to_save_dataset, self._filename)

    def _check_dataset(self, rewrite) -> bool:
        if rewrite: return True
        return not os.path.exists(self._full_dataset_path())

    def create_backend(self, backend_type: str, backend_kwargs: dict, rewrite) -> StorageBackend:
        if not backend_type in AVAILABLE_BACKEND_TYPES:
            raise ValueError(f"Unsupported backed: {backend_type}")
        self._enable_generate = self._check_dataset(rewrite)
        if not self._enable_generate:
            print(f"Directory already have dataset files. Load backend from existing")
            if backend_type == 's2p':
                backend = FileBackend(self._path_to_save_dataset, **backend_kwargs)
            elif backend_type == 'hdf5':
                backend = HDF5Backend(self._full_dataset_path(), **backend_kwargs)
            elif backend_type == 'ram':
                backend = RAMBackend.load_pickle(self._full_dataset_path(), **backend_kwargs)
        else:
            print(f"Write data into directory: {self._path_to_save_dataset}")
            if not os.path.exists(self._path_to_save_dataset):
                os.makedirs(self._path_to_save_dataset)
            if backend_type == 's2p':
                backend = FileBackend(self._path_to_save_dataset, **backend_kwargs)
            elif backend_type == 'hdf5':
                backend = HDF5Backend(self._full_dataset_path(), mode='w', **backend_kwargs)
            elif backend_type == 'ram':
                backend = RAMBackend([], **backend_kwargs)
        return backend

    @property
    def backend(self):
        return self._backend

    @property
    def path_to_dataset(self):

        if self._backend_type == 'ram' or self._backend_type == 'hdf5':
            return self._full_dataset_path()
        elif self._backend_type == 's2p':
            return self._path_to_save_dataset

    @property
    def origin_filter(self):
        return self._origin_filter

    def generate(self, samplers: CMTheoreticalDatasetGeneratorSamplers):
        if not self._enable_generate:
            return
        if len(samplers.pss) != len(samplers.cms):
            raise ValueError(f"Размер сэмплера с фазовыми сдвигами (ps_shifts): {len(samplers.pss)}"
                             f" должен равняться размеру сэмплера со сдвигами элементов матрицы связи (cm_shifts): "
                             f"{len(samplers.cms)}")
        size = len(samplers.cms)
        for idx in tqdm(range(size), desc=f"Генерация датасета в путь: {self._path_to_save_dataset}"):
            cm_factors = samplers.cms[idx]
            new_matrix = CouplingMatrix.from_factors(factors=torch.tensor(cm_factors, dtype=torch.float32),
                                                     links=self.origin_filter.coupling_matrix.links,
                                                     matrix_order=self.origin_filter.coupling_matrix.matrix_order)
            if torch.isnan(new_matrix).any() or torch.isinf(new_matrix).any():
                raise ValueError("⚠️ Input to model contains NaN or Inf")
            ps_shifts = samplers.pss[idx]
            # Q = samplers.qs[idx].item()
            Q = self._origin_filter.Q
            s_params = MWFilter.response_from_coupling_matrix(M=new_matrix, f0=self._origin_filter.f0,
                                                              FBW=self._origin_filter.fbw, Q=Q,
                                                              frange=self._origin_filter.f / 1e6, PSs=ps_shifts)

            new_filter = MWFilter(f0=self._origin_filter.f0, order=self._origin_filter.order, bw=self._origin_filter.bw,
                                  Q=Q, matrix=new_matrix, frequency=self._origin_filter.f,
                                  s=s_params, z0=50)
            ts = new_filter.to_touchstone_data()
            params = torch.tensor(list(ts.params.values()), dtype=torch.float32)
            if torch.isnan(params).any() or torch.isinf(params).any():
                raise ValueError("⚠️ Params contains NaN or Inf")
            self._backend.append(ts)
        if self._backend_type == 'ram':
            self._backend.dump_pickle(self._full_dataset_path())