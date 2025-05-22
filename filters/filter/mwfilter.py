from pathlib import Path

import mwlab
import skrf as rf
import re
import numpy as np
from mwlab import TouchstoneData
from .couplilng_matrix import CouplingMatrix
from copy import deepcopy as copy
import torch


class MWFilter(rf.Network):
    def __init__(self, order:int , f0: float, bw: float, Q: float, matrix: np.array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Новые поля
        self._order: int = order
        self._f0: float = f0
        self._bw: float = bw
        self._Q: float = Q
        self._coupling_matrix: CouplingMatrix = CouplingMatrix(matrix)

    @staticmethod
    def matrix_from_touchstone_data_parameters(params: dict, N=None) -> np.array:
        if N is None:
            N = int(params.get("N"))
        M = np.zeros((N + 2, N + 2), dtype=float)
        for k, v in params.items():
            if k.startswith("m_"):
                _, i, j = k.split("_")
                M[int(i), int(j)] = v
                M[int(j), int(i)] = v  # предполагается симметричность
        return M

    def to_touchstone_data(self, path=None) -> mwlab.TouchstoneData:
        params = {"f0": self.f0, "bw": self.bw, "Q": self.Q, "N": self.order}
        # Добавляем элементы матрицы связи
        M = self.coupling_matrix.matrix
        for i in range(M.shape[0]):
            for j in range(i, M.shape[1]):  # верхняя треугольная часть (или вся)
                val = M[i, j]
                if abs(val) > 1e-12:  # пропускаем нули
                    params.update({f"m_{i}_{j}": val})
        td = mwlab.TouchstoneData(network=self, params=params, path=path)
        return td

    @classmethod
    def from_file(cls, filename: str):
        """
                Считывает матричные элементы, сохранённые в одной строке комментария.
                Возвращает: словарь {(i, j): значение}
                """
        td = TouchstoneData.load(filename)
        net = td.network
        params = td.params
        matrix = MWFilter.matrix_from_touchstone_data_parameters(params)
        return cls(
            f0=params.get("f0"),
            bw=params.get("bw"),
            Q=params.get("Q"),
            order=params.get("N"),
            matrix=matrix,
            frequency=net.f,
            s=net.s,
            z0=net.z0
        )

    @classmethod
    def from_touchstone_dataset_item(cls, item: tuple[dict, rf.Network]):
        params, net = item
        f0 = params.get("f0")
        bw = params.get("bw")
        Q = params.get("Q")
        order = int(params.get("N"))
        matrix = MWFilter.matrix_from_touchstone_data_parameters(params)
        if f0 is None or bw is None or Q is None or order is None or matrix is None:
            raise ValueError(f"Некорректные параметры для инициализации класса: f0={f0}, bw={bw}, Q={Q}, "
                             f"order={order}, \nM={matrix}")
        return cls(
            f0=f0,
            bw=bw,
            Q=Q,
            order=order,
            matrix=matrix,
            frequency=net.f,
            s=net.s,
            z0=net.z0
        )

    def write_touchstone(self, filename: str | Path = None, *args, **kwargs) -> None:
        # Собираем параметры
        param_parts = [
            f"f0={self.f0:.6f}",
            f"bw={self.bw:.6f}",
            f"Q={self.Q:.6f}",
            f"N={int(self.order)}"
        ]

        # Добавляем элементы матрицы связи
        M = self.coupling_matrix.matrix
        for i in range(M.shape[0]):
            for j in range(i, M.shape[1]):  # верхняя треугольная часть (или вся)
                val = M[i, j]
                if abs(val) > 1e-12:  # пропускаем нули
                    param_parts.append(f"m_{i}_{j}={val:.6f}")

        # Формируем одну строку комментария
        param_comment = " Parameters = {" + "; ".join(param_parts) + "}\n"

        if self.comments is not None:
            self.comments += param_comment
        else:
            self.comments = param_comment
        # Сохраняем обычный touchstone файл (без комментариев)
        super().write_touchstone(filename)

    @staticmethod
    def to_db(s):
        return 20 * torch.log10(abs(torch.tensor(s)))

    @property
    def s_db(self):
        return self.to_db(self.s)

    @property
    def coupling_matrix(self):
        return self._coupling_matrix

    @property
    def f0(self):
        return self._f0

    @property
    def bw(self):
        return self._bw

    @property
    def order(self):
        return self._order

    @property
    def Q(self):
        return self._Q

    @property
    def fbw(self):
        return self.bw/self.f0


    ## CLASS METHODS
    def copy(self, *, shallow_copy: bool = False):
        """
        Return a copy of this Network.

        Needed to allow pass-by-value for a Network instead of
        pass-by-reference

        Parameters
        ----------
        shallow_copy : bool, optional
            If True, the method creates a new Network object with empty s-parameters that share the same shape
            as the original Network, but without copying the actual s-parameters data. This is useful when you
            plan to immediately modify the s-parameters after creating the Network, as a deep copy would be
            unnecessary and costly. Using `shallow_copy` improves performance by leveraging lazy initialization
            through `numpy's np.empty()`, which allocates virtual memory without immediate physical memory
            allocation, deferring actual memory initialization until first access. This approach can significantly
            enhance `copy()` performance when dealing with large `Network` objects, when you are intended for
            immediate modification after the Network's creation.

        Note
        ----
        If you require a complete copy of the `Network` instance or need to perform operation on the s-parameters
        of the copied Network, it is essential not to use the `shallow_copy` parameter!

        Returns
        -------
        ntwk : :class:`Network`
            Copy of the Network

        """
        ntwk = MWFilter(order=self.order, f0=self.f0, bw=self.bw, Q=self.Q, matrix=self.coupling_matrix.matrix,
                        z0=self.z0, s_def=self.s_def, comments=self.comments)

        ntwk._s = (
            np.empty(shape=self.s.shape, dtype=self.s.dtype)
            if shallow_copy
            else self.s.copy()
        )
        ntwk.frequency._f = self.frequency._f.copy()
        ntwk.frequency.unit = self.frequency.unit
        ntwk.port_modes = self.port_modes.copy()

        if self.params is not None:
            ntwk.params = self.params.copy()

        ntwk.name = self.name

        if self.noise is not None and self.noise_freq is not None:
          ntwk.noise = self.noise.copy()
          ntwk.noise_freq = self.noise_freq.copy()

        # copy special attributes (such as _is_circuit_port) but skip methods
        ntwk._ext_attrs = self._ext_attrs.copy()

        try:
            ntwk.port_names = copy(self.port_names)
        except(AttributeError):
            ntwk.port_names = None
        return ntwk

    @property
    def f_norm(self):
        freqs = self.f/1e6
        f0 = self.f0
        fbw = self.fbw
        nfreq = (1 / fbw) * (freqs / f0 - f0 / freqs)
        return nfreq


    def response(self, frange, NRNlist=[], Rs=1, Rl=1, PSs=None):
        resp = self.response_from_coupling_matrix(M=self.coupling_matrix.matrix, f0=self.f0, FBW=self.fbw, Q=self.Q,
                                                  frange=frange, NRNlist=NRNlist, Rs=Rs, Rl=Rl, PSs=PSs)
        return resp

    @staticmethod
    @torch.no_grad()
    def response_from_coupling_matrix(M, f0, FBW, Q, frange, NRNlist=[], Rs=1, Rl=1, PSs=None):
        """
        Векторизованный расчёт S-параметров фильтра из матрицы связи на GPU/CPU.
        """

        def RespM2_gpu(M, f0, FBW, Q, frange, NRNlist=[], Rs=1, Rl=1, PSs=None, device='cuda'):
            # Переводим входы в тензоры
            M = torch.tensor(M, dtype=torch.complex64, device=device)
            frange = torch.tensor(frange, dtype=torch.float32, device=device)
            N = M.shape[0]

            # Проверки
            if M.shape[0] != M.shape[1]:
                raise ValueError("Матрица связи M должна быть квадратной.")
            if not torch.allclose(M, M.T.conj()):
                raise ValueError("Матрица связи M должна быть эрмитовой (симметричной по диагонали).")

            if isinstance(Q, list) and len(Q) == 1:
                Qvec = torch.full((N - 2,), Q[0], dtype=torch.float32, device=device)
            elif isinstance(Q, np.ndarray) and len(Q) == 1:
                Qvec = torch.full((N - 2,), Q[0], dtype=torch.float32, device=device)
            else:
                if isinstance(Q, (list, tuple, torch.Tensor)):
                    Qvec = torch.tensor(Q, dtype=torch.float32, device=device)
                    if Qvec.ndim == 0:
                        Qvec = Qvec.repeat(N - 2)
                    elif Qvec.numel() != N - 2:
                        raise ValueError(f"Размер Q должен быть {N - 2}, но получен {len(Qvec)}.")
                else:
                    Qvec = torch.full((N - 2,), Q, dtype=torch.float32, device=device)

            # Единичная матрица с учетом NRN
            I = torch.eye(N, dtype=torch.complex64, device=device)
            I[0, 0] = 0
            I[-1, -1] = 0
            for nrn in NRNlist:
                I[nrn, nrn] = 0

            # Матрица нагрузки
            J = torch.zeros((N, N), dtype=torch.complex64, device=device)
            J[0, 0] = 1j * Rs
            J[-1, -1] = 1j * Rl

            # Матрица потерь
            G = torch.zeros((N, N), dtype=torch.complex64, device=device)
            for res in range(1, N - 1):
                G[res, res] = 1 / (FBW * Qvec[res - 1])
            Mpr = M - 1j * G

            # Вектор частот → ламбда
            if f0 == 0:
                lam = frange / FBW
            else:
                lam = (frange / f0 - f0 / frange) / FBW

            # Батчевое создание матриц A
            lam_exp = lam.view(-1, 1, 1)  # (B, 1, 1)
            A = lam_exp * I - J + Mpr  # (B, N, N)

            # Обратные матрицы
            Ainv = torch.linalg.inv(A)  # (B, N, N)

            # Расчет S-параметров
            A00 = Ainv[:, 0, 0]
            ANN = Ainv[:, -1, -1]
            AN0 = Ainv[:, -1, 0]

            S11 = 1 + 2j * Rs * A00
            S22 = 1 + 2j * Rl * ANN
            S21 = -2j * torch.sqrt(torch.tensor(Rs * Rl, dtype=torch.float32, device=device)) * AN0

            if PSs is not None:
                phi11, phi21, theta11, theta21 = PSs
                phi11, phi21, theta11, theta21 = map(lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                                                     (phi11, phi21, theta11, theta21))
                phase11 = torch.exp(-1j * 2 * (phi11 + lam * theta11))
                phase21 = torch.exp(-1j * 2 * (phi21 + lam * theta21))
                S11 *= phase11
                S22 *= phase11
                S21 *= phase21

            # Собираем выходной массив S (B, 2, 2)
            B = len(frange)
            S = torch.zeros((B, 2, 2), dtype=torch.complex64, device=device)
            S[:, 0, 0] = S11
            S[:, 1, 1] = S22
            S[:, 0, 1] = S21
            S[:, 1, 0] = S21

            return S.cpu().numpy()

        return RespM2_gpu(M, f0, FBW, Q, frange, NRNlist, Rs, Rl, PSs, device='cpu')
