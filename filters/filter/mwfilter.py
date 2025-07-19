from pathlib import Path
import mwlab
import skrf as rf
import re
import numpy as np
from mwlab import TouchstoneData
from filters.filter.couplilng_matrix import CouplingMatrix
from copy import deepcopy as copy
import torch


class MWFilter(rf.Network):
    def __init__(self, order: int, f0: float, bw: float, Q: float, matrix: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._order: int = order
        self._f0: float = f0
        self._bw: float = bw
        self._Q: float = Q
        self._coupling_matrix: CouplingMatrix = CouplingMatrix(matrix)

    @staticmethod
    def freq_to_nfreq(freq, f0, bw):
        nfreq = (f0/bw) * (freq/f0 - f0/freq)
        return nfreq

    @staticmethod
    def nfreq_to_freq(nfreq, f0, bw):
        freq = (bw*nfreq + np.sqrt(pow(bw * nfreq, 2) + 4 * pow(f0,2))) / 2
        return freq

    @staticmethod
    def matrix_from_touchstone_data_parameters(params: dict, N=None) -> torch.Tensor:
        if N is None:
            N = int(params.get("N"))
        M = torch.zeros((N + 2, N + 2), dtype=torch.float32)
        for k, v in params.items():
            if k.startswith("m_"):
                _, i, j = k.split("_")
                i, j = int(i), int(j)
                M[i, j] = torch.tensor(v, dtype=torch.float32)
                M[j, i] = torch.tensor(v, dtype=torch.float32)  # symmetric
        return M

    def to_touchstone_data(self, path=None) -> mwlab.TouchstoneData:
        params = {"f0": self.f0, "bw": self.bw, "Q": self.Q, "N": self.order}
        M = self.coupling_matrix.matrix.cpu().numpy() if torch.is_tensor(
            self.coupling_matrix.matrix) else self.coupling_matrix.matrix

        for i in range(M.shape[0]):
            for j in range(i, M.shape[1]):
                val = M[i, j]
                params.update({f"m_{i}_{j}": val})

        td = mwlab.TouchstoneData(network=self, params=params, path=path)
        return td

    @classmethod
    def from_file(cls, filename: str, device='cpu'):
        td = TouchstoneData.load(filename)
        net = td.network
        params = td.params
        matrix = MWFilter.matrix_from_touchstone_data_parameters(params).to(device)
        return cls(
            f0=params.get("f0"),
            bw=params.get("bw"),
            Q=params.get("Q"),
            order=params.get("N"),
            matrix=matrix,
            frequency=net.f,
            s=torch.tensor(net.s, dtype=torch.complex64, device=device),
            z0=net.z0
        )

    @classmethod
    def from_touchstone_dataset_item(cls, item: tuple[dict, rf.Network], device='cpu'):
        params, net = item
        f0 = params.get("f0")
        bw = params.get("bw")
        Q = params.get("Q")
        order = int(params.get("N"))
        matrix = MWFilter.matrix_from_touchstone_data_parameters(params).to(device)

        if f0 is None or bw is None or Q is None or order is None or matrix is None:
            raise ValueError(f"Invalid parameters: f0={f0}, bw={bw}, Q={Q}, order={order}, M={matrix}")

        return cls(
            f0=f0,
            bw=bw,
            Q=Q,
            order=order,
            matrix=matrix,
            frequency=net.f,
            s=torch.tensor(net.s, dtype=torch.complex64, device=device),
            z0=net.z0
        )

    def write_touchstone(self, filename: str | Path = None, *args, **kwargs) -> None:
        param_parts = [
            f"f0={self.f0:.6f}",
            f"bw={self.bw:.6f}",
            f"Q={self.Q:.6f}",
            f"N={int(self.order)}"
        ]

        M = self.coupling_matrix.matrix.cpu().numpy() if torch.is_tensor(
            self.coupling_matrix.matrix) else self.coupling_matrix.matrix
        for i in range(M.shape[0]):
            for j in range(i, M.shape[1]):
                val = M[i, j]
                param_parts.append(f"m_{i}_{j}={val:.6f}")

        param_comment = " Parameters = {" + "; ".join(param_parts) + "}\n"
        self.comments = param_comment if self.comments is None else self.comments + param_comment
        super().write_touchstone(filename)

    @staticmethod
    def to_db(s: torch.Tensor) -> torch.Tensor:
        return 20 * torch.log10(torch.abs(s))

    @property
    def s_db(self) -> torch.Tensor:
        return self.to_db(self.s)

    @property
    def coupling_matrix(self) -> CouplingMatrix:
        return self._coupling_matrix

    @property
    def f0(self) -> float:
        return self._f0

    @property
    def bw(self) -> float:
        return self._bw

    @property
    def order(self) -> int:
        return self._order

    @property
    def Q(self) -> float:
        return self._Q

    @property
    def fbw(self) -> float:
        return self.bw / self.f0

    def copy(self, *, shallow_copy: bool = False):
        ntwk = MWFilter(
            order=self.order,
            f0=self.f0,
            bw=self.bw,
            Q=self.Q,
            matrix=self.coupling_matrix.matrix.clone(),
            z0=self.z0,
            s_def=self.s_def,
            comments=self.comments
        )

        if torch.is_tensor(self.s):
            ntwk._s = (
                torch.empty_like(self.s) if shallow_copy
                else self.s.clone()
            )
        else:
            ntwk._s = (
                np.empty(shape=self.s.shape, dtype=self.s.dtype) if shallow_copy
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

        ntwk._ext_attrs = self._ext_attrs.copy()

        try:
            ntwk.port_names = copy(self.port_names)
        except AttributeError:
            ntwk.port_names = None

        return ntwk

    @property
    def f_norm(self) -> torch.Tensor:
        freqs = torch.tensor(self.f / 1e6, dtype=torch.float32)
        f0 = self.f0
        fbw = self.fbw
        return (1 / fbw) * (freqs / f0 - f0 / freqs)

    def response(self, frange, NRNlist=[], Rs=1, Rl=1, PSs=None, device='cpu'):
        return self.response_from_coupling_matrix(
            M=self.coupling_matrix.matrix,
            f0=self.f0,
            FBW=self.fbw,
            Q=self.Q,
            frange=frange,
            NRNlist=NRNlist,
            Rs=Rs,
            Rl=Rl,
            PSs=PSs,
            device=device
        )

    @staticmethod
    def response_from_coupling_matrix(M, f0, FBW, Q, frange, NRNlist=[], Rs=1, Rl=1, PSs=None, device='cpu'):
        """Vectorized calculation of filter S-parameters from coupling matrix on GPU/CPU."""
        M = M.to(device) if torch.is_tensor(M) else torch.tensor(M, dtype=torch.complex64, device=device)
        frange = torch.tensor(frange, dtype=torch.float32, device=device)
        N = M.shape[0]

        if M.shape[0] != M.shape[1]:
            raise ValueError("Coupling matrix M must be square.")
        if not torch.allclose(M, M.T.conj()):
            raise ValueError("Coupling matrix M must be Hermitian (symmetric along diagonal).")

        # Handle Q input (scalar, list, or array)
        if isinstance(Q, (list, np.ndarray)) and len(Q) == 1:
            Qvec = torch.full((N - 2,), Q[0], dtype=torch.float32, device=device)
        elif isinstance(Q, (list, tuple, np.ndarray, torch.Tensor)):
            Qvec = torch.tensor(Q, dtype=torch.float32, device=device)
            if Qvec.ndim == 0:
                Qvec = Qvec.repeat(N - 2)
            elif Qvec.numel() != N - 2:
                raise ValueError(f"Q size should be {N - 2}, but got {len(Qvec)}.")
        else:
            Qvec = torch.full((N - 2,), Q, dtype=torch.float32, device=device)

        # Identity matrix with NRN adjustments
        I = torch.eye(N, dtype=torch.complex64, device=device)
        I[0, 0] = 0
        I[-1, -1] = 0
        for nrn in NRNlist:
            I[nrn, nrn] = 0

        # Load matrix
        J = torch.zeros((N, N), dtype=torch.complex64, device=device)
        J[0, 0] = 1j * Rs
        J[-1, -1] = 1j * Rl

        # Loss matrix
        G = torch.zeros((N, N), dtype=torch.complex64, device=device)
        for res in range(1, N - 1):
            G[res, res] = 1 / (FBW * Qvec[res - 1])
        Mpr = M - 1j * G

        # Frequency to lambda conversion
        if f0 == 0:
            lam = frange / FBW
        else:
            lam = (frange / f0 - f0 / frange) / FBW

        # Batch matrix creation and inversion
        lam_exp = lam.view(-1, 1, 1)  # (B, 1, 1)
        A = lam_exp * I - J + Mpr  # (B, N, N)
        Ainv = torch.linalg.inv(A)  # (B, N, N)

        # S-parameters calculation
        A00 = Ainv[:, 0, 0]
        ANN = Ainv[:, -1, -1]
        AN0 = Ainv[:, -1, 0]

        S11 = 1 + 2j * Rs * A00
        S22 = 1 + 2j * Rl * ANN
        S21 = -2j * torch.sqrt(torch.tensor(Rs * Rl, dtype=torch.float32, device=device)) * AN0

        if PSs is not None:
            phi11, phi21, theta11, theta21 = PSs
            phi11, phi21, theta11, theta21 = map(
                lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                (phi11, phi21, theta11, theta21)
            )
            phase11 = torch.exp(-1j * 2 * (phi11 + lam * theta11))
            phase21 = torch.exp(-1j * 2 * (phi21 + lam * theta21))
            S11 *= phase11
            S22 *= phase11
            S21 *= phase21

        # Assemble output S (B, 2, 2)
        B = len(frange)
        S = torch.zeros((B, 2, 2), dtype=torch.complex64, device=device)
        S[:, 0, 0] = S11
        S[:, 1, 1] = S22
        S[:, 0, 1] = S21
        S[:, 1, 0] = S21

        return S.cpu().numpy()