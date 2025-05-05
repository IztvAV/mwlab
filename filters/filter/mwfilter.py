import skrf as rf
import re
import numpy as np
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
        # if not from_matrix:
        #     self._parse_comments()

    @classmethod
    def from_file(cls, filename: str):
        """
                Считывает матричные элементы, сохранённые в одной строке комментария.
                Возвращает: словарь {(i, j): значение}
                """
        net = rf.Network(filename)
        matrix_elements = {}
        lines = net.comments
        order, f0, bw, Q, matrix = None, None, None, None, None

        for line in lines.split("\n"):
            if line.startswith(" N") or line.startswith(" f0") or line.startswith(" bw") or line.startswith(" Q"):
                pattern = re.compile(r" (\w+)\s*:\s*([\d.]+)\s*(MHz)?", re.IGNORECASE)
                match = pattern.match(line)
                key = match.group(1)
                if key == "f0":
                    f0 = float(match.group(2))
                elif key == "bw":
                    bw = float(match.group(2))
                elif key == "N":
                    order = int(match.group(2))
                elif key == "Q":
                    Q = float(match.group(2))
            elif line.startswith(" matrix"):
                # Найти все пары вида m_1_2 = 0.123
                matches = re.findall(r"m_(\d+)_(\d+)\s*=\s*([-+eE0-9.]+)", line)
                for i_str, j_str, val_str in matches:
                    i, j, val = int(i_str), int(j_str), float(val_str)
                    matrix_elements[(i, j)] = val
                matrix = np.zeros((order + 2, order + 2))
                rows, cols = zip(*matrix_elements.keys())
                matrix[rows, cols] = list(matrix_elements.values())
                coupling_matrix = np.rot90(matrix, 2) + matrix - np.diag(np.diag(matrix))
        if f0 is None or bw is None or order is None or Q is None or coupling_matrix is None:
            raise ImportError("Считан неправильный файл. В комментариях должна присутствовать информация о "
                              "центральной частоте фильтра (f0), ширине полосы пропускания (bw), порядке фильтра (N), "
                              "добротности резонаторов (Q) и матрице связи (m_i_j)")
        return cls(order, f0, bw, Q, coupling_matrix, filename)

    #
    # def _parse_comments(self):
    #     """
    #     Считывает матричные элементы, сохранённые в одной строке комментария.
    #     Возвращает: словарь {(i, j): значение}
    #     """
    #     matrix_elements = {}
    #     lines = self.comments
    #
    #     for line in lines.split("\n"):
    #         if line.startswith(" N") or line.startswith(" f0") or line.startswith(" bw") or line.startswith(" Q"):
    #             pattern = re.compile(r" (\w+)\s*:\s*([\d.]+)\s*(MHz)?", re.IGNORECASE)
    #             match = pattern.match(line)
    #             key = match.group(1)
    #             if key == "f0":
    #                 self._f0 = float(match.group(2))
    #             elif key == "bw":
    #                 self._bw = float(match.group(2))
    #             elif key == "N":
    #                 self._order = int(match.group(2))
    #             elif key == "Q":
    #                 self._Q = float(match.group(2))
    #         elif line.startswith(" matrix"):
    #             # Найти все пары вида m_1_2 = 0.123
    #             matches = re.findall(r"m_(\d+)_(\d+)\s*=\s*([-+eE0-9.]+)", line)
    #             for i_str, j_str, val_str in matches:
    #                 i, j, val = int(i_str), int(j_str), float(val_str)
    #                 matrix_elements[(i, j)] = val
    #             matrix = np.zeros((self._order+2, self._order+2))
    #             rows, cols = zip(*matrix_elements.keys())
    #             matrix[rows, cols] = list(matrix_elements.values())
    #             self._coupling_matrix = CouplingMatrix(np.rot90(matrix, 2) + matrix - np.diag(np.diag(matrix)))

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
        return self.f0/self.bw


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

    def response(self, frange, NRNlist=[], Rs=1, Rl=1, PSs=None):
        resp = self.response_from_coupling_matrix(M=self.coupling_matrix, f0=self.f0, BW=self.bw, Q=self.Q,
                                                  frange=frange, NRNlist=NRNlist, Rs=Rs, Rl=Rl, PSs=PSs)
        return resp

    @staticmethod
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

        def RespM2_cpu(M, f0, FBW, Q, frange, NRNlist=[], Rs=1, Rl=1, PSs=None):
            n, n2 = M.shape
            # print ("Order=",n-2)
            if n != n2:
                print("ERROR: M is not Square")
                return (1)
            # for row in range(n):
            #     for col in range(n):
            #         if M[row, col] != M[col, row]:
            #             print("ERROR: M is not symmetric about diagonal")
            #             return (1)

            ######## Q Vector Setup ################

            if isinstance(Q, (list, tuple, np.ndarray)):
                if len(Q) != n - 2 and len(Q) != 1:
                    print("Q Vector length ", len(Q), " is not equal to order ", n - 2, )
                    exit(1)
                if len(Q) != 1:
                    Qvec = np.array(Q)
                else:
                    Qvec = np.ones(n - 2) * Q
            else:
                Qvec = np.ones(n - 2) * Q
            # print("Q-Vector=",Qvec)
            # Qvec = np.ones(n - 2) * Q[1]

            # %%%%%%% Matrix fill %%%%%%%%%%%%%%%%%%%
            I = np.eye(n, dtype=complex)
            I[0, 0] = 0
            I[-1, -1] = 0
            ######### consider non resonating nodes
            for nrn in NRNlist:
                # print("Insert NRN at Res",nrn)
                I[nrn, nrn] = 0
            ######### Resistor Load Matrix

            J = 0 * np.identity(n, dtype=complex)
            J[0, 0] = 1j * Rs
            J[-1, -1] = 1j * Rl

            ## Add losses to resonators due to Q ############# see http://www.jpier.org/PIER/pier115/18.11021604.pdf
            G = 0 * np.identity(n, dtype=complex)
            for res in range(1, n - 1):
                if f0 == 0:
                    G[res, res] = 1 / FBW / Qvec[res - 1]
                else:
                    G[res, res] = 1 / FBW / Qvec[res - 1]
            Mpr = M - 1j * G
            # %%%%%%% Sweep frequencies for S-Paramter %%%%%%%%
            flist = []
            s21list = np.zeros(len(frange), dtype="complex")
            s11list = np.zeros(len(frange), dtype="complex")
            s22list = np.zeros(len(frange), dtype="complex")
            Slist = np.zeros((len(frange), 2, 2), dtype="complex")
            ii = 0
            for f in frange:
                if f0 == 0:
                    lam = 1 / FBW * f
                else:
                    lam = 1 / FBW * (f / f0 - f0 / f)

                Ainv = np.linalg.inv(lam * I - J + Mpr)
                S11 = 1 + 2j * Rs * Ainv[0, 0]
                S22 = 1 + 2j * Rl * Ainv[-1, -1]
                S21 = -2j * np.sqrt(Rs * Rl) * Ainv[-1, 0]

                # take into account phase shifts (PSs)
                if PSs is not None:
                    phi11, phi21, theta11, theta21 = PSs
                    S11 = S11 * np.exp(-1j * 2 * (phi11 + lam * theta11))
                    S22 = S22 * np.exp(-1j * 2 * (phi11 + lam * theta11))
                    S21 = S21 * np.exp(-1j * 2 * (phi21 + lam * theta21))

                s21list[ii] = S21
                s11list[ii] = S11
                s22list[ii] = S22
                Slist[ii] = np.matrix([[S11, S21], [S21, S22]])
                ii += 1

            return Slist

        if torch.cuda.is_available():
            return RespM2_gpu(M, f0, FBW, Q, frange, NRNlist, Rs, Rl, PSs)
        else:
            return RespM2_cpu(M, f0, FBW, Q, frange, NRNlist, Rs, Rl, PSs)

