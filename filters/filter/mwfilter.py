import skrf as rf
import re
import numpy as np
from filter.couplilng_matrix import CouplingMatrix
from copy import deepcopy as copy


class MWFilter(rf.Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Новые поля
        self._order: int = 0
        self._f0: float = 0.0
        self._bw: float = 0.0
        self._Q: float = 0.0
        self._coupling_matrix: CouplingMatrix | None = None
        self._parse_comments()

    def _parse_comments(self):
        """
        Считывает матричные элементы, сохранённые в одной строке комментария.
        Возвращает: словарь {(i, j): значение}
        """
        matrix_elements = {}
        lines = self.comments

        for line in lines.split("\n"):
            if line.startswith(" N") or line.startswith(" f0") or line.startswith(" bw") or line.startswith(" Q"):
                pattern = re.compile(r" (\w+)\s*:\s*([\d.]+)\s*(MHz)?", re.IGNORECASE)
                match = pattern.match(line)
                key = match.group(1)
                if key == "f0":
                    self._f0 = float(match.group(2))
                elif key == "bw":
                    self._bw = float(match.group(2))
                elif key == "N":
                    self._order = int(match.group(2))
                elif key == "Q":
                    self._Q = float(match.group(2))
            elif line.startswith(" matrix"):
                # Найти все пары вида m_1_2 = 0.123
                matches = re.findall(r"m_(\d+)_(\d+)\s*=\s*([-+eE0-9.]+)", line)
                for i_str, j_str, val_str in matches:
                    i, j, val = int(i_str), int(j_str), float(val_str)
                    matrix_elements[(i, j)] = val
                matrix = np.zeros((self._order+2, self._order+2))
                rows, cols = zip(*matrix_elements.keys())
                matrix[rows, cols] = list(matrix_elements.values())
                self._coupling_matrix = CouplingMatrix(np.rot90(matrix, 2) + matrix - np.diag(np.diag(matrix)))

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
        ntwk = MWFilter(z0=self.z0, s_def=self.s_def, comments=self.comments)

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

