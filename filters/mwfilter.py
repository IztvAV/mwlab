import skrf as rf
import couplilng_matrix as cm
import re
import numpy as np


class MWFilter(rf.Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Новые поля
        self._order = 0
        self._f0 = 0
        self._bw = 0
        self._Q = 0
        self._coupling_matrix = None
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
                    self._order = int(match.group(2))
            elif line.startswith(" matrix"):
                # Найти все пары вида m_1_2 = 0.123
                matches = re.findall(r"m_(\d+)_(\d+)\s*=\s*([-+eE0-9.]+)", line)
                for i_str, j_str, val_str in matches:
                    i, j, val = int(i_str), int(j_str), float(val_str)
                    matrix_elements[(i, j)] = val
                matrix = np.zeros((self._order+2, self._order+2))
                rows, cols = zip(*matrix_elements.keys())
                matrix[rows, cols] = list(matrix_elements.values())
                self._coupling_matrix = np.rot90(matrix, 2) + matrix - np.diag(np.diag(matrix))

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
