import numpy as np


class CouplingMatrix:
    def __init__(self, matrix: np.array):
        self._matrix = matrix
        self._links = self.get_links_from_matrix(matrix)
        self._matrix_order = matrix.shape[0]

    @classmethod
    def from_file(cls, file_path):
        """
        Считывает элементы матрицы связи из файла и делает её симметричной.

        Параметры:
        file_path (str): Путь к файлу матрицы связи.

        Возвращает:
        matrix (np.ndarray): симметричная квадратная матрица связи.
        """
        matrix = []
        with open(file_path, mode='r') as file:
            lines = file.readlines()
            for line in lines:
                row = list(map(float, line.strip().split(',')))
                matrix.append(row)

        matrix = np.array(matrix)

        # Проверяем, что матрица квадратная
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной")

        # Делать матрицу симметричной
        sym_matrix = (matrix + matrix.T) / 2
        return cls(sym_matrix)

    @staticmethod
    def get_links_from_matrix(matrix):
        return list(zip(*np.where(np.triu(matrix) != 0)))

    @property
    def matrix(self):
        return self._matrix.copy()

    @property
    def links(self):
        return self._links

    @property
    def matrix_order(self):
        return self._matrix_order

    @property
    def factors(self):
        rows, cols = zip(*self.links)
        factors = self.matrix[rows, cols]
        return np.array(factors)
