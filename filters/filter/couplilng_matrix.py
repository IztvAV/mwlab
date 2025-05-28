import numpy as np
import matplotlib.pyplot as plt


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
    def from_factors(factors, links, matrix_order):
        M = np.zeros(shape=(matrix_order, matrix_order), dtype=float)
        rows, cols = zip(*links)
        M[rows, cols] = factors
        M[cols, rows] = factors
        return M


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

    def plot_matrix(self, decimals=4, title="Coupling matrix", cmap="viridis"):
        """
        Отображает матрицу в виде таблицы на графике

        Параметры:
            matrix : numpy.ndarray или torch.Tensor
                Входная матрица (2D)
            decimals : int, optional
                Количество знаков после запятой (по умолчанию 2)
            title : str, optional
                Заголовок графика
            cmap : str, optional
                Цветовая схема (из matplotlib)
        """
        # Конвертируем в numpy (если это torch.Tensor)
        if hasattr(self.matrix, 'numpy'):
            self._matrix = self.matrix.detach().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        # Скрываем оси
        ax.axis('off')
        ax.axis('tight')

        # Создаем таблицу
        table = ax.table(
            cellText=np.round(self.matrix, decimals=decimals),
            cellLoc='center',
            loc='center',
            colLabels=None,
            rowLabels=None
        )

        # Настраиваем стиль
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Масштабирование

        # Добавляем цветовую карту
        # Добавляем цветовую карту и обработку нулей
        if cmap is not None:
            # Нормализуем данные (исключая нули для цветовой карты)
            norm_matrix = self._matrix.copy()
            non_zero_mask = self._matrix != 0
            if np.any(non_zero_mask):
                norm_matrix[non_zero_mask] = (self._matrix[non_zero_mask] - np.min(self._matrix[non_zero_mask])) / \
                                             (np.max(self._matrix[non_zero_mask]) - np.min(self._matrix[non_zero_mask]))

            colors = plt.cm.get_cmap(cmap)(norm_matrix)

            for (i, j), val in np.ndenumerate(self._matrix):
                if val == 0:
                    table[(i, j)].set_facecolor("white")
                else:
                    table[(i, j)].set_facecolor(colors[i, j])

        plt.title(title)
