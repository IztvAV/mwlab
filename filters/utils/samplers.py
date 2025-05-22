import enum
import numpy as np
from scipy.stats import qmc


class SamplerTypes(enum.Enum):
    SAMPLER_UNIFORM = 0,
    SAMPLER_FACTORIAL = 1,
    SAMPLER_RANDOM = 2,
    SAMPLER_LATIN_HYPERCUBE = 3,
    SAMPLER_STD = 4,
    SAMPLER_GAUSSIAN_SADDLE = 5,


class Sampler:
    def __init__(self, type: SamplerTypes, space: np.array):
        self._type = type
        self._space = space

    @classmethod
    def uniform(cls, start: np.array, stop: np.array, num: int):
        space = np.linspace(start, stop, num)
        return cls(type=SamplerTypes.SAMPLER_UNIFORM, space=space)

    @classmethod
    def lhs(cls, start, stop, num):
        start = np.asarray(start, dtype=float)
        stop = np.asarray(stop, dtype=float)

        assert start.shape == stop.shape, "Start and stop must have the same shape"

        # Маска тех элементов, где требуется генерация
        active_mask = (start != 0) | (stop != 0)

        # Извлекаем только активные элементы
        start_flat = start[active_mask]
        stop_flat = stop[active_mask]

        if not np.all(start_flat < stop_flat):
            raise ValueError("Each nonzero `start` must be strictly less than corresponding `stop`.")

        dim = start_flat.size

        # Генерация точек
        sampler = qmc.LatinHypercube(d=dim)
        sample = sampler.random(n=num)
        scaled = qmc.scale(sample, start_flat, stop_flat)  # (num, dim)

        # Восстановим полную структуру
        full_samples = np.zeros((num, *start.shape), dtype=float)
        for i in range(num):
            full_samples[i][active_mask] = scaled[i]

        return cls(type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE, space=full_samples)

    @classmethod
    def std(cls, start, stop, num):
        # Проверка на одинаковую длину векторов min и max
        if len(start) != len(stop):
            raise ValueError("Длины векторов min и max должны совпадать.")

        # Вычисляем среднее (μ) и стандартное отклонение (σ)
        mu = (start + stop) / 2
        sigma = (stop - mu) / 3  # Правило трёх сигм

        # Генерируем нормальное распределение
        space = np.random.normal(loc=mu, scale=sigma, size=(num, len(mu)))
        return cls(SamplerTypes.SAMPLER_STD, space)

    @classmethod
    def gaussian_saddle(cls, start, stop, num, depth):
        """
         Генерирует матрицу бимодальных распределений для каждого элемента.

         Параметры:
             min_vector (np.array): Вектор минимальных значений (shape=(n,)).
             max_vector (np.array): Вектор максимальных значений (shape=(n,)).
             size (int): Размер выборки для каждого элемента.

         Возвращает:
             np.array: Матрица shape=(n, size) с бимодальными распределениями.
         """
        min_vector = np.asarray(start)
        max_vector = np.asarray(stop)

        if len(min_vector) != len(max_vector):
            raise ValueError("Длины min_vector и max_vector должны совпадать")

        n = len(min_vector)
        matrix = np.zeros((n, num))

        for i in range(n):
            # Центры двух мод (сдвинуты к min и max)
            mu1 = min_vector[i] + 0.3 * (max_vector[i] - min_vector[i])
            mu2 = max_vector[i] - 0.3 * (max_vector[i] - min_vector[i])

            # Стандартное отклонение (10% от диапазона)
            sigma = 0.1 * (max_vector[i] - min_vector[i])

            # Генерация двух мод
            mode1 = np.random.normal(mu1, sigma, num // 2)
            mode2 = np.random.normal(mu2, sigma, num // 2)

            # Объединение и перемешивание
            matrix[i] = np.concatenate([mode1, mode2])
            np.random.shuffle(matrix[i])
        space = matrix.T

        return cls(SamplerTypes.SAMPLER_GAUSSIAN_SADDLE, space)

    @classmethod
    def for_type(cls, start, stop, num, type, **kwargs):
        if type == SamplerTypes.SAMPLER_UNIFORM:
            return cls.uniform(start, stop, num)
        elif type == SamplerTypes.SAMPLER_STD:
            return cls.std(start, stop, num)
        elif type == SamplerTypes.SAMPLER_LATIN_HYPERCUBE:
            return cls.lhs(start, stop, num)
        elif type == SamplerTypes.SAMPLER_GAUSSIAN_SADDLE:
            return cls.gaussian_saddle(start, stop, num, **kwargs)
        else:
            raise ValueError(f"Выбранный сэмплер не реализован: {type}")

    @property
    def type(self):
        return self._type

    @property
    def range(self):
        return self._space

    def __str__(self):
        return f"Sampler with type: {self._type.name}"

    def __len__(self):
        return len(self._space)

    def __getitem__(self, idx):
        return self._space[idx]