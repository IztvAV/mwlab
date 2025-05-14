import enum
import numpy as np
from scipy.stats import qmc


class SamplerTypes(enum.Enum):
    SAMPLER_UNIFORM = 0,
    SAMPLER_FACTORIAL = 1,
    SAMPLER_RANDOM = 2,
    SAMPLER_LATIN_HYPERCUBE = 3,


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