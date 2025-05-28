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
    SAMPLER_COMBINE = 6

    def __init__(self, value):
        self._value_ = value
        self.one_param = False  # Дефолтное значение дополнительного параметра

    def __call__(self, one_param=None):
        if one_param is not None:
            self.one_param = one_param
        return self


class Sampler:
    def __init__(self, type: SamplerTypes, space: np.array):
        self._type = type
        self._space = space

    @classmethod
    def uniform(cls, start: np.array, stop: np.array, num: int, one_param=False):
        """
        Равномерное распределение с опцией изменения одного параметра
        """
        if one_param:
            return cls._one_param_uniform(start, stop, num)
        space = np.linspace(start, stop, num)
        return cls(type=SamplerTypes.SAMPLER_UNIFORM, space=space)

    @classmethod
    def _one_param_uniform(cls, start, stop, num):
        median = (start + stop) / 2
        n_params = len(start)
        space = np.zeros((n_params * num, n_params))

        for param_idx in range(n_params):
            param_values = np.linspace(start[param_idx], stop[param_idx], num)

            for i in range(num):
                row = median.copy()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_ONE_PARAM, space=space)

    @classmethod
    def std(cls, start, stop, num, mu=0.0, std=1.0, one_param=False):
        """
        Нормальное распределение с опцией изменения одного параметра
        """
        if one_param:
            return cls._one_param_std(start, stop, num, mu, std)

        if len(start) != len(stop):
            raise ValueError("Длины start и stop должны совпадать")

        mu = (start + stop) / 2 + mu * (stop - start) / 2
        base_sigma = np.abs(stop - start) / 6
        sigma = base_sigma * std

        space = np.random.normal(loc=mu, scale=sigma, size=(num, len(mu)))
        return cls(type=SamplerTypes.SAMPLER_STD, space=space)

    @classmethod
    def _one_param_std(cls, start, stop, num, mu, std):
        median = (start + stop) / 2
        n_params = len(start)
        space = np.zeros((n_params * num, n_params))

        for param_idx in range(n_params):
            param_mu = (start[param_idx] + stop[param_idx]) / 2 + mu * (stop[param_idx] - start[param_idx]) / 2
            param_sigma = np.abs(stop[param_idx] - start[param_idx]) / 6 * std

            param_values = np.random.normal(loc=param_mu, scale=param_sigma, size=num)
            param_values = np.clip(param_values, start[param_idx], stop[param_idx])

            for i in range(num):
                row = median.copy()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_ONE_PARAM, space=space)

    @classmethod
    def lhs(cls, start, stop, num, one_param=False):
        """
        Latin Hypercube Sampling с опцией изменения одного параметра
        """
        if one_param:
            return cls._one_param_lhs(start, stop, num)

        # start = np.asarray(start, dtype=float)
        # stop = np.asarray(stop, dtype=float)
        #
        # assert start.shape == stop.shape, "Start and stop must have the same shape"
        # active_mask = (start != 0) | (stop != 0)
        # start_flat = start[active_mask]
        # stop_flat = stop[active_mask]
        #
        # if not np.all(start_flat < stop_flat):
        #     raise ValueError("Each nonzero `start` must be strictly less than corresponding `stop`.")
        #
        # dim = start_flat.size
        # sampler = qmc.LatinHypercube(d=dim)
        # sample = sampler.random(n=num)
        # scaled = qmc.scale(sample, start_flat, stop_flat)
        #
        # full_samples = np.zeros((num, *start.shape), dtype=float)
        # for i in range(num):
        #     full_samples[i][active_mask] = scaled[i]
        # Количество ненулевых параметров

        # Генерируем LHS выборку
        n_params = len(start)
        sampler = qmc.LatinHypercube(d=n_params)
        samples = sampler.random(n=num)  # В диапазоне [0, 1]^d

        # Масштабируем к заданным диапазонам
        space = qmc.scale(samples, start, stop)

        return cls(type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE, space=space)

    @classmethod
    def _one_param_lhs(cls, start, stop, num):
        median = (start + stop) / 2
        n_params = len(start)
        space = np.zeros((n_params * num, n_params))

        for param_idx in range(n_params):
            sampler = qmc.LatinHypercube(d=1)
            sample = sampler.random(n=num)
            param_values = qmc.scale(sample, [start[param_idx]], [stop[param_idx]]).flatten()

            for i in range(num):
                row = median.copy()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE(one_param=True), space=space)

    @classmethod
    def gaussian_saddle(cls, start, stop, num, depth=1.0, one_param=False):
        """
        Бимодальное распределение с опцией изменения одного параметра
        """
        if one_param:
            return cls._one_param_gaussian_saddle(start, stop, num, depth)

        min_vector = np.asarray(start)
        max_vector = np.asarray(stop)

        if len(min_vector) != len(max_vector):
            raise ValueError("Длины min_vector и max_vector должны совпадать")

        n = len(min_vector)
        matrix = np.zeros((n, num))

        for i in range(n):
            mu1 = min_vector[i] + 0.3 * (max_vector[i] - min_vector[i])
            mu2 = max_vector[i] - 0.3 * (max_vector[i] - min_vector[i])
            sigma = 0.1 * (max_vector[i] - min_vector[i])

            mode1 = np.random.normal(mu1, sigma, num // 2)
            mode2 = np.random.normal(mu2, sigma, num // 2)
            matrix[i] = np.concatenate([mode1, mode2])
            np.random.shuffle(matrix[i])
        space = matrix.T

        return cls(type=SamplerTypes.SAMPLER_GAUSSIAN_SADDLE, space=space)

    @classmethod
    def _one_param_gaussian_saddle(cls, start, stop, num, depth):
        median = (start + stop) / 2
        n_params = len(start)
        space = np.zeros((n_params * num, n_params))

        for param_idx in range(n_params):
            mu1 = start[param_idx] + 0.3 * (stop[param_idx] - start[param_idx])
            mu2 = stop[param_idx] - 0.3 * (stop[param_idx] - start[param_idx])
            sigma = 0.1 * (stop[param_idx] - start[param_idx])

            mode1 = np.random.normal(mu1, sigma, num // 2)
            mode2 = np.random.normal(mu2, sigma, num // 2)
            param_values = np.concatenate([mode1, mode2])[:num]
            param_values = np.clip(param_values, start[param_idx], stop[param_idx])

            for i in range(num):
                row = median.copy()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_ONE_PARAM, space=space)

    @classmethod
    def for_type(cls, start, stop, num, type: SamplerTypes, **kwargs):
        if type == SamplerTypes.SAMPLER_UNIFORM:
            return cls.uniform(start, stop, num, one_param=type.one_param, **kwargs)
        elif type == SamplerTypes.SAMPLER_STD:
            return cls.std(start, stop, num, one_param=type.one_param, **kwargs)
        elif type == SamplerTypes.SAMPLER_LATIN_HYPERCUBE:
            return cls.lhs(start, stop, num, one_param=type.one_param, **kwargs)
        elif type == SamplerTypes.SAMPLER_GAUSSIAN_SADDLE:
            return cls.gaussian_saddle(start, stop, num, one_param=type.one_param, **kwargs)
        else:
            raise ValueError(f"Выбранный сэмплер не реализован: {type}")

    @classmethod
    def concat(cls, samplers:tuple):
        new_type = SamplerTypes.SAMPLER_COMBINE
        new_space = np.concatenate([sampler.space for sampler in samplers])
        return cls(type=new_type, space=new_space)

    @property
    def type(self):
        return self._type

    @property
    def space(self):
        return self._space

    def shuffle(self, ratio=0.3, dim=1, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        space = self.space.copy()
        n_rows, n_cols = space.shape

        if dim == 1:
            for col in range(n_cols):
                n_shuffle = int(n_rows * ratio)
                shuffle_indices = np.random.choice(n_rows, size=n_shuffle, replace=False)
                shuffled_values = np.random.permutation(space[shuffle_indices, col])
                space[shuffle_indices, col] = shuffled_values
        else:
            for row in range(n_rows):
                n_shuffle = int(n_cols * ratio)
                shuffle_indices = np.random.choice(n_cols, size=n_shuffle, replace=False)
                shuffled_values = np.random.permutation(space[row, shuffle_indices])
                space[row, shuffle_indices] = shuffled_values
        return Sampler(type=self.type, space=space)

    def __str__(self):
        return f"Sampler with type: {self._type.name}"

    def __len__(self):
        return len(self._space)

    def __getitem__(self, idx):
        return self._space[idx]
