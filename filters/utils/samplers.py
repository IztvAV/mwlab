import enum
import math

import torch
from scipy.stats import qmc
from torch.quasirandom import SobolEngine


class SamplerTypes(enum.Enum):
    SAMPLER_UNIFORM = 0
    SAMPLER_FACTORIAL = 1
    SAMPLER_RANDOM = 2
    SAMPLER_LATIN_HYPERCUBE = 3
    SAMPLER_STD = 4
    SAMPLER_GAUSSIAN_SADDLE = 5
    SAMPLER_COMBINE = 6
    SAMPLER_SOBOL = 7

    def __init__(self, value):
        self._value_ = value
        self.one_param = False  # Default value for additional parameter

    def __call__(self, one_param=None):
        if one_param is not None:
            self.one_param = one_param
        return self


class Sampler:
    def __init__(self, type: SamplerTypes, space: torch.Tensor):
        self._type = type
        self._space = space

    @classmethod
    def uniform(cls, start: torch.Tensor, stop: torch.Tensor, num: int, one_param=False, device='cpu'):
        """
        Uniform distribution with option to vary one parameter at a time
        """
        if one_param:
            return cls._one_param_uniform(start, stop, num, device)

        start = torch.as_tensor(start, device=device)
        stop = torch.as_tensor(stop, device=device)
        space = torch.linspace(0, 1, num, device=device).unsqueeze(-1) * (stop - start) + start
        return cls(type=SamplerTypes.SAMPLER_UNIFORM, space=space)

    @classmethod
    def _one_param_uniform(cls, start, stop, num, device):
        median = (start + stop) / 2
        n_params = len(start)
        space = torch.zeros((n_params * num, n_params), device=device)

        for param_idx in range(n_params):
            param_values = torch.linspace(start[param_idx], stop[param_idx], num, device=device)

            for i in range(num):
                row = median.clone()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_UNIFORM(one_param=True), space=space)

    @classmethod
    def std(cls, start, stop, num, mu=0.0, std=1.0, one_param=False, device='cpu'):
        """
        Normal distribution with option to vary one parameter at a time

        Parameters:
            start: Tensor or array-like of minimum values
            stop: Tensor or array-like of maximum values
            num: Number of samples to generate
            mu: Center shift (-1 to 1)
            std: Standard deviation scale factor
            one_param: Whether to vary one parameter at a time
            device: Device to use ('cpu' or 'cuda')
        """
        if one_param:
            return cls._one_param_std(start, stop, num, mu, std, device)

        start = torch.as_tensor(start, device=device)
        stop = torch.as_tensor(stop, device=device)

        if len(start) != len(stop):
            raise ValueError("Start and stop must have same length")

        # Calculate mean and std
        mu_tensor = (start + stop) / 2 + mu * (stop - start) / 2
        base_sigma = torch.abs(stop - start) / 6
        sigma = base_sigma * std

        # Expand to (num, vector_size) shape
        mu_expand = mu_tensor.unsqueeze(0).repeat(num, 1)
        sigma_expand = sigma.unsqueeze(0).repeat(num, 1)

        # Generate normal distribution samples
        space = torch.normal(mu_expand, sigma_expand)

        return cls(type=SamplerTypes.SAMPLER_STD, space=space)

    @classmethod
    def _one_param_std(cls, start, stop, num, mu, std, device):
        median = (start + stop) / 2
        n_params = len(start)
        space = torch.zeros((n_params * num, n_params), device=device)

        for param_idx in range(n_params):
            param_mu = (start[param_idx] + stop[param_idx]) / 2 + mu * (stop[param_idx] - start[param_idx]) / 2
            param_sigma = torch.abs(stop[param_idx] - start[param_idx]) / 6 * std

            param_values = torch.normal(mean=param_mu, std=param_sigma, size=(num,), device=device)
            param_values = torch.clamp(param_values, start[param_idx], stop[param_idx])

            for i in range(num):
                row = median.clone()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_STD(one_param=True), space=space)

    @classmethod
    def lhs(cls, start, stop, num, one_param=False, device='cpu'):
        """
        Latin Hypercube Sampling with option to vary one parameter at a time
        """
        if one_param:
            return cls._one_param_lhs(start, stop, num, device)

        n_params = len(start)
        sampler = qmc.LatinHypercube(d=n_params)
        samples = sampler.random(n=num)
        space = torch.tensor(qmc.scale(samples, start, stop), device=device)
        return cls(type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE, space=space)

    @classmethod
    def _one_param_lhs(cls, start, stop, num, device):
        median = (start + stop) / 2
        n_params = len(start)
        space = torch.zeros((n_params * num, n_params), device=device)

        for param_idx in range(n_params):
            sampler = qmc.LatinHypercube(d=1)
            sample = sampler.random(n=num)
            param_values = torch.tensor(qmc.scale(sample, [start[param_idx]], [stop[param_idx]]).flatten(),
                                        device=device)

            for i in range(num):
                row = median.clone()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE(one_param=True), space=space)

    @classmethod
    def gaussian_saddle(cls, start, stop, num, depth=1.0, one_param=False, device='cpu'):
        """
        Bimodal distribution with option to vary one parameter at a time
        """
        if one_param:
            return cls._one_param_gaussian_saddle(start, stop, num, depth, device)

        min_vector = torch.as_tensor(start, device=device)
        max_vector = torch.as_tensor(stop, device=device)

        if len(min_vector) != len(max_vector):
            raise ValueError("min_vector and max_vector must have same length")

        n = len(min_vector)
        matrix = torch.zeros((n, num), device=device)

        for i in range(n):
            mu1 = min_vector[i] + 0.3 * (max_vector[i] - min_vector[i])
            mu2 = max_vector[i] - 0.3 * (max_vector[i] - min_vector[i])
            sigma = 0.1 * (max_vector[i] - min_vector[i])

            mode1 = torch.normal(mu1, sigma, (num // 2,), device=device)
            mode2 = torch.normal(mu2, sigma, (num // 2,), device=device)
            matrix[i] = torch.cat([mode1, mode2])[:num]
            matrix[i] = matrix[i][torch.randperm(num, device=device)]

        space = matrix.T
        return cls(type=SamplerTypes.SAMPLER_GAUSSIAN_SADDLE, space=space)

    @classmethod
    def _one_param_gaussian_saddle(cls, start, stop, num, depth, device):
        median = (start + stop) / 2
        n_params = len(start)
        space = torch.zeros((n_params * num, n_params), device=device)

        for param_idx in range(n_params):
            mu1 = start[param_idx] + 0.3 * (stop[param_idx] - start[param_idx])
            mu2 = stop[param_idx] - 0.3 * (stop[param_idx] - start[param_idx])
            sigma = 0.1 * (stop[param_idx] - start[param_idx])

            mode1 = torch.normal(mu1, sigma, (num // 2,), device=device)
            mode2 = torch.normal(mu2, sigma, (num // 2,), device=device)
            param_values = torch.cat([mode1, mode2])[:num]
            param_values = torch.clamp(param_values, start[param_idx], stop[param_idx])

            for i in range(num):
                row = median.clone()
                row[param_idx] = param_values[i]
                space[param_idx * num + i] = row

        return cls(type=SamplerTypes.SAMPLER_GAUSSIAN_SADDLE(one_param=True), space=space)

    @classmethod
    def sobol(
            cls,
            start,
            stop,
            num,
            device: str = "cpu",
            *,
            one_param: bool = False,
            seed: int | None = None,
            baseline=None,
    ) -> torch.Tensor:
        start = torch.as_tensor(start, dtype=torch.float32, device=device)
        stop = torch.as_tensor(stop, dtype=torch.float32, device=device)
        assert start.shape == stop.shape, "start и stop должны совпадать по размеру"
        if torch.any(start >= stop):
            raise ValueError("start[i] должен быть меньше stop[i]")
        d = start.numel()

        if not one_param:
            # классический гиперкуб Соболя
            eng = SobolEngine(dimension=d, scramble=True, seed=seed)
            u = eng.draw(num).to(device)  # (num, d) в [0,1]
            space = start + u * (stop - start)
            return cls(type=SamplerTypes.SAMPLER_SOBOL(one_param=False), space=space)

        else:
            # базовые (фиксированные) значения
            if baseline is None:
                baseline = (start + stop) * 0.5
            baseline = torch.as_tensor(baseline, dtype=torch.float32, device=device)
            assert baseline.shape == start.shape, "baseline должен совпадать по размеру"

            # сколько точек на каждую координату
            n_per = math.ceil(num / d)
            samples = []

            for i in range(d):
                # линейная последовательность на отрезке [start[i], stop[i]]
                vals = torch.linspace(start[i], stop[i], n_per, device=device)
                # клонируем baseline и подставляем vals по i‑й координате
                block = baseline.repeat(n_per, 1)
                block[:, i] = vals
                samples.append(block)

            space = torch.cat(samples, dim=0)[:num]  # обрезаем до нужного размера
            return cls(type=SamplerTypes.SAMPLER_SOBOL(one_param=True), space=space)

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
        elif type == SamplerTypes.SAMPLER_SOBOL:
            return cls.sobol(start, stop, num, one_param=type.one_param, **kwargs)
        else:
            raise ValueError(f"Selected sampler not implemented: {type}")

    @classmethod
    def concat(cls, samplers: tuple):
        new_type = SamplerTypes.SAMPLER_COMBINE
        new_space = torch.cat([sampler.space for sampler in samplers])
        return cls(type=new_type, space=new_space)

    @property
    def type(self):
        return self._type

    @property
    def space(self):
        return self._space

    def shuffle(self, ratio=0.3, dim=1, random_state=None):
        if random_state is not None:
            torch.manual_seed(random_state)

        space = self.space.clone()
        n_rows, n_cols = space.shape

        if dim == 1:
            for col in range(n_cols):
                n_shuffle = int(n_rows * ratio)
                shuffle_indices = torch.randperm(n_rows, device=space.device)[:n_shuffle]
                shuffled_values = space[shuffle_indices, col][torch.randperm(n_shuffle, device=space.device)]
                space[shuffle_indices, col] = shuffled_values
        else:
            for row in range(n_rows):
                n_shuffle = int(n_cols * ratio)
                shuffle_indices = torch.randperm(n_cols, device=space.device)[:n_shuffle]
                shuffled_values = space[row, shuffle_indices][torch.randperm(n_shuffle, device=space.device)]
                space[row, shuffle_indices] = shuffled_values

        return Sampler(type=self.type, space=space)

    def to(self, device):
        """Move the sampler to specified device"""
        self._space = self._space.to(device)
        return self

    def __str__(self):
        return f"Sampler with type: {self._type.name}"

    def __len__(self):
        return len(self._space)

    def __getitem__(self, idx):
        return self._space[idx]