import enum
import numpy as np


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
    def uniform(cls, start, stop, num):
        space = np.linspace(start, stop, num)
        return cls(type=SamplerTypes.SAMPLER_UNIFORM, space=space)

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