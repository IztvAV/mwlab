#
#   100% работает с
#   python 3.11
#   numpy 2.2.4
#   pyDOE3 1.0.4 (для распределения гиперкубом)
#
from typing import Iterable

import numpy as np
from pyDOE3 import lhs

class Scaler:
    def __init__(self, min, max):
        self.min = min
        self.span = max - min

    def scale(self, values):
        return self.min + values * self.span


class Sampler(Iterable):
    """
    Примеры использования:

    Гиперкуб со скидыванием семплов в список семплов
    >>> sampler = Sampler.hypercube(num_factors=1, num_samples=10)
    >>> samples = sampler.samples()
    >>> print(samples)

    Рандом с печатью семплов в цикле
    >>> sampler = Sampler.random(num_factors=19, num_samples=10, seed=42)
    >>> for s in sampler:
    >>>     print(s)

    Масштабирование отдельно от сэмплера
    >>> vmin = np.linspace(3, 4, 19, endpoint=True) # Какой-то вектор минимальных значений
    >>> vmax = np.linspace(10, 11, 19, endpoint=True) # Какой-то вектор максимальных значения
    >>> vspan = vmax - vmin
    >>> scaler = Scaler(vmin, vmax)
    >>> for s in sampler:
    >>>     print(vmin + s * vspan)

    Масштабирование внутри сэмплера
    >>> scaler = Scaler(3, 4)
    >>> sampler.scaler = scaler
    >>> print(sampler.samples())

    Из Sampler1d
    >>> seed = np.random.RandomState(42)
    >>> s1 = Sampler1d.hypercube(10, seed=seed)
    >>> s2 = Sampler1d.hypercube(10, seed=seed)
    >>> s3 = Sampler1d.hypercube(10, seed=seed)
    >>>
    >>> print(s1.samples())
    >>>
    >>> sampler = Sampler.from_sampler1d([s1, s2, s3])
    >>> print(sampler.samples())

    """

    def __init__(self, iterator_factory, scaler:Scaler=None):
        self.iterator_factory = iterator_factory
        self.scaler = scaler

    def __iter__(self):
        if self.scaler is None:
            return self.iterator_factory()
        return _ScalingIterator(scaler=self.scaler, iterator=self.iterator_factory())

    def samples(self):
        return np.vstack(list(self.__iter__()))

    @staticmethod
    def from_iterable(values: Iterable, scaler:Scaler=None) -> "Sampler":
        return Sampler(lambda: values.__iter__(), scaler=scaler)

    @staticmethod
    def from_sampler1d(samplers: list["Sampler1d"]):
        return Sampler(lambda: _SamplerNdIterator(samplers=samplers))

    @staticmethod
    def hypercube(num_factors, num_samples, seed:int=None, scaler:Scaler=None):
        return Sampler.from_iterable(
            values=lhs(n=num_factors, samples=num_samples, random_state=seed),
            scaler=scaler)

    @staticmethod
    def random(num_factors, num_samples, seed:int=None, scaler:Scaler=None):
        return Sampler(
            iterator_factory=lambda: _RandomSampleIterator(n=num_factors, samples=num_samples, rng_or_seed=seed),
            scaler=scaler)


class Sampler1d(Iterable):
    def __init__(self, iterator_factory, scaler:Scaler=None):
        self.iterator_factory = iterator_factory
        self.scaler = scaler

    def __iter__(self):
        if self.scaler is None:
            return self.iterator_factory()
        return _ScalingIterator(scaler=self.scaler, iterator=self.iterator_factory())

    def samples(self):
        return np.fromiter(self, float)


    def scale_from_to(self, min, max):
        if min == max:
            raise ValueError('range length is zero')
        if min > max:
            temp = min
            min = max
            max = temp

        return min + self.samples() * (max - min)

    def scale_center_delta(self, center, delta):
        if delta == 0:
            raise ValueError('range length is zero')
        if delta < 0:
            delta = np.abs(delta)
        return (center - delta) + self.samples() * (delta * delta)


    @staticmethod
    def from_iterable(values: Iterable, scaler:Scaler=None) -> "Sampler1d":
        return Sampler1d(lambda: values.__iter__(), scaler=scaler)


    @staticmethod
    def random(len: int, seed: int, scaler:Scaler=None) -> "Sampler1d":
        return Sampler1d(lambda: _RandomSample1dIterator(len, seed), scaler=scaler)


    @staticmethod
    def uniform(len: int, endpoint:bool=True, scaler:Scaler=None) -> "Sampler1d":
        return Sampler1d.from_iterable(np.linspace(0, 1, num=len, endpoint=endpoint), scaler=scaler)


    @staticmethod
    def hypercube(len: int, scaler:Scaler=None):
        return Sampler1d.from_iterable(lhs(1, len)[:, 0], scaler=scaler)


def _get_generator(rng_or_seed):
    if rng_or_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(rng_or_seed)

def _get_random_state(random_state):
    if random_state is None:
        return np.random.RandomState()
    if not isinstance(random_state, np.random.RandomState):
        return np.random.RandomState(random_state)
    return random_state


class _ScalingIterator:
    def __init__(self, scaler: Scaler, iterator):
        self.scaler = scaler
        self.it = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return self.scaler.scale(self.it.__next__())


class SampleIterator:
    def __init__(self, len:int):
        self.len = len
        self.index = 0

    def __iter__(self):
        return self

    def next_element(self):
        pass

    def __next__(self):
        if self.index < self.len:
            result = self.next_element()
            self.index += 1
            return result
        else:
            raise StopIteration

class _RandomSampleIterator(SampleIterator):
    def __init__(self, n: int, samples: int, rng_or_seed=None):
        super().__init__(samples)
        self.rng = _get_generator(rng_or_seed)
        self.n = n

    def next_element(self):
        return self.rng.random(self.n)



class _RandomSample1dIterator(SampleIterator):
    def __init__(self, len: int, seed: int):
        super().__init__(len)
        self.rng = _get_generator(seed)

    def next_element(self):
        return self.rng.random()


class _SamplerNdIterator:
    def __init__(self, samplers: list[Sampler1d]):
        self.iters = [s.__iter__() for s in samplers]
        self.n = len(samplers)

    def __iter__(self):
        return self

    def __next__(self):
        values = np.zeros(self.n)
        for i in range(self.n):
            values[i] = self.iters[i].__next__()
        return values

