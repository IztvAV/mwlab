#mwlab/opt/objectives/aggregators.py
"""
aggregators.py
==============
«Как» из 1-D вектора сделать скаляр.

* MaxAgg      – максимум;
* MinAgg      – минимум;
* MeanAgg     – среднее;
* RippleAgg   – max-min.

Все агрегаторы игнорируют `freq`, но получают его на вход для совместимости.
"""
from __future__ import annotations
import numpy as np

from .base import BaseAggregator, register_aggregator


# ---------------------------------------------------------------- max / min / mean
@register_aggregator("max")
class MaxAgg(BaseAggregator):
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.max(vals))


@register_aggregator("min")
class MinAgg(BaseAggregator):
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.min(vals))


@register_aggregator("mean")
class MeanAgg(BaseAggregator):
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.mean(vals))


# ---------------------------------------------------------------- ripple
@register_aggregator("ripple")
class RippleAgg(BaseAggregator):
    """Разброс max-min."""

    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.max(vals) - np.min(vals))


# ---------------------------------------------------------------- TODO-заглушки
@register_aggregator("std")
class StdAgg(BaseAggregator):
    """Стандартное отклонение (пока не реализовано)."""
    def __call__(self, *_):
        raise NotImplementedError
