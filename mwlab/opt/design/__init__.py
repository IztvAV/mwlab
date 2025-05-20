# mwlab/opt/design/__init__.py
"""
MWLab · opt · design
====================

Подпакет для описания проектного пространства (DesignSpace), переменных различных типов
(ContinuousVar, IntegerVar, OrdinalVar, CategoricalVar), генерации DoE‑точек через plug-in сэмплеры
(Sobol, Halton, LHS и др.) и работы с ограничениями.

Основные компоненты:
- Переменные: ContinuousVar, IntegerVar, OrdinalVar, CategoricalVar
- Контейнер пространства: DesignSpace
- Сэмплеры: get_sampler, SobolSampler, HaltonSampler, LHSampler, LHSMaximinSampler, NormalSampler, FactorialFullSampler
"""

from .space import (
    DesignSpace,
    ContinuousVar,
    IntegerVar,
    OrdinalVar,
    CategoricalVar,
)
from .samplers import (
    get_sampler,
    SobolSampler,
    HaltonSampler,
    LHSampler,
    LHSMaximinSampler,
    NormalSampler,
    FactorialFullSampler,
)

__all__ = [
    "DesignSpace",
    "ContinuousVar",
    "IntegerVar",
    "OrdinalVar",
    "CategoricalVar",
    "get_sampler",
    "SobolSampler",
    "HaltonSampler",
    "LHSampler",
    "LHSMaximinSampler",
    "NormalSampler",
    "FactorialFullSampler",
]
