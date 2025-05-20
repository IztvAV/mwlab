#mwlab/opt/objectives/__init__.py
"""
mwlab.opt.objectives
====================
Фасадный модуль для системы целей, критериев и спецификаций СВЧ-проектирования.

- Реализация **Criterion**: селектор → агрегатор → компаратор.
- Регистр/фабрики для компонентов: register_selector, get_selector и др.
- Готовые критерии, агрегаторы, компараторы.
- Класс Specification для комплексных ТЗ.
- YieldObjective: вычисление yield/доли «годных» по surrogate-модели.
"""

from .base import (
    register_selector, register_aggregator, register_comparator,
    get_selector,  # при необходимости можно раскрыть get_aggregator/...
    BaseSelector, BaseAggregator, BaseComparator, BaseCriterion,
)
from .selectors import *
from .aggregators import *
from .comparators import *
from .specification import Specification
from .yield_max import YieldObjective

__all__ = [
    # Фабрики и регистры
    "register_selector", "register_aggregator", "register_comparator",
    "get_selector",
    # Базы
    "BaseSelector", "BaseAggregator", "BaseComparator", "BaseCriterion",
    # Реализации (импортированы через *)
    # ...
    "Specification",
    "YieldObjective",
]
