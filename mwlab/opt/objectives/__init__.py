# mwlab/opt/objectives/__init__.py
"""
mwlab.opt.objectives
====================

Фасад (public API) подсистемы целей/ограничений.

Подсистема предназначена для вычисления численных метрик/штрафов по частотным кривым
и поддержки «спецификаций» (комплексных ТЗ), включая интегральные/знаковые меры и
yield-ориентированные цели.

Архитектура (концептуальная цепочка)
------------------------------------
    Selector -> Transform -> Aggregator -> Comparator
                    \-> Criterion (склейка цепочки в объект вычисления)

- Selector извлекает частотную зависимость из источника данных (обычно rf.Network).
- Transform выполняет предобработку (band, сглаживание, ресэмплинг, производные и т.п.).
- Aggregator сворачивает кривую в скаляр.
- Comparator интерпретирует скаляр как ограничение/штраф.
- Specification собирает набор критериев в комплексное ТЗ.

Плагины и регистрация
--------------------
Компоненты (selectors/transforms/aggregators/comparators) регистрируются через декораторы
register_* в момент импорта соответствующих модулей. Поэтому этот фасадный модуль
намеренно импортирует «встроенные» подмодули при загрузке пакета, чтобы реестры были
готовы сразу после:

    import mwlab.opt.objectives as obj

Рекомендуемый способ импорта
----------------------------
Используйте этот фасад вместо импорта из внутренних модулей:

    from mwlab.opt.objectives import (
        # базовые абстракции
        BaseSelector, BaseTransform, BaseAggregator, BaseComparator, BaseCriterion,
        # фабрики/реестры
        register_selector, register_transform, register_aggregator, register_comparator,
        get_selector, get_transform, get_aggregator, get_comparator,
        # готовые классы
        BandTransform, ResampleTransform, MeanAgg, UpIntAgg,
        Specification,
    )

Примечание по стабильности API
------------------------------
- Всё, что перечислено в __all__, считается публичным интерфейсом.
- Внутренние вспомогательные функции из подмодулей не реэкспортируются.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Базы и реестры/фабрики (ядро API)
# ---------------------------------------------------------------------------

from .base import (
    # реестры/регистрация
    register_selector,
    register_transform,
    register_aggregator,
    register_comparator,
    # фабрики
    get_selector,
    get_transform,
    get_aggregator,
    get_comparator,
    # базовые абстракции
    BaseSelector,
    BaseTransform,
    BaseAggregator,
    BaseComparator,
    BaseCriterion,
)

# ---------------------------------------------------------------------------
# Встроенные реализации и side-effects импортов для регистрации
# ---------------------------------------------------------------------------

from . import selectors as _selectors
from . import transforms as _transforms
from . import aggregators as _aggregators
from . import comparators as _comparators

def _reexport(module, names: Iterable[str]) -> None:
    """
    Реэкспорт имён из подмодуля в пространство имён пакета.

    Почему так сделано:
    - минимизируем `import *`;
    - `__all__` формируется явно из подмодулей;
    - пользователю удобно импортировать из mwlab.opt.objectives,
      не зная внутреннюю структуру пакета.
    """
    g = globals()
    for name in names:
        g[name] = getattr(module, name)


# Реэкспортируем только то, что подмодуль объявляет публичным через __all__.
_reexport(_transforms, getattr(_transforms, "__all__", ()))
_reexport(_aggregators, getattr(_aggregators, "__all__", ()))
_reexport(_comparators, getattr(_comparators, "__all__", ()))

if _selectors is not None:
    _reexport(_selectors, getattr(_selectors, "__all__", ()))

# ---------------------------------------------------------------------------
# Высокоуровневые сущности (Specification / Objectives)
# ---------------------------------------------------------------------------

from .specification import Specification
from .yield_max import YieldObjective
from .penalty import PenaltyObjective, FeasibleYieldObjective


# ---------------------------------------------------------------------------
# Публичный экспорт (public API)
# ---------------------------------------------------------------------------

__all__ = [
    # --- фабрики и регистры ---
    "register_selector",
    "register_transform",
    "register_aggregator",
    "register_comparator",
    "get_selector",
    "get_transform",
    "get_aggregator",
    "get_comparator",
    # --- базовые абстракции ---
    "BaseSelector",
    "BaseTransform",
    "BaseAggregator",
    "BaseComparator",
    "BaseCriterion",
    # --- готовые реализации из подмодулей (см. их __all__) ---
    *tuple(getattr(_transforms, "__all__", ())),
    *tuple(getattr(_aggregators, "__all__", ())),
    *tuple(getattr(_comparators, "__all__", ())),
    *tuple(getattr(_selectors, "__all__", ())),
    # --- высокоуровневые сущности ---
    "Specification",
    "YieldObjective",
    "PenaltyObjective",
    "FeasibleYieldObjective",
]
