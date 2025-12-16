# mwlab/opt/objectives/base.py
"""
mwlab.opt.objectives.base
=========================

**Пакет “objectives” строится из трех ортогональных кирпичей**

1. **Selector**   – «что» брать из `rf.Network`
2. **Aggregator** – «как» сворачивать вектор частот → скаляр
3. **Comparator** – «хорошо / плохо» либо числовая `penalty()`

Комбинация этих трех объектов образует **Criterion**.  Набор Criterion-ов,
объединенный конъюнкцией AND, – это *Specification* (будет в соседнем
файле).  Такая декомпозиция повторяет подход Ansys HFSS-Goals и Keysight ADS
(Metric + Aggregation + Goal) и дает гибкость: чтобы описать новую цель,
достаточно создать *один* мозычек и зарегистрировать его декоратором.

В этом файле:

* базовые абстрактные классы;
* минимальный “think-ahead” API (`select → ndarray`, `aggregate → float`,
  `is_ok → bool`, `penalty → float`);
* **registry-pattern** для автоподключения модулей.

> Заглушки и реализация по-умолчанию (NotImplementedError) позволяют
> писать тесты и код следующих уровней, не боясь сломать контракт,
> — просто подменяем конкретный компонент позже.

Примеры:
--------
from mwlab.opt.objectives.base import BaseCriterion
from mwlab.opt.objectives.selectors import SMagSelector
from mwlab.opt.objectives.aggregators import MaxAgg, MinAgg, MeanAgg, RippleAgg, StdAgg, UpIntAgg, LoIntAgg, RippleIntAgg
from mwlab.opt.objectives.comparators import LEComparator, GEComparator, SoftLEComparator, HingeLEComparator

# Критерий: максимальные обратные потери (|S11| в дБ) должны быть ≤ –22 дБ
crit_rl = BaseCriterion(
    selector   = SMagSelector(1, 1, band=(2.0, 2.4), db=True),
    aggregator = MaxAgg(),                         # берём максимум
    comparator = LEComparator(limit=-22, unit="dB"),
    name       = "S11_max"
)

# Критерий: минимальные проходные потери (|S21|) должны быть ≥ –1 дБ
crit_il = BaseCriterion(
    selector   = SMagSelector(2, 1, band=(2.0, 2.4), db=True),
    aggregator = MinAgg(),                         # берём минимум
    comparator = GEComparator(limit=-1.0, unit="dB"),
    name       = "S21_min"
)

# Критерий: среднее ослабление в полосе должно быть ≤ –20 дБ
crit_avg = BaseCriterion(
    selector   = SMagSelector(3, 1, band=(3.0, 3.5), db=True),
    aggregator = MeanAgg(),
    comparator = LEComparator(limit=-20, unit="dB"),
    name       = "S31_avg"
)

# Критерий: рябь (размах) в полосе ≤ 0.5 дБ
crit_ripple = BaseCriterion(
    selector   = SMagSelector(2, 1, band=(2.0, 2.4), db=True),
    aggregator = RippleAgg(),
    comparator = LEComparator(limit=0.5, unit="dB"),
    name       = "S21_ripple"
)

# Критерий: стандартное отклонение ослабления ≤ 0.25 дБ
crit_std = BaseCriterion(
    selector   = SMagSelector(2, 1, band=(2.0, 2.4), db=True),
    aggregator = StdAgg(),
    comparator = LEComparator(limit=0.25, unit="dB"),
    name       = "S21_std"
)

# Критерий: интеграл нарушений по |S11| ≤ –22 дБ
crit_rl_area = BaseCriterion(
    selector   = SMagSelector(1, 1, band=(2.0, 2.4), db=True),
    aggregator = UpIntAgg(limit=-22, p=2, method="mean", normalize="bandwidth*limit"),
    comparator = HingeLEComparator(limit=0.0, scale=1.0),  # сравнение с нулём
    name       = "S11_area"
)

# Критерий: интеграл нарушений по |S21| ≥ –1 дБ
crit_il_area = BaseCriterion(
    selector   = SMagSelector(2, 1, band=(2.0, 2.4), db=True),
    aggregator = LoIntAgg(limit=-1.0, p=2, method="trapz", normalize="bandwidth"),
    comparator = SoftLEComparator(limit=0.0, margin=1.0),  # мягкий штраф
    name       = "S21_area"
)

# Критерий: интегральная рябь пропускания в полосе ≤ 0.2 дБ
crit_il_ripple = BaseCriterion(
    selector   = SMagSelector(2, 1, band=(2.0, 2.4), db=True),
    aggregator = RippleIntAgg(target="linear", deadzone=0.1, p=2,
                              method="mean", normalize="bandwidth"),
    comparator = HingeLEComparator(limit=0.0, scale=1.0),
    name       = "S21_ripple_area"
)

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar, Callable, Sequence, Tuple, List, Union, Iterable

import numpy as np
import skrf as rf


# ────────────────────────────────────────────────────────────────────────────
#  Registry helpers ― чтобы @register_selector(...) было максимально похоже
#  на существующие регистры (samplers, surrogates и т. д.)
# ────────────────────────────────────────────────────────────────────────────
T = TypeVar("T")

def _make_register(reg: Dict[str, Type[T]], kind: str) -> Callable[[Union[str, Iterable[str]]], Callable[[Type[T]], Type[T]]]:
    def _decor(alias: Union[str, Iterable[str]]):
        aliases = [alias] if isinstance(alias, str) else list(alias)
        def _wrap(cls: Type[T]) -> Type[T]:
            for name in aliases:
                if name in reg:
                    raise KeyError(f"{kind.capitalize()} alias '{name}' already exists")
                reg[name] = cls
            setattr(cls, "_aliases", aliases)
            return cls
        return _wrap
    return _decor


_SELECTOR_REG: Dict[str, "BaseSelector"] = {}
_AGGREGATOR_REG: Dict[str, "BaseAggregator"] = {}
_COMPARATOR_REG: Dict[str, "BaseComparator"] = {}


register_selector   = _make_register(_SELECTOR_REG,   "selector")
register_aggregator = _make_register(_AGGREGATOR_REG, "aggregator")
register_comparator = _make_register(_COMPARATOR_REG, "comparator")


def get_selector(alias: str, **kw):
    if alias not in _SELECTOR_REG:
        raise KeyError(f"Selector '{alias}' not found. Available: {list(_SELECTOR_REG)}")
    return _SELECTOR_REG[alias](**kw)


def get_aggregator(alias: str, **kw):
    return _AGGREGATOR_REG[alias](**kw)

def get_comparator(alias: str, **kw):
    return _COMPARATOR_REG[alias](**kw)

# ────────────────────────────────────────────────────────────────────────────
#  Base-классы
# ────────────────────────────────────────────────────────────────────────────
class BaseSelector(ABC):
    """
    Выбирает из `rf.Network` вектор значений (например |S11|[dB]).
    """

    @abstractmethod
    def __call__(self, net: rf.Network) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает кортеж *(freq_GHz, values)*

        *freq_GHz*  – 1-D вектор частот (для информации агрегатору);
        *values*    – 1-D вектор значений длиной = n_freqs.
        """


class BaseAggregator(ABC):
    """Сворачивает 1-D массив → скаляр (max, ripple, mean…)."""

    @abstractmethod
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        ...


class BaseComparator(ABC):
    """Решает «норма выполняется?» и/или вычисляет штраф."""

    @abstractmethod
    def is_ok(self, value: float) -> bool: ...

    def penalty(self, value: float) -> float:
        """
        ≥0, 0 означает *ok*.  По-умолчанию — бинарная ступень (0/1).
        Гладкие пеналти переопределяются в наследниках.
        """
        return 0.0 if self.is_ok(value) else 1.0


# ────────────────────────────────────────────────────────────────────────────
#  Criterion = Selector ∘ Aggregator ∘ Comparator
# ────────────────────────────────────────────────────────────────────────────
class BaseCriterion:
    """
    Мини-инкапсуляция тройки *(selector, aggregator, comparator)*.

    * Не содержит “band”; Selector сам решает по каким частотам брать
      значения (band можно передать в его конструктор).
    * `name` используется в логах и таблицах метрик.
    """

    def __init__(
        self,
        selector: BaseSelector,
        aggregator: BaseAggregator,
        comparator: BaseComparator,
        weight: float = 1.0,
        name: str = "",
    ):
        self.selector = selector
        self.agg = aggregator
        self.comp = comparator
        self.weight = float(weight)
        if self.weight < 0:
            raise ValueError("weight must be >= 0")
        self.name = name or getattr(selector, "name", "crit")

    # ---------------------------------------------------------------- value
    def value(self, net: rf.Network) -> float:
        """Скалярное значение критерия (до сравнения с порогом)."""
        freq, vals = self.selector(net)
        return self.agg(freq, vals)

    # ---------------------------------------------------------------- ok / penalty
    def is_ok(self, net: rf.Network) -> bool:
        return self.comp.is_ok(self.value(net))

    def penalty(self, net: rf.Network) -> float:
        raw = self.comp.penalty(self.value(net))
        return self.weight * raw

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return f"Criterion({self.name})"


# ────────────────────────────────────────────────────────────────────────────
#  __all__  – public export
# ────────────────────────────────────────────────────────────────────────────
__all__ = [
    # регистры
    "register_selector", "register_aggregator", "register_comparator",
    "get_selector",  # пока только селектор, других при необходимости
    # базы
    "BaseSelector", "BaseAggregator", "BaseComparator", "BaseCriterion",
]
