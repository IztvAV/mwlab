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

Пример:
crit = BaseCriterion(
    selector   = SMagSelector(1,1, band=(2e9,2.4e9), db=True),
    aggregator = MaxAgg(),
    comparator = LEComparator(limit=-22, unit="dB"),
    weight = 10,
    name="S11_inband"
)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type, Callable, Sequence, Tuple, List

import numpy as np
import skrf as rf


# ────────────────────────────────────────────────────────────────────────────
#  Registry helpers ― чтобы @register_selector(...) было максимально похоже
#  на существующие регистры (samplers, surrogates и т. д.)
# ────────────────────────────────────────────────────────────────────────────
_SELECTOR_REG: Dict[str, "BaseSelector"] = {}
_AGGREGATOR_REG: Dict[str, "BaseAggregator"] = {}
_COMPARATOR_REG: Dict[str, "BaseComparator"] = {}


def _make_register(reg: Dict[str, Type], kind: str) -> Callable[[str], Callable]:
    """Фабрика декораторов `register_selector / aggregator / comparator`."""
    def _decor(alias: str | Sequence[str]):
        aliases: List[str] = [alias] if isinstance(alias, str) else list(alias)

        def _wrap(cls):
            for name in aliases:
                if name in reg:
                    raise KeyError(f"{kind.capitalize()} alias '{name}' already exists")
                reg[name] = cls
            cls._aliases = aliases      # type: ignore[attr-defined]
            return cls
        return _wrap
    return _decor


register_selector   = _make_register(_SELECTOR_REG,   "selector")
register_aggregator = _make_register(_AGGREGATOR_REG, "aggregator")
register_comparator = _make_register(_COMPARATOR_REG, "comparator")


def get_selector(alias: str, **kw):
    if alias not in _SELECTOR_REG:
        raise KeyError(f"Selector '{alias}' not found. Available: {list(_SELECTOR_REG)}")
    return _SELECTOR_REG[alias](**kw)


# аналогичные фабрики можно добавить при необходимости
# get_aggregator / get_comparator


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
