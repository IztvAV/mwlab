#mwlab/opt/objectives/comparators.py
"""
comparators.py
==============
Сравнение скалярного значения с порогом + расчет penalty-cost.

* LEComparator      – value ≤ limit (жесткое, penalty = 0/1)
* GEComparator      – value ≥ limit
* SoftLEComparator  – квадратичный штраф за превышение
                      limit (+margin)  →  ((v-limit)/margin)^power
"""
from __future__ import annotations

from .base import BaseComparator, register_comparator


# ────────────────────────────────────────────────────────────────────────────
@register_comparator(("le", "<="))
class LEComparator(BaseComparator):
    def __init__(self, limit: float, unit: str = ""):
        self.limit = float(limit)
        self.unit = unit

    def is_ok(self, value: float) -> bool:
        return value <= self.limit

    def __repr__(self):  # pragma: no cover
        return f"<= {self.limit} {self.unit}"


# ---------------------------------------------------------------- ≥
@register_comparator(("ge", ">="))
class GEComparator(BaseComparator):
    def __init__(self, limit: float, unit: str = ""):
        self.limit = float(limit)
        self.unit = unit

    def is_ok(self, value: float) -> bool:
        return value >= self.limit

    def __repr__(self):  # pragma: no cover
        return f">= {self.limit} {self.unit}"


# ---------------------------------------------------------------- soft-penalty
@register_comparator("soft_le")
class SoftLEComparator(LEComparator):
    """
    Мягкое ограничение «≤ limit» с квадратичным штрафом
    вне буфера `margin`.

    penalty = 0, если v ≤ limit
    penalty = ((v - limit)/margin)^power, иначе
    """

    def __init__(self, limit: float, margin: float, power: int = 2, unit: str = ""):
        super().__init__(limit, unit)
        if margin <= 0:
            raise ValueError("margin must be > 0")
        self.margin = float(margin)
        self.power = int(power)

    # override
    def penalty(self, value: float) -> float:
        if self.is_ok(value):
            return 0.0
        return ((value - self.limit) / self.margin) ** self.power

    def __repr__(self):  # pragma: no cover
        return (f"soft ≤ {self.limit}±{self.margin} "
                f"(pow={self.power}) {self.unit}")


# ---------------------------------------------------------------- TODO-заглушка
@register_comparator("window")
class WindowComparator(BaseComparator):
    def __init__(self, *args, **kw): ...
    def is_ok(self, value): raise NotImplementedError
