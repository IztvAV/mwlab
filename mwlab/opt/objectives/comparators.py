#mwlab/opt/objectives/comparators.py
"""
mwlab.opt.objectives.comparators
================================

Сравнение скалярного значения с порогом + расчёт penalty-cost.

БАЗОВЫЕ (жёсткие) КОМПАРАТОРЫ
-----------------------------
* LEComparator      – проверка `value ≤ limit` (penalty = 0/1)
* GEComparator      – проверка `value ≥ limit` (penalty = 0/1)

МЯГКИЕ (ГЛАДКИЕ/КУСОЧНО-ГЛАДКИЕ) КОМПАРАТОРЫ
--------------------------------------------
* HingeLEComparator / HingeGEComparator
    ReLU-подобный штраф: penalty ~ max(0, нарушение / scale)
    (кусочно-линейный «шарнир», удобно для быстрой настройки)

* SoftPlusLEComparator / SoftPlusGEComparator
    Гладкая версия ReLU: softplus(beta * x) / beta / scale
    (x — нормированное нарушение). Хорошо для глобальной оптимизации.

* HuberLEComparator / HuberGEComparator
    Квадратично рядом с порогом, линейно вдали (робастный компромисс).

СОВМЕСТИМОСТЬ
-------------
* Сохраняется существующий SoftLEComparator (квадратичный штраф за превышение
  `limit` вне буфера `margin`), чтобы не ломать существующий код. Для новых
  задач рекомендуется использовать SoftPlus* / Hinge* / Huber*.

МАСШТАБИРОВАНИЕ/НОРМАЛИЗАЦИЯ ШТРАФА
-----------------------------------
* Для базовых агрегаторов (Max/Min/Mean/…) удобно масштабировать штраф
  через параметр `scale` в компараторе. Это позволяет приводить разные
  цели к одному порядку величин без изменения агрегаторов.
* Для интегральных агрегаторов (UpIntAgg/LoIntAgg/RippleIntAgg) нормализация
  выполняется на уровне агрегатора, а компаратор обычно сравнивает с нулём
  (`limit=0`) и использует лишь мягкую форму штрафа.

ЗАЩИТА ОТ ДЕЛЕНИЯ НА НОЛЬ
-------------------------
* Во всех компараторах параметры `scale`, `delta`, `margin` валидируются
  как положительные числа (>0). При использовании `scale='auto'` внутри
  вычисляется положительное значение по простой политике.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .base import BaseComparator, register_comparator


# ────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ХЕЛПЕРЫ
# ────────────────────────────────────────────────────────────────────────────

def _softplus(x: np.ndarray | float, beta: float) -> np.ndarray | float:
    """
    Численно устойчивая softplus:
        softplus(z) = ln(1 + exp(z))
    Здесь используем форму с масштабом beta:
        softplus(beta * x) / beta
    """
    b = float(beta)
    if isinstance(x, np.ndarray):
        z = b * x
        # Для больших z softplus(z) ~ z (во избежание переполнений):
        out = np.where(z > 20.0, z, np.log1p(np.exp(z)))
        return out / b
    else:
        z = b * float(x)
        return (z if z > 20.0 else np.log1p(np.exp(z))) / b


def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return v


def _resolve_scale(limit: float, scale: Union[float, str]) -> float:
    """
    Преобразует параметр `scale` в положительное число.
    Поддерживается:
      * число > 0;
      * строка 'auto' – простая политика авто-масштаба.
    Политика 'auto':
      - если |limit| > 0 → scale = max(|limit|, 1e-6)
      - иначе → scale = 1.0
    """
    if isinstance(scale, (int, float)):
        return _require_positive("scale", float(scale))
    if isinstance(scale, str):
        if scale == "auto":
            base = abs(float(limit))
            return base if base > 0.0 else 1.0
        raise ValueError("scale must be a positive number or 'auto'")
    raise TypeError("scale must be float|int|'auto'")


# ────────────────────────────────────────────────────────────────────────────
# БАЗОВЫЕ (ЖЁСТКИЕ) КОМПАРАТОРЫ: ≤, ≥
# ────────────────────────────────────────────────────────────────────────────

@register_comparator(("le", "<="))
class LEComparator(BaseComparator):
    """
    Жёсткая проверка `value ≤ limit`. penalty по умолчанию бинарный (0/1)
    за счёт реализации в BaseComparator.penalty().
    """
    def __init__(self, limit: float, unit: str = ""):
        self.limit = float(limit)
        self.unit = unit

    def is_ok(self, value: float) -> bool:
        return value <= self.limit

    def __repr__(self):  # pragma: no cover
        return f"<= {self.limit} {self.unit}".strip()


@register_comparator(("ge", ">="))
class GEComparator(BaseComparator):
    """
    Жёсткая проверка `value ≥ limit`. penalty по умолчанию бинарный (0/1).
    """
    def __init__(self, limit: float, unit: str = ""):
        self.limit = float(limit)
        self.unit = unit

    def is_ok(self, value: float) -> bool:
        return value >= self.limit

    def __repr__(self):  # pragma: no cover
        return f">= {self.limit} {self.unit}".strip()


# ────────────────────────────────────────────────────────────────────────────
# СОВМЕСТИМОСТЬ: существующий мягкий компаратор для `≤`
# ────────────────────────────────────────────────────────────────────────────

@register_comparator("soft_le")
class SoftLEComparator(LEComparator):
    """
    Мягкое ограничение «≤ limit» с квадратичным штрафом вне буфера `margin`.

    penalty(value) =
        0,                           если value ≤ limit
        ((value - limit)/margin)^2,  если value > limit

    ПРИМЕЧАНИЕ: оставлен для совместимости со старым кодом. В новых
    сценариях чаще удобнее использовать Hinge/SoftPlus/Huber-семейства.
    """
    def __init__(self, limit: float, margin: float, power: int = 2, unit: str = ""):
        super().__init__(limit, unit)
        self.margin = _require_positive("margin", margin)
        self.power = int(power)

    def penalty(self, value: float) -> float:
        if self.is_ok(value):
            return 0.0
        return ((value - self.limit) / self.margin) ** self.power

    def __repr__(self):  # pragma: no cover
        return (f"soft ≤ {self.limit}±{self.margin} (pow={self.power}) {self.unit}").strip()


# ────────────────────────────────────────────────────────────────────────────
# НОВЫЕ МЯГКИЕ КОМПАРАТОРЫ
# ────────────────────────────────────────────────────────────────────────────
# 1) Hinge (ReLU-подобные)
# ---------------------------------------------------------------------------

@register_comparator("hinge_le")
class HingeLEComparator(LEComparator):
    """
    ReLU-подобный штраф для цели `value ≤ limit`:

        r = max(0, value - limit)
        penalty = r / scale

    Где `scale > 0` задаёт чувствительность (масштаб) штрафа.
    Удобен, когда нужна простая линейная зависимость без «ступеньки».
    """
    def __init__(self, limit: float, scale: Union[float, str] = 1.0, unit: str = ""):
        super().__init__(limit, unit)
        self._scale_param = scale

    def penalty(self, value: float) -> float:
        r = max(0.0, float(value) - self.limit)
        sc = _resolve_scale(self.limit, self._scale_param)
        return r / sc

    def __repr__(self):  # pragma: no cover
        return f"hinge ≤ {self.limit} / scale={self._scale_param} {self.unit}".strip()


@register_comparator("hinge_ge")
class HingeGEComparator(GEComparator):
    """
    ReLU-подобный штраф для цели `value ≥ limit`:

        r = max(0, limit - value)
        penalty = r / scale
    """
    def __init__(self, limit: float, scale: Union[float, str] = 1.0, unit: str = ""):
        super().__init__(limit, unit)
        self._scale_param = scale

    def penalty(self, value: float) -> float:
        r = max(0.0, self.limit - float(value))
        sc = _resolve_scale(self.limit, self._scale_param)
        return r / sc

    def __repr__(self):  # pragma: no cover
        return f"hinge ≥ {self.limit} / scale={self._scale_param} {self.unit}".strip()


# 2) SoftPlus (гладкие ReLU)
# ---------------------------------------------------------------------------

@register_comparator("softplus_le")
class SoftPlusLEComparator(LEComparator):
    """
    Гладкая версия ReLU для цели `value ≤ limit`:

        r = (value - limit) / scale
        penalty = softplus(beta * r) / beta

    Где:
      * `scale > 0` – масштаб нарушения (нормализация),
      * `beta > 0`  – «крутизна» аппроксимации (чем больше, тем ближе к ReLU).
    """
    def __init__(
        self,
        limit: float,
        scale: Union[float, str] = 1.0,
        beta: float = 8.0,
        unit: str = "",
    ):
        super().__init__(limit, unit)
        self._scale_param = scale
        self.beta = _require_positive("beta", beta)

    def penalty(self, value: float) -> float:
        sc = _resolve_scale(self.limit, self._scale_param)
        r = (float(value) - self.limit) / sc
        base = _softplus(0.0, self.beta)  # = ln(2)/beta
        return float(_softplus(r, self.beta) - base)

    def __repr__(self):  # pragma: no cover
        return f"softplus ≤ {self.limit} / scale={self._scale_param}, beta={self.beta} {self.unit}".strip()


@register_comparator("softplus_ge")
class SoftPlusGEComparator(GEComparator):
    """
    Гладкая версия ReLU для цели `value ≥ limit`:

        r = (limit - value) / scale
        penalty = softplus(beta * r) / beta
    """
    def __init__(
        self,
        limit: float,
        scale: Union[float, str] = 1.0,
        beta: float = 8.0,
        unit: str = "",
    ):
        super().__init__(limit, unit)
        self._scale_param = scale
        self.beta = _require_positive("beta", beta)

    def penalty(self, value: float) -> float:
        sc = _resolve_scale(self.limit, self._scale_param)
        r = (self.limit - float(value)) / sc
        base = _softplus(0.0, self.beta)
        return float(_softplus(r, self.beta) - base)

    def __repr__(self):  # pragma: no cover
        return f"softplus ≥ {self.limit} / scale={self._scale_param}, beta={self.beta} {self.unit}".strip()


# 3) Huber (квадратично рядом с порогом, линейно вдали)
# ---------------------------------------------------------------------------

@register_comparator("huber_le")
class HuberLEComparator(LEComparator):
    """
    Huber-штраф для цели `value ≤ limit`:

        r = max(0, value - limit)
        penalty =
            0.5 * (r/δ)^2,      если r ≤ δ
            (r/δ) - 0.5,        если r > δ

    где `δ = delta > 0` – ширина квадратичной зоны. Безразмерный штраф.
    """
    def __init__(self, limit: float, delta: float, unit: str = ""):
        super().__init__(limit, unit)
        self.delta = _require_positive("delta", delta)

    def penalty(self, value: float) -> float:
        r = max(0.0, float(value) - self.limit)
        t = r / self.delta
        return float(0.5 * t * t if r <= self.delta else t - 0.5)

    def __repr__(self):  # pragma: no cover
        return f"huber ≤ {self.limit} / delta={self.delta} {self.unit}".strip()


@register_comparator("huber_ge")
class HuberGEComparator(GEComparator):
    """
    Huber-штраф для цели `value ≥ limit`:

        r = max(0, limit - value)
        penalty =
            0.5 * (r/δ)^2,      если r ≤ δ
            (r/δ) - 0.5,        если r > δ
    """
    def __init__(self, limit: float, delta: float, unit: str = ""):
        super().__init__(limit, unit)
        self.delta = _require_positive("delta", delta)

    def penalty(self, value: float) -> float:
        r = max(0.0, self.limit - float(value))
        t = r / self.delta
        return float(0.5 * t * t if r <= self.delta else t - 0.5)

    def __repr__(self):  # pragma: no cover
        return f"huber ≥ {self.limit} / delta={self.delta} {self.unit}".strip()


# ────────────────────────────────────────────────────────────────────────────
# ЗАГЛУШКИ/ПРОЧЕЕ (можно реализовать позже при необходимости)
# ────────────────────────────────────────────────────────────────────────────

@register_comparator("window")
class WindowComparator(BaseComparator):
    """
    TODO: окно (коридор) [low, high] с мягкими/жёсткими бортами.
    Пример интерфейса (на будущее):
        WindowComparator(low, high, mode='hard'|'soft', margin=..., unit='')
    """
    def __init__(self, *args, **kw):
        ...
    def is_ok(self, value: float) -> bool:  # pragma: no cover
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────────────
# ПУБЛИЧНЫЙ ЭКСПОРТ
# ────────────────────────────────────────────────────────────────────────────

__all__ = [
    # базовые
    "LEComparator", "GEComparator",
    # совместимость
    "SoftLEComparator",
    # новые мягкие
    "HingeLEComparator", "HingeGEComparator",
    "SoftPlusLEComparator", "SoftPlusGEComparator",
    "HuberLEComparator", "HuberGEComparator",
    # заглушка
    "WindowComparator",
]
