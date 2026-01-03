# mwlab/opt/objectives/comparators.py
"""
mwlab.opt.objectives.comparators
================================

Comparator отвечает на вопрос: «как интерпретировать скаляр, полученный от Aggregator?».

Назначение
----------
Компаратор получает одно число (float) и предоставляет две операции:

1) is_ok(value) -> bool
   Проверка выполнения требования (ограничения).

2) penalty(value) -> float >= 0
   Штраф (мера “плохости”), удобная для оптимизации.
   Важно: penalty **всегда неотрицателен**.

Comparator не видит частотную зависимость
-----------------------------------------
Comparator работает только со скаляром. Все операции над кривыми (обрезка полос,
сглаживание, производные/крутизна, расчёт ГВЗ, интегралы нарушений и т.п.)
должны выполняться до компаратора: Selector / Transform / Aggregator.

Политика обработки NaN/Inf
--------------------------
В оптимизационных пайплайнах NaN/Inf могут появляться после дифференцирования,
ресэмплинга, апертурных операций и т.п. Для предсказуемого поведения компараторы
поддерживают единый параметр finite_policy:

- "fail"  : считать требование невыполненным и вернуть non_finite_penalty
- "ok"    : считать значение проходным и вернуть 0
- "raise" : выбросить исключение (удобно для отладки)

non_finite_penalty всегда приводится к неотрицательному значению.

Мягкие штрафы и масштабирование
------------------------------
Для “мягких” одно-сторонних штрафов (hinge/softplus/power/huber) обычно полезно
ввести масштаб (scale/margin/delta), который задаёт характерную величину нарушения.
Это позволяет получать штрафы сопоставимого порядка при разных единицах измерения.

Параметр scale="auto"
---------------------
Для некоторых компараторов допускается scale="auto". Тогда масштаб выбирается
по простой эвристике:

- для ограничений <= и >= : max(|limit|, AUTO_EPS)
- для окна [low, high]    : max(high - low, AUTO_EPS)

AUTO_EPS нужен, чтобы избегать деления на ноль.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Literal

import numpy as np

from .registry import register_comparator
from .base import BaseComparator


# =============================================================================
# Типы и константы
# =============================================================================

FinitePolicy = Literal["fail", "raise", "ok"]
PenaltyKindOneSided = Literal["hard", "hinge", "softplus", "huber", "power"]

# Нижняя граница масштаба для scale="auto"
_AUTO_EPS = 1e-6


# =============================================================================
# Внутренние хелперы: проверки и функции штрафов
# =============================================================================

def _is_finite_scalar(x: float) -> bool:
    """True, если число конечное (не NaN и не Inf)."""
    return bool(np.isfinite(float(x)))


def _norm_finite_policy(policy: str) -> FinitePolicy:
    """Нормализовать finite_policy и проверить допустимость."""
    p = str(policy).strip().lower()
    if p not in ("fail", "raise", "ok"):
        raise ValueError("finite_policy должен быть 'fail'|'raise'|'ok'")
    return p  # type: ignore[return-value]


def _nonneg(x: float) -> float:
    """Обрезать число снизу к 0 (для штрафов)."""
    v = float(x)
    return v if v >= 0.0 else 0.0


def _require_positive(name: str, value: float) -> float:
    """Проверить, что параметр строго положительный."""
    v = float(value)
    if v <= 0.0:
        raise ValueError(f"{name} должен быть > 0")
    return v


def _handle_non_finite(
    *,
    finite_policy: FinitePolicy,
    non_finite_penalty: float,
) -> Tuple[bool, float]:
    """
    Единая политика обработки NaN/Inf.

    Возвращает (ok, penalty):
      - fail  -> (False, non_finite_penalty)
      - ok    -> (True,  0.0)
      - raise -> исключение
    """
    pol = _norm_finite_policy(finite_policy)

    if pol == "fail":
        return False, _nonneg(non_finite_penalty)

    if pol == "ok":
        return True, 0.0

    # pol == "raise"
    raise ValueError("Comparator получил невалидное значение (NaN/Inf)")


def _resolve_scale_for_limit(limit: float, scale: Union[float, str]) -> float:
    """
    Преобразовать scale в положительное число.

    scale:
      - положительное число
      - "auto" -> max(|limit|, _AUTO_EPS)
    """
    if isinstance(scale, (int, float)):
        return _require_positive("scale", float(scale))

    if isinstance(scale, str):
        s = scale.strip().lower()
        if s == "auto":
            return max(abs(float(limit)), _AUTO_EPS)
        raise ValueError("scale должен быть положительным числом или 'auto'")

    raise TypeError("scale должен быть числом (float/int) или строкой 'auto'")


def _resolve_scale_for_window(low: float, high: float, scale: Union[float, str]) -> float:
    """
    Масштаб для окна [low, high].

    scale:
      - положительное число
      - "auto" -> max(high - low, _AUTO_EPS)
    """
    if isinstance(scale, (int, float)):
        return _require_positive("scale", float(scale))

    if isinstance(scale, str):
        s = scale.strip().lower()
        if s == "auto":
            return max(float(high - low), _AUTO_EPS)
        raise ValueError("scale должен быть положительным числом или 'auto'")

    raise TypeError("scale должен быть числом (float/int) или строкой 'auto'")


def _softplus(x: float, beta: float) -> float:
    """
    Численно устойчивая softplus.

    Возвращает величину:
        softplus(beta*x)/beta = ln(1 + exp(beta*x))/beta

    Параметр beta управляет “жёсткостью” приближения hinge:
      beta -> +inf  => softplus ~ max(0, x)
    """
    b = _require_positive("beta", beta)
    z = b * float(x)

    # Для больших z: ln(1+exp(z)) ≈ z, чтобы избежать переполнения exp(z).
    if z > 40.0:
        return z / b
    return float(np.log1p(np.exp(z)) / b)


def _penalty_one_sided(
    residual: float,
    *,
    kind: PenaltyKindOneSided,
    scale: float = 1.0,
    beta: float = 8.0,
    delta: float = 1.0,
    margin: float = 1.0,
    power: int = 2,
) -> float:
    """
    Унифицированный штраф для одно-стороннего ограничения residual <= 0.

    residual:
      residual <= 0  -> норма (penalty=0)
      residual >  0  -> нарушение

    kind:
      - "hard"     : 0 или 1
      - "hinge"    : max(0,residual)/scale
      - "softplus" : softplus(max(0,residual)/scale) - softplus(0)
      - "huber"    : Huber-подобный штраф по положительной части residual
      - "power"    : (max(0,residual)/margin)^power
    """
    r = max(0.0, float(residual))

    if kind == "hard":
        return 0.0 if r == 0.0 else 1.0

    if kind == "hinge":
        sc = _require_positive("scale", scale)
        return r / sc

    if kind == "softplus":
        sc = _require_positive("scale", scale)
        x = r / sc
        base = _softplus(0.0, beta)  # чтобы penalty(0)=0
        return max(0.0, _softplus(x, beta) - base)

    if kind == "huber":
        # Нормированная версия: при малых r ~ 0.5*(r/delta)^2, при больших ~ (r/delta) - 0.5
        d = _require_positive("delta", delta)
        t = r / d
        return float(0.5 * t * t if r <= d else t - 0.5)

    if kind == "power":
        m = _require_positive("margin", margin)
        p = int(power)
        if p <= 0:
            raise ValueError("power должен быть >= 1")
        return float((r / m) ** p)

    raise ValueError("Неизвестный тип одностороннего штрафа (kind)")


# =============================================================================
# Жёсткие компараторы: <= и >=
# =============================================================================

@register_comparator(("LEComparator", "le", "<="))
class LEComparator(BaseComparator):
    """
    Жёсткое ограничение: value <= limit.

    penalty:
      0.0 если value <= limit
      1.0 иначе

    finite_policy:
      - fail  -> ok=False, штраф non_finite_penalty
      - ok    -> ok=True,  штраф 0
      - raise -> исключение
    """

    def __init__(
        self,
        limit: float,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1.0,
    ):
        self.limit = float(limit)
        self.unit = str(unit)

        self.finite_policy: FinitePolicy = _norm_finite_policy(finite_policy)
        self.non_finite_penalty = _nonneg(non_finite_penalty)

    def is_ok(self, value: float) -> bool:
        v = float(value)
        if not _is_finite_scalar(v):
            ok, _ = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return ok
        return v <= self.limit

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        return 0.0 if v <= self.limit else 1.0


@register_comparator(("GEComparator", "ge", ">="))
class GEComparator(BaseComparator):
    """
    Жёсткое ограничение: value >= limit.

    penalty:
      0.0 если value >= limit
      1.0 иначе
    """

    def __init__(
        self,
        limit: float,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1.0,
    ):
        self.limit = float(limit)
        self.unit = str(unit)

        self.finite_policy: FinitePolicy = _norm_finite_policy(finite_policy)
        self.non_finite_penalty = _nonneg(non_finite_penalty)

    def is_ok(self, value: float) -> bool:
        v = float(value)
        if not _is_finite_scalar(v):
            ok, _ = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return ok
        return v >= self.limit

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        return 0.0 if v >= self.limit else 1.0


# =============================================================================
# Мягкие одно-сторонние компараторы (степенной / hinge / softplus / huber)
# =============================================================================

@register_comparator(("SoftLEComparator", "soft_le", "power_le"))
class SoftLEComparator(LEComparator):
    """
    Мягкое ограничение value <= limit со степенным штрафом:

      penalty = 0, если value <= limit
      penalty = ((value - limit)/margin)^power, если value > limit

    margin задаёт характерную шкалу нарушения.
    """

    def __init__(
        self,
        limit: float,
        margin: float,
        power: int = 2,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.margin = _require_positive("margin", margin)
        self.power = int(power)
        if self.power <= 0:
            raise ValueError("power должен быть >= 1")

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        return _penalty_one_sided(v - self.limit, kind="power", margin=self.margin, power=self.power)


@register_comparator(("SoftGEComparator", "soft_ge", "power_ge"))
class SoftGEComparator(GEComparator):
    """
    Мягкое ограничение value >= limit со степенным штрафом:

      penalty = 0, если value >= limit
      penalty = ((limit - value)/margin)^power, если value < limit
    """

    def __init__(
        self,
        limit: float,
        margin: float,
        power: int = 2,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.margin = _require_positive("margin", margin)
        self.power = int(power)
        if self.power <= 0:
            raise ValueError("power должен быть >= 1")

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        return _penalty_one_sided(self.limit - v, kind="power", margin=self.margin, power=self.power)


@register_comparator(("HingeLEComparator", "hinge_le"))
class HingeLEComparator(LEComparator):
    """
    Hinge (ReLU) штраф для value <= limit:

      penalty = max(0, value - limit)/scale

    scale:
      - положительное число
      - "auto" -> max(|limit|, AUTO_EPS)
    """

    def __init__(
        self,
        limit: float,
        scale: Union[float, str] = "auto",
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.scale = scale

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        sc = _resolve_scale_for_limit(self.limit, self.scale)
        return _penalty_one_sided(v - self.limit, kind="hinge", scale=sc)


@register_comparator(("HingeGEComparator", "hinge_ge"))
class HingeGEComparator(GEComparator):
    """
    Hinge (ReLU) штраф для value >= limit:

      penalty = max(0, limit - value)/scale
    """

    def __init__(
        self,
        limit: float,
        scale: Union[float, str] = "auto",
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.scale = scale

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        sc = _resolve_scale_for_limit(self.limit, self.scale)
        return _penalty_one_sided(self.limit - v, kind="hinge", scale=sc)


@register_comparator(("SoftPlusLEComparator", "softplus_le"))
class SoftPlusLEComparator(LEComparator):
    """
    Гладкий hinge (softplus) для value <= limit:

      penalty = softplus( max(0, (value - limit)/scale) ) - softplus(0)

    beta регулирует “крутизну” (чем больше beta, тем ближе к hinge).
    """

    def __init__(
        self,
        limit: float,
        scale: Union[float, str] = "auto",
        beta: float = 8.0,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.scale = scale
        self.beta = _require_positive("beta", beta)

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        sc = _resolve_scale_for_limit(self.limit, self.scale)
        return _penalty_one_sided(
            v - self.limit,
            kind="softplus",
            scale=sc,
            beta=self.beta,
        )


@register_comparator(("SoftPlusGEComparator", "softplus_ge"))
class SoftPlusGEComparator(GEComparator):
    """
    Гладкий hinge (softplus) для value >= limit:

      penalty = softplus( max(0, (limit - value)/scale) ) - softplus(0)
    """

    def __init__(
        self,
        limit: float,
        scale: Union[float, str] = "auto",
        beta: float = 8.0,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.scale = scale
        self.beta = _require_positive("beta", beta)

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p

        sc = _resolve_scale_for_limit(self.limit, self.scale)
        return _penalty_one_sided(
            self.limit - v,
            kind="softplus",
            scale=sc,
            beta=self.beta,
        )


@register_comparator(("HuberLEComparator", "huber_le"))
class HuberLEComparator(LEComparator):
    """
    Huber-штраф для value <= limit по положительной части (value - limit).

    delta задаётся в единицах value и определяет границу “квадратичного” участка.
    """

    def __init__(
        self,
        limit: float,
        delta: float,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.delta = _require_positive("delta", delta)

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        return _penalty_one_sided(v - self.limit, kind="huber", delta=self.delta)


@register_comparator(("HuberGEComparator", "huber_ge"))
class HuberGEComparator(GEComparator):
    """
    Huber-штраф для value >= limit по положительной части (limit - value).
    """

    def __init__(
        self,
        limit: float,
        delta: float,
        unit: str = "",
        *,
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        super().__init__(
            limit=limit,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )
        self.delta = _require_positive("delta", delta)

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return p
        return _penalty_one_sided(self.limit - v, kind="huber", delta=self.delta)


# =============================================================================
# Окно (коридор) low..high и алиас "target ± tol"
# =============================================================================

@register_comparator(("WindowComparator", "window", "corridor"))
class WindowComparator(BaseComparator):
    """
    Коридор (окно): low <= value <= high.

    mode:
      - "hard"     : 0/1
      - "hinge"    : hinge(low - v) + hinge(v - high)
      - "softplus" : softplus(low - v) + softplus(v - high)
      - "huber"    : huber(low - v) + huber(v - high)

    scale (для hinge/softplus):
      - положительное число
      - "auto" -> max(high-low, AUTO_EPS)

    delta (для huber):
      - положительное число в единицах value
    """

    def __init__(
        self,
        low: float,
        high: float,
        *,
        mode: Literal["hard", "hinge", "softplus", "huber"] = "hard",
        scale: Union[float, str] = "auto",
        beta: float = 8.0,
        delta: float = 1.0,
        unit: str = "",
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        lo = float(low)
        hi = float(high)
        if hi < lo:
            lo, hi = hi, lo

        self.low = lo
        self.high = hi

        self.mode = str(mode).strip().lower()  # type: ignore[assignment]
        if self.mode not in ("hard", "hinge", "softplus", "huber"):
            raise ValueError("mode должен быть 'hard'|'hinge'|'softplus'|'huber'")

        self.scale = scale
        self.beta = _require_positive("beta", beta)
        self.delta = _require_positive("delta", delta)

        self.unit = str(unit)
        self.finite_policy: FinitePolicy = _norm_finite_policy(finite_policy)
        self.non_finite_penalty = _nonneg(non_finite_penalty)

    def is_ok(self, value: float) -> bool:
        v = float(value)
        if not _is_finite_scalar(v):
            ok, _ = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=self.non_finite_penalty,
            )
            return ok
        return (self.low <= v) and (v <= self.high)

    def penalty(self, value: float) -> float:
        v = float(value)
        if not _is_finite_scalar(v):
            # Для hard-режима удобно возвращать “единичный” штраф,
            # для мягких режимов — большой non_finite_penalty.
            default = 1.0 if self.mode == "hard" else self.non_finite_penalty
            _, p = _handle_non_finite(
                finite_policy=self.finite_policy,
                non_finite_penalty=default,
            )
            return p

        if self.mode == "hard":
            return 0.0 if (self.low <= v <= self.high) else 1.0

        left_res = self.low - v    # хотим <= 0
        right_res = v - self.high  # хотим <= 0

        if self.mode in ("hinge", "softplus"):
            sc = _resolve_scale_for_window(self.low, self.high, self.scale)
            if self.mode == "hinge":
                p_left = _penalty_one_sided(left_res, kind="hinge", scale=sc)
                p_right = _penalty_one_sided(right_res, kind="hinge", scale=sc)
                return float(p_left + p_right)

            # softplus
            p_left = _penalty_one_sided(left_res, kind="softplus", scale=sc, beta=self.beta)
            p_right = _penalty_one_sided(right_res, kind="softplus", scale=sc, beta=self.beta)
            return float(p_left + p_right)

        # huber
        p_left = _penalty_one_sided(left_res, kind="huber", delta=self.delta)
        p_right = _penalty_one_sided(right_res, kind="huber", delta=self.delta)
        return float(p_left + p_right)


@register_comparator(("TargetComparator", "target"))
class TargetComparator(WindowComparator):
    """
    Удержание значения около target с допуском ±tol.

    Это удобный алиас:
      WindowComparator(low=target-tol, high=target+tol)
    """

    def __init__(
        self,
        target: float,
        tol: float,
        *,
        mode: Literal["hard", "hinge", "softplus", "huber"] = "hard",
        scale: Union[float, str] = "auto",
        beta: float = 8.0,
        delta: float = 1.0,
        unit: str = "",
        finite_policy: FinitePolicy = "fail",
        non_finite_penalty: float = 1e6,
    ):
        t = float(target)
        w = _require_positive("tol", tol)
        # Важно для serde: TargetComparator.__init__ имеет параметры target/tol,
        # значит объект обязан их хранить, иначе BaseComparator.serde_params() упадёт.
        self.target = t
        self.tol = w
        super().__init__(
            low=t - w,
            high=t + w,
            mode=mode,
            scale=scale,
            beta=beta,
            delta=delta,
            unit=unit,
            finite_policy=finite_policy,
            non_finite_penalty=non_finite_penalty,
        )


# =============================================================================
# Публичный экспорт
# =============================================================================

__all__ = [
    # hard
    "LEComparator",
    "GEComparator",
    # power
    "SoftLEComparator",
    "SoftGEComparator",
    # hinge
    "HingeLEComparator",
    "HingeGEComparator",
    # softplus
    "SoftPlusLEComparator",
    "SoftPlusGEComparator",
    # huber
    "HuberLEComparator",
    "HuberGEComparator",
    # window / target
    "WindowComparator",
    "TargetComparator",
]
