"""
mwlab.opt.objectives.aggregators
================================

«Как» из 1-D вектора сделать скаляр.

БАЗОВЫЕ АГРЕГАТОРЫ
------------------
* MaxAgg      – максимум;
* MinAgg      – минимум;
* MeanAgg     – среднее;
* RippleAgg   – размах (max - min);
* StdAgg      – стандартное отклонение.

ИНТЕГРАЛЬНЫЕ АГРЕГАТОРЫ (для гладких целевых функций)
-----------------------------------------------------
* UpIntAgg     – интеграл (или среднее) ВЕРХНИХ нарушений для целей вида `value ≤ limit`;
* LoIntAgg     – интеграл (или среднее) НИЖНИХ нарушений для целей вида `value ≥ limit`;
* RippleIntAgg – интегральная «неравномерность» относительно опорной линии.

ЗНАКОВЫЕ ИНТЕГРАТОРЫ (минималистичная и строгая формулировка)
-------------------------------------------------------------
* SignedUpIntAgg     – для целей `value ≤ limit(f)`;
* SignedLoIntAgg     – для целей `value ≥ limit(f)`;
* SignedRippleIntAgg – для «рябь ≤ deadzone» относительно опорной линии.

Ключевые идеи знаковых интеграторов (новая версия):
    1) Вводим остаток r(f) так, чтобы цель была r(f) ≤ 0.
       - Up   : r = v - L(f)
       - Lo   : r = L(f) - v
       - Ripple: r = |v - T(f)| - deadzone
       Тогда нарушения: [r]_+ = max(r, 0), запас: [r]_- = max(-r, 0).

    2) Положительная часть A_plus строится как раньше:
       mean/trapz от [r]_+^p и делится на общий нормировочный множитель Z>0.

    3) «Награда» за запас считается по худшему запасу, но гладко:
       M_τ = log-mean-exp (LME) от вектора s=[r]_-, с автоподбором τ.
       Это даёт корректное M_τ=0, когда запасов нет, и гладкость вместо max.

    4) Жёсткий выключатель награды (gate): если A_plus > eps → награда = 0.
       (eps очень мал — чтобы не было награды при наличии нарушений.)

    5) Затухание награды одной фиксированной формой:
       Reward = S * exp(-S/ρ), где S = raw_minus/Z — нормированный размер «худшего запаса».
       Такой вид даёт естественную «сладкую точку» при S≈ρ и затухание при чрезмерных запасах,
       чтобы один критерий не «перетягивал одеяло».

Параметры, оставленные пользователю для знаковых интеграторов:
    - p: {1,2} (по умолчанию 2),
    - method: {'mean','trapz'},
    - normalize: см. ниже,
    - rho: float>0 — «масштаб разумного запаса» для затухания exp(-S/rho).

Остальные детали (порог gate, τ для LME, устойчивость интегрирования) — внутри.

"""

from __future__ import annotations

from typing import Callable, Union, Tuple, Sequence

import numpy as np

from .base import BaseAggregator, register_aggregator


# ────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ТИПЫ И ХЕЛПЕРЫ
# ────────────────────────────────────────────────────────────────────────────

# Тип для limit/target: константа или функция от частоты (в ГГц) → массив
LimitFunc = Union[float, Callable[[np.ndarray], np.ndarray]]


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Безопасная обёртка над трапецией:
    * в новых NumPy используем np.trapezoid;
    * fallback на np.trapz для совместимости со старыми версиями.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    # крайний случай: численное интегрирование вручную
    x = x.astype(float)
    y = y.astype(float)
    if x.size < 2:
        return 0.0
    dx = np.diff(x)
    ym = 0.5 * (y[1:] + y[:-1])
    return float(np.sum(dx * ym))


def _as_limit_fn(limit: LimitFunc) -> Callable[[np.ndarray], np.ndarray]:
    """
    Приводит переданный limit (константа или функция) к вызываемой форме:
    f_GHz: ndarray → limit(f_GHz): ndarray
    """
    if callable(limit):
        return limit
    v = float(limit)
    return lambda f: np.full_like(f, v, dtype=float)


def _norm_factor(
    normalize: Union[str, float, Sequence[Union[str, float]]],
    f_ghz: np.ndarray,
    ref_line: np.ndarray,
) -> float:
    """
    Вычисляет множитель нормализации. Возвращает ВСЕГДА > 0.

    normalize:
      - 'none'                → 1.0
      - 'bandwidth'|'bw'      → ширина диапазона (f[-1] - f[0]) или 1.0
      - 'limit'               → среднее |ref_line| по диапазону или 1.0
      - 'bandwidth*limit'     → произведение двух нормировок
      - число (float)         → явная шкала
      - последовательность    → композиция нормировок (перемножение факторов)
    """
    EPS = 1e-12

    def _one(item) -> float:
        if isinstance(item, (int, float)):
            return max(float(item), EPS)
        if item in ("bandwidth", "bw"):
            if f_ghz.size >= 2:
                bw = float(f_ghz[-1] - f_ghz[0])
                return max(bw, EPS)
            return 1.0
        if item == "limit":
            s = float(np.mean(np.abs(ref_line))) if ref_line.size else 0.0
            return max(s, 1.0) if s > 0.0 else 1.0
        if item in ("bandwidth*limit", "bw*limit"):
            return _one("bandwidth") * _one("limit")
        # 'none' или неизвестное значение → без нормализации
        return 1.0

    if isinstance(normalize, (list, tuple)):
        factor = 1.0
        for it in normalize:
            factor *= _one(it)
        return max(factor, EPS)

    return _one(normalize)


def _ensure_non_empty(vals: np.ndarray, who: str):
    """Единообразная диагностика пустых массивов для базовых агрегаторов."""
    if vals.size == 0:
        raise ValueError(f"{who}: пустой массив значений (диапазон не попал в сетку)")


def _bandwidth(f_ghz: np.ndarray) -> float:
    """Ширина диапазона частот (в ГГц). Если точек < 2, возвращает 1.0 как безопасный масштаб."""
    if f_ghz.size >= 2:
        return float(f_ghz[-1] - f_ghz[0])
    return 1.0


# ────────────────────────────────────────────────────────────────────────────
# БАЗОВЫЕ ПРОСТЫЕ АГРЕГАТОРЫ
# ────────────────────────────────────────────────────────────────────────────

@register_aggregator("max")
class MaxAgg(BaseAggregator):
    """Возвращает максимум значений."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        _ensure_non_empty(vals, "MaxAgg")
        return float(np.max(vals))


@register_aggregator("min")
class MinAgg(BaseAggregator):
    """Возвращает минимум значений."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        _ensure_non_empty(vals, "MinAgg")
        return float(np.min(vals))


@register_aggregator("mean")
class MeanAgg(BaseAggregator):
    """Возвращает среднее значений."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        _ensure_non_empty(vals, "MeanAgg")
        return float(np.mean(vals))


@register_aggregator("ripple")
class RippleAgg(BaseAggregator):
    """Разброс (max - min)."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        _ensure_non_empty(vals, "RippleAgg")
        return float(np.max(vals) - np.min(vals))


@register_aggregator("std")
class StdAgg(BaseAggregator):
    """Стандартное отклонение (population std)."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        _ensure_non_empty(vals, "StdAgg")
        return float(np.std(vals))


# ────────────────────────────────────────────────────────────────────────────
# ИНТЕГРАЛЬНЫЕ АГРЕГАТОРЫ ДЛЯ ГЛАДКИХ ЦЕЛЕЙ (БЕЗ ЗНАКА)
# ────────────────────────────────────────────────────────────────────────────

@register_aggregator(("upint", "UpIntAgg"))
class UpIntAgg(BaseAggregator):
    """
    Интегральная мера ВЕРХНИХ нарушений для целей вида `value ≤ limit`.

    Формально:
        lim(f) = константа или функция от частоты (в ГГц)
        viol = clip(vals - lim(f), 0, +inf)
        acc  = mean(viol**p)   или   trapz(viol**p, f)
        out  = acc / norm_factor
    """
    def __init__(
        self,
        limit: LimitFunc,
        p: int = 2,
        method: str = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        on_empty: str = "raise",
    ):
        self.limit = limit
        self.p = int(p)
        self.method = str(method)
        self.normalize = normalize
        self.on_empty = str(on_empty)

    def __call__(self, freq_ghz: np.ndarray, vals: np.ndarray) -> float:
        if vals.size == 0:
            if self.on_empty == "ok":
                return 0.0
            raise ValueError("UpIntAgg: пустой массив значений (диапазон не попал в сетку)")

        f = freq_ghz.astype(float)
        lim = _as_limit_fn(self.limit)(f)  # shape = (N,)
        viol = np.clip(vals - lim, 0.0, None)  # верхние нарушения
        v = viol ** self.p

        if self.method == "trapz":
            raw = _trapz(v, f)
        elif self.method == "mean":
            raw = float(np.mean(v))
        else:
            raise ValueError("UpIntAgg.method должен быть 'mean' или 'trapz'")

        scale = _norm_factor(self.normalize, f, lim)
        return raw / scale


@register_aggregator(("loint", "LoIntAgg"))
class LoIntAgg(BaseAggregator):
    """
    Интегральная мера НИЖНИХ нарушений для целей вида `value ≥ limit`.

    Формально:
        lim(f) = константа или функция от частоты (в ГГц)
        viol = clip(lim(f) - vals, 0, +inf)
        acc  = mean(viol**p)   или   trapz(viol**p, f)
        out  = acc / norm_factor
    """
    def __init__(
        self,
        limit: LimitFunc,
        p: int = 2,
        method: str = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        on_empty: str = "raise",
    ):
        self.limit = limit
        self.p = int(p)
        self.method = str(method)
        self.normalize = normalize
        self.on_empty = str(on_empty)

    def __call__(self, freq_ghz: np.ndarray, vals: np.ndarray) -> float:
        if vals.size == 0:
            if self.on_empty == "ok":
                return 0.0
            raise ValueError("LoIntAgg: пустой массив значений (диапазон не попал в сетку)")

        f = freq_ghz.astype(float)
        lim = _as_limit_fn(self.limit)(f)
        viol = np.clip(lim - vals, 0.0, None)  # нижние нарушения
        v = viol ** self.p

        if self.method == "trapz":
            raw = _trapz(v, f)
        elif self.method == "mean":
            raw = float(np.mean(v))
        else:
            raise ValueError("LoIntAgg.method должен быть 'mean' или 'trapz'")

        scale = _norm_factor(self.normalize, f, lim)
        return raw / scale


@register_aggregator(("rippleint", "RippleIntAgg"))
class RippleIntAgg(BaseAggregator):
    """
    Интегральная «неравномерность» относительно опорной линии.

    Идея: измеряем отклонения |vals - target(f)|,
    возможно, игнорируя малые колебания через `deadzone`,
    затем накапливаем (L¹/L²) и нормируем.

    Формально:
        tgt(f) = опорная линия (константа, 'mean', 'median', 'linear' или функция)
        dev_raw = |vals - tgt(f)|
        dev = clip(dev_raw - deadzone, 0, +inf)
        acc = mean(dev**p) или trapz(dev**p, f)
        out = acc / norm_factor
    """
    def __init__(
        self,
        target: Union[str, float, Callable[[np.ndarray], np.ndarray]] = "mean",
        deadzone: float = 0.0,
        p: int = 2,
        method: str = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        on_empty: str = "raise",
    ):
        self.target = target
        self.deadzone = float(deadzone)
        self.p = int(p)
        self.method = str(method)
        self.normalize = normalize
        self.on_empty = str(on_empty)

    def __call__(self, freq_ghz: np.ndarray, vals: np.ndarray) -> float:
        if vals.size == 0:
            if self.on_empty == "ok":
                return 0.0
            raise ValueError("RippleIntAgg: пустой массив значений (диапазон не попал в сетку)")

        f = freq_ghz.astype(float)

        # 1) строим опорную линию tgt(f)
        if callable(self.target):
            tgt = self.target(f)
        elif isinstance(self.target, (int, float)):
            tgt = np.full_like(vals, float(self.target), dtype=float)
        elif self.target == "mean":
            tgt = np.full_like(vals, float(np.mean(vals)), dtype=float)
        elif self.target == "median":
            tgt = np.full_like(vals, float(np.median(vals)), dtype=float)
        elif self.target == "linear":
            # МНК-приближение прямой: vals ≈ a + b f
            A = np.vstack([np.ones_like(f), f]).T
            coef, *_ = np.linalg.lstsq(A, vals, rcond=None)  # coef = [a, b]
            tgt = (coef[0] + coef[1] * f).astype(float)
        else:
            raise ValueError("RippleIntAgg.target должен быть {'mean','median','linear'} | float | callable")

        # 2) модуль отклонений и «мёртвая зона»
        dev = np.abs(vals - tgt)
        if self.deadzone > 0.0:
            dev = np.clip(dev - self.deadzone, 0.0, None)

        v = dev ** self.p

        # 3) свёртка по частоте
        if self.method == "trapz":
            raw = _trapz(v, f)
        elif self.method == "mean":
            raw = float(np.mean(v))
        else:
            raise ValueError("RippleIntAgg.method должен быть 'mean' или 'trapz'")

        # 4) нормализация (для 'limit' используем tgt как ref_line)
        scale = _norm_factor(self.normalize, f, tgt)
        return raw / scale


# ────────────────────────────────────────────────────────────────────────────
# ЗНАКОВЫЕ АГРЕГАТОРЫ (минимальный API: rho, p, method, normalize)
# ────────────────────────────────────────────────────────────────────────────

# Константы поведения «по-умолчанию» (внутренние; не выносить в публичный API)
_GATE_EPS = 1e-12          # жёсткий порог для отключения награды при наличии нарушений
_LME_TAU_MIN = 1e-12       # минимальный τ для LME (в тех же единицах, что s_i)
_LME_TAU_REL = 0.1         # τ = max(τ_min, τ_rel * m), где m = max(s_i)


def _lme_soft_max(s: np.ndarray) -> float:
    """
    Log-Mean-Exp мягкий максимум для вектора неотрицательных s_i = [r]_-(f_i).
    Свойства:
      * Если все s_i=0 → возвращает 0 (нет «ложной» награды).
      * Гладкая аппроксимация max(s).
      * τ подбирается автоматически: τ = max(τ_min, τ_rel * m), m=max(s).
    """
    if s.size == 0:
        return 0.0
    s = np.asarray(s, dtype=float)
    m = float(np.max(s))
    if m == 0.0:
        return 0.0
    tau = max(_LME_TAU_MIN, _LME_TAU_REL * m)
    z = (s - m) / tau
    # LME: m + τ * log(mean(exp(z)))
    # численно стабильно: z ≤ 0, exp(z) ∈ (0,1], mean(exp(z)) ∈ (0,1]
    return float(m + tau * np.log(np.mean(np.exp(z))))


def _signed_core(
    f: np.ndarray,
    r: np.ndarray,
    *,
    p: int,
    method: str,
    normalize: Union[str, float, Sequence[Union[str, float]]],
    rho: float,
    ref_for_norm: np.ndarray,
) -> float:
    """
    Общая «начинка» для трёх знаковых агрегаторов.

    Вход:
        f   — вектор частот (ГГц),
        r   — остаток (хотим r ≤ 0),
        p, method, normalize — как в интегральных агрегаторах,
        rho — масштаб разумного запаса для затухания,
        ref_for_norm — опорная линия для нормировочного фактора Z (как в *_IntAgg).

    Выход:
        X = A_plus - Reward, где
            A_plus  = mean/trapz([r]_+^p)/Z
            Reward  = 0, если A_plus > eps
                    = S * exp(-S/rho), S = raw_minus/Z
            raw_minus = M_τ^p * (BW, если method='trapz'), M_τ = LME([r]_-)
    """
    if r.size == 0:
        # Нарушение пустого диапазона — оставляем поведение как у интегральных:
        # вызывающий класс решит через on_empty ('raise' или вернуть 0).
        return 0.0

    f = f.astype(float)
    p = int(p)
    method = str(method)
    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("rho must be > 0")

    # 1) Положительная часть: нарушения
    pos = np.clip(r, 0.0, None) ** p
    if method == "trapz":
        raw_plus = _trapz(pos, f)
    elif method == "mean":
        raw_plus = float(np.mean(pos))
    else:
        raise ValueError("method должен быть 'mean' или 'trapz'")

    Z = _norm_factor(normalize, f, ref_for_norm)
    A_plus = raw_plus / Z

    # 2) Жёсткий gate: если есть нарушения — награды нет
    if A_plus > _GATE_EPS:
        return A_plus

    # 3) «Худший запас» — мягкий максимум (LME)
    neg = np.clip(-r, 0.0, None)
    M_tau = _lme_soft_max(neg)    # уже гладко; 0 если запасов нет
    if method == "trapz":
        raw_minus = (M_tau ** p) * _bandwidth(f)
    else:  # 'mean'
        raw_minus = (M_tau ** p)

    # 4) Нормированная величина запаса S и затухающая награда
    S = raw_minus / Z
    # Единственный выбранный закон затухания: S * exp(-S/rho)
    Reward = S * float(np.exp(-S / rho))

    # 5) Итог
    return A_plus - Reward


@register_aggregator(("signed_upint", "SignedUpIntAgg"))
class SignedUpIntAgg(BaseAggregator):
    """
    Знаковая версия UpIntAgg для целей `value ≤ limit(f)` с минимальным API.

    Остаток: r(f) = vals - limit(f), хотим r ≤ 0.

    Формула:
        X = A_plus - Reward
        A_plus  = mean/trapz([r]_+^p) / Z
        Reward  = 0, если A_plus > eps
                = S * exp(-S/rho), S = raw_minus / Z
        raw_minus = (LME([r]_-))^p * (BW, если method='trapz')
        Z = _norm_factor(normalize, f, limit(f))
    """
    def __init__(
        self,
        limit: LimitFunc,
        p: int = 2,
        method: str = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        *,
        rho: float = 0.25,
        on_empty: str = "raise",
    ):
        self.limit = limit
        self.p = int(p)
        self.method = str(method)
        self.normalize = normalize
        self.rho = float(rho)
        self.on_empty = str(on_empty)

    def __call__(self, freq_ghz: np.ndarray, vals: np.ndarray) -> float:
        if vals.size == 0:
            if self.on_empty == "ok":
                return 0.0
            raise ValueError("SignedUpIntAgg: пустой массив значений (диапазон не попал в сетку)")

        f = freq_ghz.astype(float)
        lim = _as_limit_fn(self.limit)(f)
        r = vals - lim
        return _signed_core(
            f, r,
            p=self.p,
            method=self.method,
            normalize=self.normalize,
            rho=self.rho,
            ref_for_norm=lim,
        )


@register_aggregator(("signed_loint", "SignedLoIntAgg"))
class SignedLoIntAgg(BaseAggregator):
    """
    Знаковая версия LoIntAgg для целей `value ≥ limit(f)` с минимальным API.

    Остаток: r(f) = limit(f) - vals, хотим r ≤ 0.

    Формула идентична SignedUpIntAgg (меняется только определение r и ref_line).
    """
    def __init__(
        self,
        limit: LimitFunc,
        p: int = 2,
        method: str = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        *,
        rho: float = 0.25,
        on_empty: str = "raise",
    ):
        self.limit = limit
        self.p = int(p)
        self.method = str(method)
        self.normalize = normalize
        self.rho = float(rho)
        self.on_empty = str(on_empty)

    def __call__(self, freq_ghz: np.ndarray, vals: np.ndarray) -> float:
        if vals.size == 0:
            if self.on_empty == "ok":
                return 0.0
            raise ValueError("SignedLoIntAgg: пустой массив значений (диапазон не попал в сетку)")

        f = freq_ghz.astype(float)
        lim = _as_limit_fn(self.limit)(f)
        r = lim - vals
        return _signed_core(
            f, r,
            p=self.p,
            method=self.method,
            normalize=self.normalize,
            rho=self.rho,
            ref_for_norm=lim,
        )


@register_aggregator(("signed_rippleint", "SignedRippleIntAgg"))
class SignedRippleIntAgg(BaseAggregator):
    """
    Знаковая «рябь» относительно опорной линии с допуском deadzone (минимальный API).

    Остаток:
        построим опорную линию tgt(f) и возьмём
        r(f) = |vals - tgt(f)| - deadzone, хотим r ≤ 0.

    Формула:
        X = A_plus - Reward
        A_plus  = mean/trapz([r]_+^p) / Z
        Reward  = 0, если A_plus > eps
                = S * exp(-S/rho), S = raw_minus / Z
        raw_minus = (LME([r]_-))^p * (BW, если method='trapz')
        Z = _norm_factor(normalize, f, tgt(f))
    """
    def __init__(
        self,
        target: Union[str, float, Callable[[np.ndarray], np.ndarray]] = "mean",
        deadzone: float = 0.0,
        p: int = 2,
        method: str = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        *,
        rho: float = 0.25,
        on_empty: str = "raise",
    ):
        self.target = target
        self.deadzone = float(max(0.0, deadzone))
        self.p = int(p)
        self.method = str(method)
        self.normalize = normalize
        self.rho = float(rho)
        self.on_empty = str(on_empty)

    def __call__(self, freq_ghz: np.ndarray, vals: np.ndarray) -> float:
        if vals.size == 0:
            if self.on_empty == "ok":
                return 0.0
            raise ValueError("SignedRippleIntAgg: пустой массив значений (диапазон не попал в сетку)")

        f = freq_ghz.astype(float)

        # Опорная линия tgt(f)
        if callable(self.target):
            tgt = self.target(f)
        elif isinstance(self.target, (int, float)):
            tgt = np.full_like(vals, float(self.target), dtype=float)
        elif self.target == "mean":
            tgt = np.full_like(vals, float(np.mean(vals)), dtype=float)
        elif self.target == "median":
            tgt = np.full_like(vals, float(np.median(vals)), dtype=float)
        elif self.target == "linear":
            A = np.vstack([np.ones_like(f), f]).T
            coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
            tgt = (coef[0] + coef[1] * f).astype(float)
        else:
            raise ValueError("SignedRippleIntAgg.target должен быть {'mean','median','linear'} | float | callable")

        dev = np.abs(vals - tgt)
        if self.deadzone > 0.0:
            r = dev - self.deadzone
        else:
            r = dev  # deadzone=0 → обычная «рябь» вокруг target

        return _signed_core(
            f, r,
            p=self.p,
            method=self.method,
            normalize=self.normalize,
            rho=self.rho,
            ref_for_norm=tgt,
        )


# ────────────────────────────────────────────────────────────────────────────
# ПУБЛИЧНЫЙ ЭКСПОРТ
# ────────────────────────────────────────────────────────────────────────────

__all__ = [
    # базовые
    "MaxAgg", "MinAgg", "MeanAgg", "RippleAgg", "StdAgg",
    # интегральные без знака
    "UpIntAgg", "LoIntAgg", "RippleIntAgg",
    # знаковые (минималистичный API)
    "SignedUpIntAgg", "SignedLoIntAgg", "SignedRippleIntAgg",
]
