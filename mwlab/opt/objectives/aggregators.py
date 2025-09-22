#mwlab/opt/objectives/aggregators.py
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
* UpIntAgg    – интеграл (или среднее) ВЕРХНИХ нарушений для целей вида `value ≤ limit`;
* LoIntAgg    – интеграл (или среднее) НИЖНИХ нарушений для целей вида `value ≥ limit`;
* RippleIntAgg – интегральная «неравномерность» относительно опорной линии.

Ключевые особенности интегральных агрегаторов:
* Принимают `limit`/`target` как константу ИЛИ функцию от частоты `f_GHz -> ndarray`.
* Возвращают НЕОТРИЦАТЕЛЬНУЮ величину, равную нулю при полном выполнении цели.
* Встроенная нормализация результата (`normalize`):
    - 'none'           – без нормализации;
    - 'bandwidth'      – деление на ширину диапазона (ГГц) или усреднение по точкам;
    - 'limit'          – деление на характерный уровень лимита (среднее |limit(f)|);
    - 'bandwidth*limit' (или ('bandwidth','limit')) – композиция двух нормировок;
    - число (float)    – деление на заданный масштаб.
* Для интегрирования используется безопасный хелпер `_trapz`, который предпочитает
  `np.trapezoid` (актуальные NumPy), но имеет fallback на `np.trapz` (для старых NumPy).

ВАЖНО:
* Интегральные агрегаторы уже возвращают «отнормированную» меру нарушения.
  Поэтому на уровне Comparator удобно сравнивать с нулём и использовать
  мягкие штрафы (ReLU/SoftPlus/Huber и т. п.).
"""

from __future__ import annotations

from typing import Callable, Union, Tuple, Sequence

import numpy as np

from .base import BaseAggregator, register_aggregator


# ────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ХЕЛПЕРЫ
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
    # np.trapz присутствовал долгое время — используем, если есть
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    # крайний случай: численное интегрирование вручную (не попадаем почти никогда)
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


# ────────────────────────────────────────────────────────────────────────────
# БАЗОВЫЕ ПРОСТЫЕ АГРЕГАТОРЫ
# ────────────────────────────────────────────────────────────────────────────

@register_aggregator("max")
class MaxAgg(BaseAggregator):
    """Возвращает максимум значений."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.max(vals))


@register_aggregator("min")
class MinAgg(BaseAggregator):
    """Возвращает минимум значений."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.min(vals))


@register_aggregator("mean")
class MeanAgg(BaseAggregator):
    """Возвращает среднее значений."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.mean(vals))


@register_aggregator("ripple")
class RippleAgg(BaseAggregator):
    """Разброс (max - min)."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.max(vals) - np.min(vals))


@register_aggregator("std")
class StdAgg(BaseAggregator):
    """Стандартное отклонение (population std)."""
    def __call__(self, _: np.ndarray, vals: np.ndarray) -> float:
        return float(np.std(vals))


# ────────────────────────────────────────────────────────────────────────────
# ИНТЕГРАЛЬНЫЕ АГРЕГАТОРЫ ДЛЯ ГЛАДКИХ ЦЕЛЕВЫХ ФУНКЦИЙ
# ────────────────────────────────────────────────────────────────────────────

@register_aggregator(("upint", "UpIntAgg"))
class UpIntAgg(BaseAggregator):
    """
    Интегральная мера ВЕРХНИХ нарушений для целей вида `value ≤ limit`.

    Идея: считаем «площадь нарушения» выше линии лимита.
    Формально:
        lim(f) = константа или функция от частоты (в ГГц)
        viol = clip(vals - lim(f), 0, +inf)
        acc  = mean(viol**p)   или   trapz(viol**p, f)
        out  = acc / norm_factor

    Параметры
    ---------
    limit : float | Callable[[ndarray], ndarray]
        Лимит-линия в тех же единицах, что и `vals` (например, дБ).
    p : {1, 2}, default=2
        Норма для накопления нарушения (L¹ или L²).
    method : {'mean', 'trapz'}, default='mean'
        Способ свертки по частоте: среднее по точкам или интеграл трапецией.
    normalize : {'none','bandwidth','limit','bandwidth*limit'} | float | Sequence, default='bandwidth'
        Нормализация результата. См. `_norm_factor`.
    on_empty : {'raise','ok'}, default='raise'
        Поведение при пустом диапазоне (маска частот пустая).

    Возвращает
    ----------
    float
        Неотрицательная величина (0 → нарушений нет).
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

    Параметры
    ---------
    target : {'mean','median','linear'} | float | Callable, default='mean'
        Опорная линия:
          - 'mean'   – горизонтальная линия на уровне среднего;
          - 'median' – горизонтальная линия на уровне медианы;
          - 'linear' – линейный тренд (МНК: vals ≈ a + b f);
          - число    – постоянная линия;
          - функция  – произвольная зависимость от f.
    deadzone : float, default=0.0
        «Мёртвая зона» (допуск): отклонения меньше `deadzone` не штрафуются.
        Полезно для отсечения численного шума и задание допуска на рябь.
    p : {1, 2}, default=2
        Норма накопления отклонений.
    method : {'mean','trapz'}, default='mean'
        Свертка по частоте: среднее по точкам или интеграл трапецией.
    normalize : {'none','bandwidth','limit','bandwidth*limit'} | float | Sequence, default='bandwidth'
        Нормализация результата. Для варианта 'limit' опорная линия `tgt` играет роль ref_line.
    on_empty : {'raise','ok'}, default='raise'
        Поведение при пустом диапазоне.

    Возвращает
    ----------
    float
        Неотрицательная величина (0 → идеально ровно с учётом deadzone).
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

        # 3) свертка по частоте
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
# ПУБЛИЧНЫЙ ЭКСПОРТ
# ────────────────────────────────────────────────────────────────────────────

__all__ = [
    # базовые
    "MaxAgg", "MinAgg", "MeanAgg", "RippleAgg", "StdAgg",
    # интегральные
    "UpIntAgg", "LoIntAgg", "RippleIntAgg",
]
