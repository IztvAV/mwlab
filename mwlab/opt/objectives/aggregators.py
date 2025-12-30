# mwlab/opt/objectives/aggregators.py
"""
mwlab.opt.objectives.aggregators
================================

Aggregator отвечает на вопрос: «как свернуть 1-D кривую Y(f) в скаляр?».

Роль в архитектуре
------------------
Типичная цепочка вычисления критерия:

    Selector -> Transform -> Aggregator -> Comparator

- Selector извлекает (freq, vals) из rf.Network и задаёт исходные единицы.
- Transform выполняет предобработку кривой (обрезка band, сглаживание, производные, и т.д.).
- Aggregator сворачивает подготовленную кривую в одно число (скаляр).
- Comparator интерпретирует число как ограничение или штраф.

Почему агрегаторы должны быть «unit-aware»
------------------------------------------
Многие ошибки в задачах оптимизации возникают из-за смешения единиц:
например, частота в MHz вместо GHz может «тихо» поменять масштаб интегралов,
производных, апертурных метрик и т.п.

Чтобы избежать подобных ошибок, агрегаторы поддерживают два режима вызова:

1) Старый режим совместимости:
   __call__(freq, vals) -> float
   - агрегатор работает «как есть», предполагая, что его параметры заданы
     в тех же единицах, что и входной freq.

2) Unit-aware режим:
   aggregate(freq, vals, *, freq_unit: str, value_unit: str) -> float
   - агрегатор получает контекст единиц и может:
     * валидировать ожидаемые единицы (expects_freq_unit / expects_value_unit),
     * конвертировать свои частотные параметры (f0, band) в текущие freq_unit,
     * выполнять физически корректные операции на базисе Hz (basis="Hz").

В вычислительном проходе критерия рекомендуется использовать именно aggregate(...).

Инварианты входных данных
-------------------------
- freq и vals должны быть 1-D массивами одинаковой длины.
- частотная ось приводится к монотонно возрастающей (sort_by_freq).
- политики обработки NaN/Inf и пустых данных задаются явно (см. ниже).

Политики обработки NaN/Inf и пустых наборов точек
------------------------------------------------
finite_policy:
- "omit"      : удалить точки с NaN/Inf (и в freq, и в vals),
- "raise"     : при NaN/Inf -> ValueError,
- "propagate" : ничего не делать (NaN может «заразить» результат).

on_empty:
- "raise" : ValueError,
- "nan"   : вернуть np.nan,
- "zero"  : вернуть 0.0.

Замечание: при finite_policy="omit" после фильтрации точек может не остаться —
тогда применяется on_empty.

Комплексные значения
-------------------
Агрегаторы, которые математически предполагают вещественные величины, по умолчанию
запрещают complex (complex_mode="raise"), чтобы исключить «тихие» отбрасывания
мнимой части. При необходимости можно выбрать:
- "abs"  : агрегировать по |y|,
- "real" : агрегировать по Re(y),
- "imag" : агрегировать по Im(y).

Структура файла
---------------
1) Внутренние типы и хелперы (политики, конверсия единиц, фильтрация).
2) Простые агрегаторы (max/min/mean/std/ripple/rms/abs/квантили).
3) Точечный агрегатор (ValueAtAgg).
4) Интегральные агрегаторы (UpIntAgg/LoIntAgg/RippleIntAgg).
5) Signed-агрегаторы (SignedUpIntAgg/SignedLoIntAgg/SignedRippleIntAgg).

"""

from __future__ import annotations

from typing import Optional, Sequence, Union, Literal, Tuple

import numpy as np

from .base import (
    BaseAggregator,
    ensure_1d,
    sort_by_freq,
    interp_at_scalar,
    normalize_freq_unit,
    freq_unit_scale_to_hz,
    register_aggregator,
)

# =============================================================================
# 1) ТИПЫ И ВНУТРЕННИЕ ХЕЛПЕРЫ
# =============================================================================

# Пороговые/опорные линии должны быть сериализуемыми:
# - константа (float)
# - табличное задание как набор точек [(f, y), ...] (линейная интерполяция по f)
LineSpec = Union[float, Sequence[Sequence[float]]]

FinitePolicy = Literal["omit", "raise", "propagate"]
EmptyPolicy = Literal["raise", "nan", "zero"]
ComplexMode = Literal["raise", "abs", "real", "imag"]
IntegralMethod = Literal["mean", "trapz"]
FreqBasis = Literal["native", "Hz"]


_FINITE_POLICIES: Tuple[str, ...] = ("omit", "raise", "propagate")
_EMPTY_POLICIES: Tuple[str, ...] = ("raise", "nan", "zero")
_COMPLEX_MODES: Tuple[str, ...] = ("raise", "abs", "real", "imag")
_INTEGRAL_METHODS: Tuple[str, ...] = ("mean", "trapz")
_FREQ_BASIS: Tuple[str, ...] = ("native", "Hz")


def _handle_empty_scalar(on_empty: str, who: str) -> float:
    """
    Унифицированная политика возвращаемого значения, когда точек нет.

    Используется везде, где агрегатор сворачивает кривую в скаляр.
    """
    mode = str(on_empty).strip().lower()
    if mode not in _EMPTY_POLICIES:
        raise ValueError(f"{who}: on_empty должен быть один из {_EMPTY_POLICIES}")
    if mode == "raise":
        raise ValueError(f"{who}: пустой набор точек (после Transform/фильтрации не осталось данных)")
    if mode == "zero":
        return 0.0
    return float(np.nan)


def _ensure_no_duplicate_freq(f: np.ndarray, who: str) -> None:
    """
    Диагностика дубликатов частоты.

    Дубликаты делают интерполяцию неоднозначной, а интегрирование/производные —
    зависимыми от случайного выбора точки.

    Политика в mwlab:
    - агрегаторы/интерполяция требуют строгой частотной оси без дублей;
    - если данные могут содержать дубли, используйте явный Transform:
        DedupFreqTransform(mode=...)
    """
    f = np.asarray(f, dtype=float)
    if f.size < 2:
        return
    df = np.diff(f)
    # sort_by_freq гарантирует неубывание; df==0 означает дубликаты.
    if bool(np.any(df == 0.0)):
        raise ValueError(
            f"{who}: обнаружены дубликаты freq (нестрого возрастающая ось). "
            "Добавьте DedupFreqTransform(...) в цепочку перед агрегатором."
        )


def _convert_freq_scalar(x: float, from_unit: str, to_unit: str) -> float:
    """
    Перевод одного значения частоты между единицами ("Hz"/"kHz"/"MHz"/"GHz").

    Перевод выполняется через масштаб к Гц:
        x_to = x_from * scale(from) / scale(to)
    """
    fu = normalize_freq_unit(from_unit)
    tu = normalize_freq_unit(to_unit)
    if fu == tu:
        return float(x)
    s_from = freq_unit_scale_to_hz(fu)
    s_to = freq_unit_scale_to_hz(tu)
    return float(x) * (s_from / s_to)


def _coerce_to_real(y: np.ndarray, *, complex_mode: str, who: str) -> np.ndarray:
    """
    Приведение массива значений к вещественному виду согласно complex_mode.

    По умолчанию complex_mode="raise" запрещает «неявное» использование complex.
    """
    y = np.asarray(y)
    if not np.iscomplexobj(y):
        return y.astype(np.float64, copy=False)

    mode = str(complex_mode).strip().lower()
    if mode not in _COMPLEX_MODES:
        raise ValueError(f"{who}: complex_mode должен быть один из {_COMPLEX_MODES}")

    if mode == "raise":
        raise TypeError(
            f"{who}: получены комплексные значения. "
            f"Выберите complex_mode='abs'|'real'|'imag' или преобразуйте данные Transform-ом."
        )
    if mode == "abs":
        return np.abs(y).astype(np.float64, copy=False)
    if mode == "real":
        return np.real(y).astype(np.float64, copy=False)
    # mode == "imag"
    return np.imag(y).astype(np.float64, copy=False)


def _eval_line(line: LineSpec, f: np.ndarray, *, who: str) -> np.ndarray:
    """
    Вычислить опорную/пороговую линию в узлах f.

    Допустимые формы line (строго сериализуемые):
    - число (float) -> константная линия
    - табличное задание: последовательность точек [(f0, y0), (f1, y1), ...]
      интерполируется линейно по частоте.
    """
    if callable(line):  # type: ignore[arg-type]
        raise TypeError(
            f"{who}: callable-значение для линии не поддерживается. "
            "Используйте число или табличное задание [(f, y), ...]."
        )

    f = np.asarray(f, dtype=float)

    # 1) Константа
    if isinstance(line, (int, float, np.number)):
        v = float(line)
        return np.full_like(f, v, dtype=float)

    # 2) Табличное задание
    pts = list(line)  # type: ignore[arg-type]
    if len(pts) < 2:
        raise ValueError(f"{who}: табличная линия должна содержать минимум 2 точки [(f, y), ...]")

    fx: list[float] = []
    fy: list[float] = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"{who}: каждая точка линии должна иметь вид (f, y)")
        fx.append(float(p[0]))
        fy.append(float(p[1]))

    fx_arr = np.asarray(fx, dtype=float)
    fy_arr = np.asarray(fy, dtype=float)

    # Сортировка по частоте
    idx = np.argsort(fx_arr)
    fx_arr = fx_arr[idx]
    fy_arr = fy_arr[idx]

    # Запрещаем повторяющиеся частоты (иначе np.interp становится неоднозначным)
    if np.any(np.diff(fx_arr) == 0.0):
        raise ValueError(f"{who}: в табличной линии обнаружены повторяющиеся значения частоты")

    if f.size == 0:
        return np.asarray([], dtype=float)

    fmin = float(fx_arr[0])
    fmax = float(fx_arr[-1])
    if float(np.min(f)) < fmin or float(np.max(f)) > fmax:
        raise ValueError(
            f"{who}: частотная ось выходит за диапазон табличной линии "
            f"[{fmin}, {fmax}]. Расширьте таблицу или согласуйте диапазоны."
        )

    return np.interp(f, fx_arr, fy_arr).astype(float, copy=False)


def _apply_finite_policy_xy(
    f: np.ndarray,
    y: np.ndarray,
    *,
    finite_policy: str,
    on_empty: str,
    who: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Применить finite_policy к (f, y).

    Возвращает (f2, y2, empty_scalar).
    Если empty_scalar is not None, агрегатор может вернуть его немедленно.
    """
    pol = str(finite_policy).strip().lower()
    if pol not in _FINITE_POLICIES:
        raise ValueError(f"{who}: finite_policy должен быть один из {_FINITE_POLICIES}")

    f = np.asarray(f, dtype=float)
    y = np.asarray(y)

    if f.size == 0:
        return f, y, _handle_empty_scalar(on_empty, who)

    if pol == "propagate":
        return f, y, None

    if pol == "raise":
        if (not np.all(np.isfinite(f))) or (not np.all(np.isfinite(y))):
            raise ValueError(f"{who}: обнаружены NaN/Inf (finite_policy='raise')")
        return f, y, None

    # pol == "omit"
    mask = np.isfinite(f) & np.isfinite(y)
    f2 = f[mask]
    y2 = y[mask]
    if f2.size == 0:
        return f2, y2, _handle_empty_scalar(on_empty, who)
    return f2, y2, None


def _apply_finite_policy_xyz(
    f: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    finite_policy: str,
    on_empty: str,
    who: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """
    Применить finite_policy к тройке массивов (f, y, z), где z — опорная/пороговая линия.

    Это полезно для интегральных и signed-агрегаторов, где невалидность limit/target
    должна обрабатываться согласованно с vals.
    """
    pol = str(finite_policy).strip().lower()
    if pol not in _FINITE_POLICIES:
        raise ValueError(f"{who}: finite_policy должен быть один из {_FINITE_POLICIES}")

    f = np.asarray(f, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if f.size == 0:
        empty = _handle_empty_scalar(on_empty, who)
        return f, y, z, empty

    if pol == "propagate":
        return f, y, z, None

    if pol == "raise":
        if (not np.all(np.isfinite(f))) or (not np.all(np.isfinite(y))) or (not np.all(np.isfinite(z))):
            raise ValueError(f"{who}: обнаружены NaN/Inf (finite_policy='raise')")
        return f, y, z, None

    # pol == "omit"
    mask = np.isfinite(f) & np.isfinite(y) & np.isfinite(z)
    f2 = f[mask]
    y2 = y[mask]
    z2 = z[mask]
    if f2.size == 0:
        empty = _handle_empty_scalar(on_empty, who)
        return f2, y2, z2, empty
    return f2, y2, z2, None


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Безопасная обёртка интегрирования по трапециям.

    np.trapezoid появился в более новых NumPy; поддерживаем и np.trapz.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _bandwidth(f: np.ndarray) -> float:
    """
    Ширина диапазона частот (для нормировок и аппроксимаций).

    Возвращает 1.0 для вырожденного случая.
    """
    f = np.asarray(f, dtype=float)
    if f.size >= 2:
        return float(f[-1] - f[0])
    return 1.0


def _norm_factor(
    normalize: Union[str, float, Sequence[Union[str, float]]],
    f: np.ndarray,
    ref_line: np.ndarray,
) -> float:
    """
    Вычисление положительного множителя нормировки.

    normalize может быть:
    - 'none'/'1'                 -> 1.0
    - 'bandwidth'|'bw'           -> max(bandwidth(f), EPS)
    - 'limit'                    -> max(mean(|ref_line|), EPS)
    - 'bw*limit'|'bandwidth*limit' -> произведение факторов
    - число > 0                  -> явная шкала
    - последовательность         -> произведение указанных факторов

    Нормировка применяется только как масштаб (в знаменателе) для получения
    сопоставимых скалярных метрик.
    """
    EPS = 1e-12

    def _one(item) -> float:
        if isinstance(item, (int, float)):
            return max(float(item), EPS)

        s = str(item).strip().lower()
        if s in ("none", "1"):
            return 1.0

        if s in ("bandwidth", "bw"):
            return max(_bandwidth(f), EPS)

        if s == "limit":
            ref = np.asarray(ref_line, dtype=float)
            if ref.size == 0:
                return 1.0
            m = float(np.mean(np.abs(ref)))
            return max(m, EPS)

        if s in ("bw*limit", "bandwidth*limit"):
            return _one("bw") * _one("limit")

        # Неизвестный режим: «без нормировки», чтобы поведение было предсказуемым.
        return 1.0

    if isinstance(normalize, (list, tuple)):
        factor = 1.0
        for it in normalize:
            factor *= _one(it)
        return max(factor, EPS)

    return _one(normalize)


def _build_target_line(
    target: Union[str, float, LineSpec],
    f: np.ndarray,
    v0: np.ndarray,
    *,
    who: str,
    on_empty: EmptyPolicy,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Унифицированное построение target-линии для Ripple*-агрегаторов.

    Возвращает (tgt, empty_scalar):
    - tgt : вектор целевого значения в узлах f;
    - empty_scalar : если не None, агрегатор может немедленно вернуть это скалярное
      значение (используется для вырожденного случая 'linear' при f.size < 2).
    """
    f = np.asarray(f, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    # 1) Константа
    if isinstance(target, (int, float, np.number)):
        return np.full_like(v0, float(target), dtype=float), None

    mode = str(target).strip().lower()

    # 2) Стандартные строковые режимы
    if mode == "mean":
        return np.full_like(v0, float(np.mean(v0)), dtype=float), None

    if mode == "median":
        return np.full_like(v0, float(np.median(v0)), dtype=float), None

    if mode == "linear":
        # Линейная аппроксимация v0 ≈ a + b f
        if f.size < 2:
            # Вырожденный случай: делегируем политику on_empty агрегатору.
            empty = _handle_empty_scalar(on_empty, who)
            return v0.astype(float, copy=False), empty

        A = np.vstack([np.ones_like(f), f]).T
        coef, *_ = np.linalg.lstsq(A, v0, rcond=None)
        tgt = (coef[0] + coef[1] * f).astype(float, copy=False)
        return tgt, None

    # 3) Табличное задание (LineSpec)
    # Не строковый режим или неизвестная строка -> интерпретируем как LineSpec.
    tgt = _eval_line(target, f, who=f"{who}.target").astype(float, copy=False)  # type: ignore[arg-type]
    return tgt, None

# =============================================================================
# 2) ПРОСТЫЕ АГРЕГАТОРЫ
# =============================================================================
class _SimpleAggBase(BaseAggregator):
    """
    Внутренняя база для простых агрегаторов.

    Предназначение:
    - единые finite_policy/on_empty,
    - единая обработка комплексности через complex_mode,
    - одинаковая подготовка входных данных.

    Публичный контракт:
    - __call__(freq, vals) — режим совместимости (без контекста единиц),
    - aggregate(freq, vals, freq_unit, value_unit) — unit-aware режим.

    По умолчанию агрегатор не навязывает ожидания по единицам, но при желании
    дочерний класс может выставить expects_value_unit/ expects_freq_unit.
    """

    expects_freq_unit: Optional[str] = None
    expects_value_unit: Optional[Union[str, Sequence[str]]] = None

    def __init__(
        self,
        *,
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        complex_mode: ComplexMode = "raise",
        validate: bool = True,
    ):
        self.finite_policy = str(finite_policy).strip().lower()
        self.on_empty = str(on_empty).strip().lower()
        self.complex_mode = str(complex_mode).strip().lower()
        self.validate = bool(validate)

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        _ = freq_unit
        return str(in_unit or "").strip()

    def _prep(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        who: str,
        complex_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        Подготовка данных: 1-D + (опц.) сортировка/валидация + finite_policy + complex->real.

        При validate=False:
        - предполагается, что freq/vals уже 1-D, согласованы по длине и отсортированы;
        - дубликаты частоты не проверяются.
        """
        if self.validate:
            f, y = ensure_1d(freq, vals, who)
            f, y = sort_by_freq(f, y)
        else:
            f = np.asarray(freq)
            y = np.asarray(vals)

        f2, y2, empty = _apply_finite_policy_xy(
            f, y,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who=who,
        )
        if empty is not None:
            return np.asarray(f2, dtype=float), np.asarray(y2), empty

        if self.validate:
            _ensure_no_duplicate_freq(np.asarray(f2, dtype=float), who)

        cm = self.complex_mode if complex_mode is None else str(complex_mode).strip().lower()
        y_real = _coerce_to_real(y2, complex_mode=cm, who=who)
        return np.asarray(f2, dtype=float), y_real, None



@register_aggregator("max")
class MaxAgg(_SimpleAggBase):
    """Максимум по кривой."""
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="MaxAgg")
        if empty is not None:
            return empty
        return float(np.max(y))


@register_aggregator("min")
class MinAgg(_SimpleAggBase):
    """Минимум по кривой."""
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="MinAgg")
        if empty is not None:
            return empty
        return float(np.min(y))


@register_aggregator("mean")
class MeanAgg(_SimpleAggBase):
    """Среднее арифметическое."""
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="MeanAgg")
        if empty is not None:
            return empty
        return float(np.mean(y))


@register_aggregator("std")
class StdAgg(_SimpleAggBase):
    """
    Стандартное отклонение (population std, ddof=0).

    Используется как мера разброса/неравномерности, когда важно «в среднем».
    """
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="StdAgg")
        if empty is not None:
            return empty
        return float(np.std(y, ddof=0))


@register_aggregator("ripple")
class RippleAgg(_SimpleAggBase):
    """
    Размах: max - min.

    Удобен как простая метрика «неравномерности» без выбора опорной линии.
    """
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="RippleAgg")
        if empty is not None:
            return empty
        return float(np.max(y) - np.min(y))


@register_aggregator(("rms", "RMSAgg"))
class RMSAgg(_SimpleAggBase):
    """
    Среднеквадратичное значение:
        RMS = sqrt(mean(y^2))

    Полезно для «энергетических» метрик и сглаженных штрафов.
    """
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="RMSAgg")
        if empty is not None:
            return empty
        return float(np.sqrt(np.mean(y * y)))


@register_aggregator(("abs_max", "AbsMaxAgg"))
class AbsMaxAgg(_SimpleAggBase):
    """
    Максимум модуля |y|.

    Этот агрегатор корректно работает для complex без дополнительных режимов.
    """

    def __init__(
            self,
            *,
            finite_policy: FinitePolicy = "omit",
            on_empty: EmptyPolicy = "raise",
            validate: bool = True,
    ):
        super().__init__(finite_policy=finite_policy, on_empty=on_empty, complex_mode="abs", validate=validate)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="AbsMaxAgg", complex_mode="abs")
        if empty is not None:
            return empty
        return float(np.max(y))

@register_aggregator(("abs_mean", "AbsMeanAgg"))
class AbsMeanAgg(_SimpleAggBase):
    """
    Среднее модуля |y|.

    В отличие от MeanAgg, для complex не возникает неоднозначности.
    """
    def __init__(
            self,
            *,
            finite_policy: FinitePolicy = "omit",
            on_empty: EmptyPolicy = "raise",
            validate: bool = True,
    ):
        super().__init__(finite_policy=finite_policy, on_empty=on_empty, complex_mode="abs", validate=validate)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="AbsMeanAgg", complex_mode="abs")
        if empty is not None:
            return empty
        return float(np.mean(y))

@register_aggregator(("quantile", "QuantileAgg"))
class QuantileAgg(_SimpleAggBase):
    """
    Квантиль q (0..1), например q=0.95 — 95% квантиль.

    Полезно вместо max/min, когда важно «почти везде», но допускаются редкие выбросы.
    """

    def __init__(
        self,
        q: float,
        *,
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        complex_mode: ComplexMode = "raise",
    ):
        super().__init__(finite_policy=finite_policy, on_empty=on_empty, complex_mode=complex_mode)
        self.q = float(q)
        if not (0.0 <= self.q <= 1.0):
            raise ValueError("QuantileAgg.q должен быть в диапазоне [0, 1]")

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        _, y, empty = self._prep(freq, vals, who="QuantileAgg")
        if empty is not None:
            return empty
        return float(np.quantile(y, self.q))


@register_aggregator(("percentile", "PercentileAgg"))
class PercentileAgg(QuantileAgg):
    """
    Перцентиль p (0..100), например p=95 — 95-й перцентиль.

    Это удобная форма задания квантиля в процентах.
    """

    def __init__(
        self,
        p: float,
        *,
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        complex_mode: ComplexMode = "raise",
    ):
        pp = float(p)
        if not (0.0 <= pp <= 100.0):
            raise ValueError("PercentileAgg.p должен быть в диапазоне [0, 100]")
        super().__init__(q=pp / 100.0, finite_policy=finite_policy, on_empty=on_empty, complex_mode=complex_mode)


# =============================================================================
# 3) ТОЧЕЧНЫЙ АГРЕГАТОР: ValueAtAgg
# =============================================================================

@register_aggregator(("value_at", "at_freq", "ValueAtAgg"))
class ValueAtAgg(BaseAggregator):
    """
    Значение кривой в заданной частоте f0 (линейная интерполяция по частоте).

    Параметры
    ---------
    f0 : float
        Опорная частота.
    f0_unit : str | None
        Единицы f0. Если None, то f0 считается заданным в тех же единицах,
        что и входной freq. Если задано (например "MHz"), то в unit-aware режиме
        f0 будет автоматически приведён к текущему freq_unit.
    complex_mode : {"raise","abs","real","imag"}
        Политика для комплексных значений:
        - "raise" : запретить complex,
        - "abs"   : вернуть |y(f0)|,
        - "real"  : вернуть Re(y(f0)),
        - "imag"  : вернуть Im(y(f0)).
    finite_policy, on_empty
        Политики обработки NaN/Inf и пустых данных.
    """

    expects_freq_unit: Optional[str] = None
    expects_value_unit: Optional[Union[str, Sequence[str]]] = None

    def __init__(
        self,
        f0: float,
        *,
        f0_unit: Optional[str] = None,
        complex_mode: ComplexMode = "raise",
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        validate: bool = True,
    ):
        self.f0 = float(f0)
        self.f0_unit = None if f0_unit is None else normalize_freq_unit(f0_unit)

        self.complex_mode = str(complex_mode).strip().lower()
        if self.complex_mode not in _COMPLEX_MODES:
            raise ValueError(f"ValueAtAgg: complex_mode должен быть один из {_COMPLEX_MODES}")

        self.finite_policy = str(finite_policy).strip().lower()
        self.on_empty = str(on_empty).strip().lower()

        self.validate = bool(validate)

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        _ = freq_unit
        return str(in_unit or "").strip()

    def aggregate(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str, value_unit: str) -> float:
        f_unit = normalize_freq_unit(freq_unit)
        f0 = self.f0
        if self.f0_unit is not None:
            f0 = _convert_freq_scalar(f0, self.f0_unit, f_unit)

        return self._value_at(freq, vals, f0=f0)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        # Режим совместимости: f0 считается заданным в единицах входного freq.
        return self._value_at(freq, vals, f0=self.f0)

    def _value_at(self, freq: np.ndarray, vals: np.ndarray, *, f0: float) -> float:
        who = "ValueAtAgg"
        if self.validate:
            f, y = ensure_1d(freq, vals, who)
            f, y = sort_by_freq(f, y)
        else:
            f = np.asarray(freq)
            y = np.asarray(vals)

        f2, y2, empty = _apply_finite_policy_xy(
            f, y,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who=who,
        )
        if empty is not None:
            return empty

        f2 = np.asarray(f2, dtype=float)
        if self.validate:
            _ensure_no_duplicate_freq(f2, who)

        if f2.size > 0 and (f0 < float(f2[0]) or f0 > float(f2[-1])):
            raise ValueError(f"{who}: f0={float(f0)} вне диапазона [{float(f2[0])}, {float(f2[-1])}]")

        if np.iscomplexobj(y2):
            # interp_at_scalar умеет интерполировать complex в рамках линейной интерполяции
            y0 = interp_at_scalar(f2, np.asarray(y2, dtype=np.complex128), float(f0))
            if self.complex_mode == "abs":
                return float(np.abs(y0))
            if self.complex_mode == "real":
                return float(np.real(y0))
            if self.complex_mode == "imag":
                return float(np.imag(y0))
            raise TypeError("ValueAtAgg: комплексные значения не поддержаны (complex_mode='raise')")

        y0r = interp_at_scalar(f2, np.asarray(y2, dtype=float), float(f0))
        return float(y0r)



# =============================================================================
# 4) ИНТЕГРАЛЬНЫЕ АГРЕГАТОРЫ
# =============================================================================

class _IntegralAggBase(BaseAggregator):
    """
    База для интегральных агрегаторов.

    Особенности:
    - валидирует 1-D и сортирует частотную ось,
    - требует вещественные значения (complex запрещён),
    - поддерживает basis="native"|"Hz" для устранения зависимости от единиц freq.

    Метод свёртки (method):
    - "mean"  : среднее по точкам (не зависит от масштаба freq),
    - "trapz" : интеграл по частоте (зависит от масштаба freq, поэтому часто полезен basis="Hz").
    """

    expects_freq_unit: Optional[str] = None
    expects_value_unit: Optional[Union[str, Sequence[str]]] = None

    def __init__(
        self,
        *,
        method: IntegralMethod = "mean",
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        self.method = str(method).strip().lower()
        if self.method not in _INTEGRAL_METHODS:
            raise ValueError(f"method должен быть один из {_INTEGRAL_METHODS}")

        self.finite_policy = str(finite_policy).strip().lower()
        self.on_empty = str(on_empty).strip().lower()

        self.normalize = normalize

        self.validate = bool(validate)

        self.basis = str(basis).strip()
        if self.basis not in _FREQ_BASIS:
            raise ValueError(f"basis должен быть один из {_FREQ_BASIS}")

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        # Интегральные метрики обычно трактуются как «штраф/score» (безразмерные),
        # так как содержат нормировку, степени p и т.п.
        _ = in_unit
        _ = freq_unit
        return ""

    def _prep_real(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        who: str,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        if self.validate:
            f, y = ensure_1d(freq, vals, who)
            f, y = sort_by_freq(f, y)
        else:
            f = np.asarray(freq)
            y = np.asarray(vals)

        if np.iscomplexobj(y):
            raise TypeError(f"{who}: ожидаются вещественные значения; преобразуйте данные Transform-ом")

        f2, y2, empty = _apply_finite_policy_xy(
            f, y.astype(float, copy=False),
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who=who,
        )
        if empty is not None:
            return np.asarray(f2, dtype=float), np.asarray(y2, dtype=float), empty

        f2 = np.asarray(f2, dtype=float)
        y2 = np.asarray(y2, dtype=float)

        if self.validate:
            _ensure_no_duplicate_freq(f2, who)

        return f2, y2, None

    def _basis_freq(self, f: np.ndarray, *, freq_unit: str) -> np.ndarray:
        """
        Частота в базисе интегрирования:
        - native: f как есть,
        - Hz    : f, приведённая к Гц по freq_unit.
        """
        f = np.asarray(f, dtype=float)
        if self.basis == "native":
            return f

        fu = normalize_freq_unit(freq_unit)
        return f * float(freq_unit_scale_to_hz(fu))

    def _reduce(self, f_basis: np.ndarray, v: np.ndarray) -> float:
        if self.method == "mean":
            return float(np.mean(v))
        return _trapz(v, f_basis)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError


@register_aggregator(("upint", "UpIntAgg"))
class UpIntAgg(_IntegralAggBase):
    """
    Интегральная мера верхних нарушений для ограничений вида:
        vals <= limit(f)

    Определение:
        r(f)    = vals(f) - limit(f)
        viol(f) = clip(r(f), 0, +inf)
        acc     = mean(viol^p)  или  trapz(viol^p, f_basis)
        out     = acc / norm_factor(...)

    Замечания:
    - limit задаётся в единицах vals (value_unit), и должен быть согласован по смыслу.
    - basis="Hz" полезен для "trapz", чтобы интеграл имел одинаковый смысл при любом freq_unit.
    """

    def __init__(
        self,
        limit: LineSpec,
        p: int = 2,
        *,
        method: IntegralMethod = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        super().__init__(
            method=method,
            normalize=normalize,
            finite_policy=finite_policy,
            on_empty=on_empty,
            basis=basis,
            validate=validate,
        )
        self.limit = limit
        self.p = int(p)
        if self.p <= 0:
            raise ValueError("p должен быть положительным целым")

    def aggregate(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str, value_unit: str) -> float:
        return self._compute(freq, vals, freq_unit=freq_unit)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        # Режим совместимости: basis="native" эквивалентен, basis="Hz" требует unit-aware вызова.
        if self.basis == "Hz":
            raise ValueError("UpIntAgg(basis='Hz') требует вызова aggregate(..., freq_unit=...)")
        return self._compute(freq, vals, freq_unit="GHz")  # значение не используется при basis='native'

    def _compute(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str) -> float:
        f, v0, empty = self._prep_real(freq, vals, who="UpIntAgg")
        if empty is not None:
            return empty

        lim = _eval_line(self.limit, f, who="UpIntAgg.limit").astype(float, copy=False)
        f, v0, lim, empty2 = _apply_finite_policy_xyz(
            f, v0, lim,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who="UpIntAgg",
        )
        if empty2 is not None:
            return empty2

        r = v0 - lim
        viol = np.clip(r, 0.0, None)
        vv = viol ** self.p

        f_basis = self._basis_freq(f, freq_unit=freq_unit)
        raw = self._reduce(f_basis, vv)

        # Для нормировки по bandwidth используем тот же базис частоты, что и для интеграла.
        Z = _norm_factor(self.normalize, f_basis, lim)
        return raw / Z


@register_aggregator(("loint", "LoIntAgg"))
class LoIntAgg(_IntegralAggBase):
    """
    Интегральная мера нижних нарушений для ограничений вида:
        vals >= limit(f)

    Определение:
        r(f)    = limit(f) - vals(f)
        viol(f) = clip(r(f), 0, +inf)
        acc     = mean(viol^p)  или  trapz(viol^p, f_basis)
        out     = acc / norm_factor(...)
    """

    def __init__(
        self,
        limit: LineSpec,
        p: int = 2,
        *,
        method: IntegralMethod = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        super().__init__(
            method=method,
            normalize=normalize,
            finite_policy=finite_policy,
            on_empty=on_empty,
            basis=basis,
            validate=validate,
        )
        self.limit = limit
        self.p = int(p)
        if self.p <= 0:
            raise ValueError("p должен быть положительным целым")

    def aggregate(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str, value_unit: str) -> float:
        return self._compute(freq, vals, freq_unit=freq_unit)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        if self.basis == "Hz":
            raise ValueError("LoIntAgg(basis='Hz') требует вызова aggregate(..., freq_unit=...)")
        return self._compute(freq, vals, freq_unit="GHz")

    def _compute(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str) -> float:
        f, v0, empty = self._prep_real(freq, vals, who="LoIntAgg")
        if empty is not None:
            return empty

        lim = _eval_line(self.limit, f, who="LoIntAgg.limit").astype(float, copy=False)
        f, v0, lim, empty2 = _apply_finite_policy_xyz(
            f, v0, lim,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who="LoIntAgg",
        )
        if empty2 is not None:
            return empty2

        r = lim - v0
        viol = np.clip(r, 0.0, None)
        vv = viol ** self.p

        f_basis = self._basis_freq(f, freq_unit=freq_unit)
        raw = self._reduce(f_basis, vv)

        Z = _norm_factor(self.normalize, f_basis, lim)
        return raw / Z


@register_aggregator(("rippleint", "RippleIntAgg"))
class RippleIntAgg(_IntegralAggBase):
    """
    Интегральная мера «неравномерности» относительно опорной линии target(f).

    Определение:
        dev_raw(f) = |vals(f) - target(f)|
        dev(f)     = clip(dev_raw(f) - deadzone, 0, +inf)
        acc        = mean(dev^p)  или  trapz(dev^p, f_basis)
        out        = acc / norm_factor(...)

    target может быть:
    - float (константа),
    - "mean"   : константа mean(vals),
    - "median" : константа median(vals),
    - "linear" : линейная аппроксимация vals ≈ a + b f,
    - табличное задание: последовательность точек [(f, y), ...] (линейная интерполяция).
    """

    def __init__(
        self,
        target: Union[str, float, LineSpec] = "mean",
        deadzone: float = 0.0,
        p: int = 2,
        *,
        method: IntegralMethod = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        super().__init__(
            method=method,
            normalize=normalize,
            finite_policy=finite_policy,
            on_empty=on_empty,
            basis=basis,
            validate=validate,
        )
        self.target = target
        self.deadzone = float(max(0.0, deadzone))
        self.p = int(p)
        if self.p <= 0:
            raise ValueError("p должен быть положительным целым")

    def aggregate(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str, value_unit: str) -> float:
        return self._compute(freq, vals, freq_unit=freq_unit)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        if self.basis == "Hz":
            raise ValueError("RippleIntAgg(basis='Hz') требует вызова aggregate(..., freq_unit=...)")
        return self._compute(freq, vals, freq_unit="GHz")

    def _compute(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str) -> float:
        f, v0, empty = self._prep_real(freq, vals, who="RippleIntAgg")
        if empty is not None:
            return empty

        # 1) Построение target-линии
        tgt, empty_tgt = _build_target_line(self.target, f, v0, who="RippleIntAgg", on_empty=self.on_empty)
        if empty_tgt is not None:
            return empty_tgt

        # 2) Согласованная обработка finite
        f, v0, tgt, empty2 = _apply_finite_policy_xyz(
            f, v0, tgt,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who="RippleIntAgg",
        )
        if empty2 is not None:
            return empty2

        # 3) Отклонения + deadzone
        dev = np.abs(v0 - tgt)
        if self.deadzone > 0.0:
            dev = np.clip(dev - self.deadzone, 0.0, None)
        vv = dev ** self.p

        # 4) Усреднение/интеграл
        f_basis = self._basis_freq(f, freq_unit=freq_unit)
        raw = self._reduce(f_basis, vv)

        # Для normalize='limit' используем tgt как ref_line.
        Z = _norm_factor(self.normalize, f_basis, tgt)
        return raw / Z


# =============================================================================
# 5) SIGNED-АГРЕГАТОРЫ
# =============================================================================

_GATE_EPS = 1e-12
_LME_TAU_MIN = 1e-12
_LME_TAU_REL = 0.1


def _lme_soft_max(s: np.ndarray) -> float:
    """
    Log-Mean-Exp мягкий максимум для неотрицательных s_i.

    Идея: использовать гладкий аналог max для устойчивого «поощрения запаса».
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 0.0
    m = float(np.max(s))
    if m == 0.0:
        return 0.0
    tau = max(_LME_TAU_MIN, _LME_TAU_REL * m)
    z = (s - m) / tau
    return float(m + tau * np.log(np.mean(np.exp(z))))


def _signed_core(
    f_basis: np.ndarray,
    r: np.ndarray,
    *,
    p: int,
    method: IntegralMethod,
    normalize: Union[str, float, Sequence[Union[str, float]]],
    rho: float,
    ref_for_norm: np.ndarray,
) -> float:
    """
    Общая «начинка» signed-метрики.

    Вход:
    - r(f) — функция нарушения, хотим r <= 0.
    - f_basis — частота в базисе интегрирования (native или Hz).
    - ref_for_norm — линия для нормировки (limit/target).

    Выход:
    - если нарушений нет, метрика может быть < 0 из-за «награды за запас».
    """
    r = np.asarray(r, dtype=float)
    f_basis = np.asarray(f_basis, dtype=float)
    ref_for_norm = np.asarray(ref_for_norm, dtype=float)

    if r.size == 0:
        return 0.0

    p = int(p)
    if p <= 0:
        raise ValueError("p должен быть положительным целым")

    mth = str(method).strip().lower()
    if mth not in _INTEGRAL_METHODS:
        raise ValueError(f"method должен быть один из {_INTEGRAL_METHODS}")

    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("rho должен быть > 0")

    # 1) Штраф за нарушения (r > 0)
    pos = np.clip(r, 0.0, None) ** p
    raw_plus = _trapz(pos, f_basis) if mth == "trapz" else float(np.mean(pos))

    Z = _norm_factor(normalize, f_basis, ref_for_norm)
    A_plus = raw_plus / Z

    # Если нарушения существенные — не начисляем «награду за запас».
    if A_plus > _GATE_EPS:
        return A_plus

    # 2) Поощрение запаса (r < 0) — мягкий максимум
    neg = np.clip(-r, 0.0, None)
    M = _lme_soft_max(neg)

    # Масштабируем «награду» совместимо с методом
    if mth == "trapz":
        raw_minus = (M ** p) * _bandwidth(f_basis)
    else:
        raw_minus = (M ** p)

    S = raw_minus / Z
    reward = S * float(np.exp(-S / rho))
    return A_plus - reward


class _SignedAggBase(BaseAggregator):
    """
    База для signed-агрегаторов.

    Signed-агрегаторы предназначены для задач оптимизации, где полезно:
    - штрафовать нарушения,
    - но при отсутствии нарушений предпочитать больший «запас» (margin),
      не превращая это в жёсткую «гонку в бесконечность».

    Ограничения:
    - vals должны быть вещественными (complex запрещён),
    - при basis="Hz" требуется unit-aware вызов aggregate(..., freq_unit=...).
    """

    expects_freq_unit: Optional[str] = None
    expects_value_unit: Optional[Union[str, Sequence[str]]] = None

    def __init__(
        self,
        p: int = 2,
        *,
        method: IntegralMethod = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        rho: float = 0.25,
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        self.p = int(p)
        if self.p <= 0:
            raise ValueError("p должен быть положительным целым")

        self.method = str(method).strip().lower()
        if self.method not in _INTEGRAL_METHODS:
            raise ValueError(f"method должен быть один из {_INTEGRAL_METHODS}")

        self.normalize = normalize

        self.rho = float(rho)
        if self.rho <= 0.0:
            raise ValueError("rho должен быть > 0")

        self.finite_policy = str(finite_policy).strip().lower()
        self.on_empty = str(on_empty).strip().lower()

        self.validate = bool(validate)

        self.basis = str(basis).strip()
        if self.basis not in _FREQ_BASIS:
            raise ValueError(f"basis должен быть один из {_FREQ_BASIS}")

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        _ = in_unit
        _ = freq_unit
        return ""

    def _prep_real(self, freq: np.ndarray, vals: np.ndarray, *, who: str) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        if self.validate:
            f, y = ensure_1d(freq, vals, who)
            f, y = sort_by_freq(f, y)
        else:
            f = np.asarray(freq)
            y = np.asarray(vals)

        if np.iscomplexobj(y):
            raise TypeError(f"{who}: ожидаются вещественные значения; преобразуйте данные Transform-ом")

        f2, y2, empty = _apply_finite_policy_xy(
            f, y.astype(float, copy=False),
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who=who,
        )
        if empty is not None:
            return np.asarray(f2, dtype=float), np.asarray(y2, dtype=float), empty

        f2 = np.asarray(f2, dtype=float)
        y2 = np.asarray(y2, dtype=float)

        if self.validate:
            _ensure_no_duplicate_freq(f2, who)

        return f2, y2, None

    def _basis_freq(self, f: np.ndarray, *, freq_unit: str) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        if self.basis == "native":
            return f
        fu = normalize_freq_unit(freq_unit)
        return f * float(freq_unit_scale_to_hz(fu))

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError


@register_aggregator(("signed_upint", "SignedUpIntAgg"))
class SignedUpIntAgg(_SignedAggBase):
    """
    Signed-версия ограничения vals <= limit(f).

    r(f) = vals(f) - limit(f), хотим r <= 0.
    """

    def __init__(
        self,
        limit: LineSpec,
        p: int = 2,
        *,
        method: IntegralMethod = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        rho: float = 0.25,
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        super().__init__(p,
                         method=method,
                         normalize=normalize,
                         rho=rho,
                         finite_policy=finite_policy,
                         on_empty=on_empty,
                         basis=basis,
                         validate=validate,
                         )
        self.limit = limit

    def aggregate(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str, value_unit: str) -> float:
        return self._compute(freq, vals, freq_unit=freq_unit)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        if self.basis == "Hz":
            raise ValueError("SignedUpIntAgg(basis='Hz') требует вызова aggregate(..., freq_unit=...)")
        return self._compute(freq, vals, freq_unit="GHz")

    def _compute(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str) -> float:
        f, v0, empty = self._prep_real(freq, vals, who="SignedUpIntAgg")
        if empty is not None:
            return empty

        lim = _eval_line(self.limit, f, who="SignedUpIntAgg.limit").astype(float, copy=False)
        f, v0, lim, empty2 = _apply_finite_policy_xyz(
            f, v0, lim,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who="SignedUpIntAgg",
        )
        if empty2 is not None:
            return empty2

        r = v0 - lim
        f_basis = self._basis_freq(f, freq_unit=freq_unit)
        return _signed_core(
            f_basis,
            r,
            p=self.p,
            method=self.method,
            normalize=self.normalize,
            rho=self.rho,
            ref_for_norm=lim,
        )


@register_aggregator(("signed_loint", "SignedLoIntAgg"))
class SignedLoIntAgg(_SignedAggBase):
    """
    Signed-версия ограничения vals >= limit(f).

    r(f) = limit(f) - vals(f), хотим r <= 0.
    """

    def __init__(
        self,
        limit: LineSpec,
        p: int = 2,
        *,
        method: IntegralMethod = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        rho: float = 0.25,
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        super().__init__(p,
                         method=method,
                         normalize=normalize,
                         rho=rho,
                         finite_policy=finite_policy,
                         on_empty=on_empty,
                         basis=basis,
                         validate=validate,
                         )
        self.limit = limit

    def aggregate(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str, value_unit: str) -> float:
        return self._compute(freq, vals, freq_unit=freq_unit)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        if self.basis == "Hz":
            raise ValueError("SignedLoIntAgg(basis='Hz') требует вызова aggregate(..., freq_unit=...)")
        return self._compute(freq, vals, freq_unit="GHz")

    def _compute(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str) -> float:
        f, v0, empty = self._prep_real(freq, vals, who="SignedLoIntAgg")
        if empty is not None:
            return empty

        lim = _eval_line(self.limit, f, who="SignedLoIntAgg.limit").astype(float, copy=False)
        f, v0, lim, empty2 = _apply_finite_policy_xyz(
            f, v0, lim,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who="SignedLoIntAgg",
        )
        if empty2 is not None:
            return empty2

        r = lim - v0
        f_basis = self._basis_freq(f, freq_unit=freq_unit)
        return _signed_core(
            f_basis,
            r,
            p=self.p,
            method=self.method,
            normalize=self.normalize,
            rho=self.rho,
            ref_for_norm=lim,
        )


@register_aggregator(("signed_rippleint", "SignedRippleIntAgg"))
class SignedRippleIntAgg(_SignedAggBase):
    """
    Signed-«рябь» относительно target линии с допуском deadzone.

    r(f) = |vals(f) - tgt(f)| - deadzone, хотим r <= 0.
    """

    def __init__(
        self,
        target: Union[str, float, LineSpec] = "mean",
        deadzone: float = 0.0,
        p: int = 2,
        *,
        method: IntegralMethod = "mean",
        normalize: Union[str, float, Sequence[Union[str, float]]] = "bandwidth",
        rho: float = 0.25,
        finite_policy: FinitePolicy = "omit",
        on_empty: EmptyPolicy = "raise",
        basis: FreqBasis = "native",
        validate: bool = True,
    ):
        super().__init__(p,
                         method=method,
                         normalize=normalize,
                         rho=rho,
                         finite_policy=finite_policy,
                         on_empty=on_empty,
                         basis=basis,
                         validate=validate,
                         )
        self.target = target
        self.deadzone = float(max(0.0, deadzone))

    def aggregate(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str, value_unit: str) -> float:
        return self._compute(freq, vals, freq_unit=freq_unit)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        if self.basis == "Hz":
            raise ValueError("SignedRippleIntAgg(basis='Hz') требует вызова aggregate(..., freq_unit=...)")
        return self._compute(freq, vals, freq_unit="GHz")

    def _compute(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str) -> float:
        f, v0, empty = self._prep_real(freq, vals, who="SignedRippleIntAgg")
        if empty is not None:
            return empty

        # target линия
        tgt, empty_tgt = _build_target_line(self.target, f, v0, who="SignedRippleIntAgg", on_empty=self.on_empty)
        if empty_tgt is not None:
            return empty_tgt


        f, v0, tgt, empty2 = _apply_finite_policy_xyz(
            f, v0, tgt,
            finite_policy=self.finite_policy,
            on_empty=self.on_empty,
            who="SignedRippleIntAgg",
        )
        if empty2 is not None:
            return empty2

        dev = np.abs(v0 - tgt)
        r = dev - self.deadzone if self.deadzone > 0.0 else dev

        f_basis = self._basis_freq(f, freq_unit=freq_unit)
        return _signed_core(
            f_basis,
            r,
            p=self.p,
            method=self.method,
            normalize=self.normalize,
            rho=self.rho,
            ref_for_norm=tgt,
        )


# =============================================================================
# 6) ПУБЛИЧНЫЙ ЭКСПОРТ
# =============================================================================

__all__ = [
    # простые
    "MaxAgg",
    "MinAgg",
    "MeanAgg",
    "StdAgg",
    "RippleAgg",
    "RMSAgg",
    "AbsMaxAgg",
    "AbsMeanAgg",
    "QuantileAgg",
    "PercentileAgg",
    # точечные
    "ValueAtAgg",
    # интегральные
    "UpIntAgg",
    "LoIntAgg",
    "RippleIntAgg",
    # signed (advanced)
    "SignedUpIntAgg",
    "SignedLoIntAgg",
    "SignedRippleIntAgg",
]
