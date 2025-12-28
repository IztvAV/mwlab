# mwlab/opt/objectives/transforms.py
"""
mwlab.opt.objectives.transforms
==============================

Transform отвечает на вопрос: «как предварительно преобразовать частотную кривую Y(f)?».

Назначение
----------
Transform работает строго с массивами (freq, vals) и не зависит от rf.Network.
Это делает преобразования:
- переиспользуемыми,
- композиционными,
- удобными для тестирования.

Место в архитектуре
-------------------
Типичная цепочка вычисления критерия:

    Selector -> Transform -> Aggregator -> Comparator

Selector выбирает «что измерять», Transform определяет «как обработать кривую»,
Aggregator сворачивает кривую в число, Comparator интерпретирует число как
ограничение или штраф.

Контекст единиц (unit-aware API)
--------------------------------
В подсистеме целей/ограничений единицы важны не только для отчётов, но и для
физически корректных вычислений.

Поэтому Transform поддерживает два уровня вызова:

1) Низкоуровневый режим (совместимость):
       tr(freq, vals) -> (freq2, vals2)
   В этом режиме Transform оперирует числами и предполагает, что вызывающий
   уже согласовал единицы.

2) Unit-aware режим (рекомендуемый для критерия):
       tr.apply(freq, vals, freq_unit="GHz", value_unit="rad") -> (freq2, vals2)
   В этом режиме Transform получает явный контекст единиц и может:
   - корректно переводить расчёты в базовые единицы (например, f -> Hz для ГВЗ),
   - проверять совместимость входных данных,
   - переводить “физические” параметры (полоса, апертура) в текущую freq_unit.

Параметры с единицами: band_unit и fw_unit
------------------------------------------
Некоторые Transform принимают числовые параметры, которые физически являются
частотами или ширинами окна:
- BandTransform: band=(f1,f2)
- SmoothApertureTransform / ApertureSlopeTransform: fw

Чтобы исключить “тихие” ошибки масштаба, такие Transform поддерживают опциональные
параметры band_unit / fw_unit:
- если unit не задан, считается, что параметр выражен в тех же единицах,
  что и входной freq (т.е. в freq_unit текущего вычислительного прохода);
- если unit задан, параметр интерпретируется в этих единицах и автоматически
  приводится к текущему freq_unit при вызове apply(...).

Инварианты и соглашения
-----------------------
1) Входные массивы должны быть 1-D и одинаковой длины (ensure_1d).
2) Частотная ось приводится к монотонно возрастающей (sort_by_freq).
3) dtype выхода стандартизуется (dtype_out):
   - real     -> float64
   - complex  -> complex128
4) Transform не меняет “числа” freq как таковые (частотная ось остаётся в тех же
   единицах, в которых пришла). Если расчёт требует базовых единиц, это делается
   внутренне и не меняет ось на выходе.
5) Transform может менять единицы vals. Для протаскивания единиц используйте
   out_value_unit(in_unit, freq_unit).

Пустой результат (on_empty)
---------------------------
Некоторые Transform могут вернуть пустую кривую (например, при обрезке band).
Чтобы поведение было предсказуемым, используется on_empty:
- "raise" : возбуждать ValueError
- "ok"    : вернуть пустые массивы

Практическая рекомендация
-------------------------
Для вычисления производных, ГВЗ и «крутизны» обычно полезно применять:
- сглаживание (SmoothPointsTransform или SmoothApertureTransform),
- обработку NaN/Inf (FiniteTransform),
- ресэмплинг (ResampleTransform),
а уже затем дифференцирование / ГВЗ / slope.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Literal

import numpy as np

from .base import (
    BaseTransform,
    OnEmpty,
    norm_on_empty,
    dtype_out,
    ensure_1d,
    handle_empty_curve,
    interp_at_scalar,
    interp_linear,
    normalize_freq_unit,
    freq_unit_scale_to_hz,
    register_transform,
    sort_by_freq,
)

# =============================================================================
# Внутренние утилиты
# =============================================================================
def _prepare_curve(
    freq: np.ndarray,
    vals: np.ndarray,
    who: str,
    *,
    validate: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Нормализация кривой (freq, vals).

    При validate=True:
      - проверяется, что freq и vals — 1-D и одинаковой длины (ensure_1d),
      - частотная ось сортируется по возрастанию (sort_by_freq).

    При validate=False:
      - выполняется только приведение к np.asarray без дополнительных проверок;
        ответственность за 1-D и монотонность freq лежит на вызывающем коде.
    """
    if validate:
        f, y = ensure_1d(freq, vals, who)
        f, y = sort_by_freq(f, y)
    else:
        f = np.asarray(freq)
        y = np.asarray(vals)
    return f, y


def _freq_scale(from_unit: str) -> float:
    """
    Масштаб для перевода частоты из from_unit в Hz: f_hz = f * scale.
    """
    return float(freq_unit_scale_to_hz(normalize_freq_unit(from_unit)))


def _convert_freq_scalar(x: float, from_unit: str, to_unit: str) -> float:
    """
    Конверсия скаляра частоты/ширины окна между единицами.
    """
    fu = normalize_freq_unit(from_unit)
    tu = normalize_freq_unit(to_unit)
    if fu == tu:
        return float(x)
    return float(x) * (_freq_scale(fu) / _freq_scale(tu))


def _convert_band(
    band: Tuple[float, float],
    *,
    band_unit: Optional[str],
    freq_unit: str,
) -> Tuple[float, float]:
    """
    Привести band к текущей freq_unit.
    """
    f1, f2 = float(band[0]), float(band[1])
    if f2 < f1:
        f1, f2 = f2, f1

    if band_unit is None:
        return f1, f2

    bu = normalize_freq_unit(band_unit)
    fu = normalize_freq_unit(freq_unit)
    return (
        _convert_freq_scalar(f1, bu, fu),
        _convert_freq_scalar(f2, bu, fu),
    )


def _convert_fw(
    fw: float,
    *,
    fw_unit: Optional[str],
    freq_unit: str,
) -> float:
    """
    Привести ширину окна fw к текущей freq_unit.
    """
    if fw_unit is None:
        return float(fw)
    return _convert_freq_scalar(float(fw), normalize_freq_unit(fw_unit), normalize_freq_unit(freq_unit))


# =============================================================================
# Compose — последовательная композиция Transform-ов
# =============================================================================

class Compose(BaseTransform):
    """
    Последовательная композиция Transform-ов в один объект.

    - В unit-aware режиме apply(...) контекст единиц передаётся дальше по цепочке.
    - value_unit обновляется через out_value_unit(), если Transform его реализует.
    """

    def __init__(self, transforms: Sequence[BaseTransform]):
        self.transforms: Tuple[BaseTransform, ...] = tuple(transforms)
        self.name = "Compose"

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, y = freq, vals
        for tr in self.transforms:
            f, y = tr(f, y)
        return f, y

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        f, y = freq, vals
        fu = normalize_freq_unit(freq_unit)
        vu = str(value_unit or "")

        for tr in self.transforms:
            fn_apply = getattr(tr, "apply", None)
            if not callable(fn_apply):
                # В unit-aware режиме нельзя тихо деградировать в __call__ без контекста единиц.
                raise TypeError(
                    f"Compose.apply: transform {tr.__class__.__name__} не поддерживает apply(...). "
                    "В unit-aware режиме все transforms должны реализовывать apply(...)."
                )

            f, y = fn_apply(f, y, freq_unit=fu, value_unit=vu)

            fn_out = getattr(tr, "out_value_unit", None)
            vu = fn_out(vu, fu) if callable(fn_out) else vu

        return f, y

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        """
        Протаскивание единиц через композицию.

        Если Transform не реализует out_value_unit(), считается, что единицы не меняются.
        """
        u = str(in_unit or "")
        fu = normalize_freq_unit(freq_unit)
        for tr in self.transforms:
            fn = getattr(tr, "out_value_unit", None)
            u = fn(u, fu) if callable(fn) else u
        return u

    def __len__(self) -> int:  # pragma: no cover
        return len(self.transforms)

    def iter_transforms(self):
        for tr in self.transforms:
            fn_iter = getattr(tr, "iter_transforms", None)
            if callable(fn_iter):
                yield from fn_iter()
            else:
                yield tr

    def __repr__(self) -> str:  # pragma: no cover
        inner = ", ".join(tr.__class__.__name__ for tr in self.transforms)
        return f"Compose([{inner}])"


# =============================================================================
# BandTransform — обрезка диапазона + включение границ интерполяцией
# =============================================================================

@register_transform(("band", "BandTransform"))
class BandTransform(BaseTransform):
    """
    Обрезка кривой по диапазону частот band=(f1, f2).

    Параметры
    ---------
    band : (float, float)
        Границы диапазона.
    include_edges : bool
        Если True, гарантирует присутствие точек f1 и f2 на выходе (если диапазон
        пересекается с исходной сеткой). Значения в этих точках вычисляются линейной
        интерполяцией.
    band_unit : str | None
        Единицы, в которых задан band. Если None, band считается заданным в тех же
        единицах, что и входной freq.
    tol, rtol : float
        Допуски для проверки “точка уже присутствует на сетке”.
    on_empty : {"raise","ok"}
        Политика при пустом результате.
    validate : bool, optional
        Включать ли структурные проверки входной кривой (1-D, одинаковая длина,
        сортировка по частоте). По умолчанию True; установите False для
        «доверенных» кривых в горячих циклах оптимизации.
    """

    def __init__(
        self,
        band: Tuple[float, float],
        *,
        include_edges: bool = False,
        band_unit: Optional[str] = None,
        tol: float = 0.0,
        rtol: float = 0.0,
        on_empty: str = "raise",
        validate: bool = True,
    ):
        self.band = (float(band[0]), float(band[1]))
        self.include_edges = bool(include_edges)

        self.band_unit = None if band_unit is None else normalize_freq_unit(band_unit)

        self.tol = float(max(0.0, tol))
        self.rtol = float(max(0.0, rtol))
        self.on_empty: OnEmpty = norm_on_empty(on_empty, "BandTransform")
        self.validate = bool(validate)
        self.name = "Band"

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = value_unit
        band = _convert_band(self.band, band_unit=self.band_unit, freq_unit=freq_unit)
        return self._apply_band(freq, vals, band)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Совместимость: в __call__ считаем, что band задан в тех же единицах, что и freq.
        band = _convert_band(self.band, band_unit=None, freq_unit="Hz")
        return self._apply_band(freq, vals, band)

    def _apply_band(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        band: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "BandTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if f.size == 0:
            return handle_empty_curve(self.on_empty, "BandTransform", y_dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        f1, f2 = float(band[0]), float(band[1])
        if f2 < f1:
            f1, f2 = f2, f1

        fmin, fmax = float(f[0]), float(f[-1])

        # Нет пересечения диапазонов
        if f2 < fmin or f1 > fmax:
            return handle_empty_curve(self.on_empty, "BandTransform", y_dtype=y_dt)

        # Клипуем band к доступному диапазону
        g1 = max(f1, fmin)
        g2 = min(f2, fmax)
        if g2 < g1:
            return handle_empty_curve(self.on_empty, "BandTransform", y_dtype=y_dt)

        mask = (f >= g1) & (f <= g2)
        f_sel = f[mask]
        y_sel = y[mask]

        # Если по маске ничего не попало, но include_edges=True, можно построить точки границ
        if f_sel.size == 0:
            if not self.include_edges:
                return handle_empty_curve(self.on_empty, "BandTransform", y_dtype=y_dt)

            if g1 == g2:
                y1 = interp_at_scalar(f, y, g1, atol=self.tol, rtol=self.rtol)
                return np.asarray([g1], dtype=float), np.asarray([y1], dtype=y_dt)

            y1 = interp_at_scalar(f, y, g1, atol=self.tol, rtol=self.rtol)
            y2 = interp_at_scalar(f, y, g2, atol=self.tol, rtol=self.rtol)
            return np.asarray([g1, g2], dtype=float), np.asarray([y1, y2], dtype=y_dt)

        if not self.include_edges:
            return np.asarray(f_sel, dtype=float), np.asarray(y_sel, dtype=y_dt)

        def _has_point(ff: np.ndarray, x0: float) -> bool:
            return bool(np.any(np.isclose(ff, x0, atol=self.tol, rtol=self.rtol)))

        out_f = np.asarray(f_sel, dtype=float)
        out_y = np.asarray(y_sel, dtype=y_dt)

        if not _has_point(out_f, g1):
            y1 = interp_at_scalar(f, y, g1, atol=self.tol, rtol=self.rtol)
            k = int(np.searchsorted(out_f, g1))
            out_f = np.insert(out_f, k, g1)
            out_y = np.insert(out_y, k, np.asarray(y1, dtype=y_dt))

        if not _has_point(out_f, g2):
            y2 = interp_at_scalar(f, y, g2, atol=self.tol, rtol=self.rtol)
            k = int(np.searchsorted(out_f, g2))
            out_f = np.insert(out_f, k, g2)
            out_y = np.insert(out_y, k, np.asarray(y2, dtype=y_dt))

        return np.asarray(out_f, dtype=float), np.asarray(out_y, dtype=y_dt)


# =============================================================================
# DedupFreqTransform — нормализация частотной оси (сортировка + дубликаты)
# =============================================================================

@register_transform(("dedup_freq", "deduplicate_freq", "DedupFreqTransform"))
class DedupFreqTransform(BaseTransform):
    """
    Нормализация частотной оси: сортировка + обработка дубликатов частоты.

    Назначение
    ----------
    Во многих производных операциях (интерполяция, интегрирование, производные)
    наличие повторяющихся значений freq делает задачу численно неоднозначной.
    Этот Transform позволяет явно задать политику работы с дублями.
    Параметры
    ---------
    mode : {"raise","keep_first","keep_last","mean"}
        "raise"      : при наличии дубликатов возбуждает ValueError;
        "keep_first" : оставить первое значение y для каждого freq;
        "keep_last"  : оставить последнее значение y;
        "mean"       : заменить группу дублей средним по y.
    on_empty : {"raise","ok"}
        Политика при пустом входе (см. handle_empty_curve).
    validate : bool, optional
        Включать ли структурные проверки входной кривой (1-D, одинаковая длина,
        сортировка по частоте). По умолчанию True; установите False для
        «доверенных» кривых.
    """
    def __init__(
        self,
        *,
        mode: Literal["raise", "keep_first", "keep_last", "mean"] = "raise",
        on_empty: str = "raise",
        validate: bool = True,
    ):
        m = str(mode).strip().lower()
        if m not in ("raise", "keep_first", "keep_last", "mean"):
            raise ValueError("DedupFreqTransform.mode должен быть 'raise'|'keep_first'|'keep_last'|'mean'")
        self.mode: Literal["raise", "keep_first", "keep_last", "mean"] = m  # type: ignore[assignment]

        self.on_empty: OnEmpty = norm_on_empty(on_empty, "DedupFreqTransform")
        self.validate = bool(validate)
        self.name = f"DedupFreq(mode={self.mode})"

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "DedupFreqTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if f.size == 0:
            return handle_empty_curve(self.on_empty, "DedupFreqTransform", y_dtype=y_dt)
        if f.size == 1:
            return np.asarray(f, dtype=float), np.asarray(y, dtype=y_dt)

        # При validate=True f уже отсортирован по частоте; при validate=False
        # порядок остаётся как есть, ответственность на вызывающем коде.
        f_sorted, y_sorted = f, y

        f_sorted = np.asarray(f_sorted, dtype=float)
        y_sorted = np.asarray(y_sorted, dtype=y_dt)
        uniq_f, idx_first, inv, counts = np.unique(
            f_sorted, return_index=True, return_inverse=True, return_counts=True
        )

        has_dups = bool(np.any(counts > 1))
        if not has_dups:
            # Уже строгая монотонность по freq.
            return f_sorted, y_sorted

        if self.mode == "raise":
            raise ValueError("DedupFreqTransform: обнаружены дубликаты частоты; задайте mode!='raise' для их обработки")

        if self.mode == "keep_first":
            y_new = y_sorted[idx_first]
        elif self.mode == "keep_last":
            idx_last = idx_first + counts - 1
            y_new = y_sorted[idx_last]
        else:  # "mean"
            if np.iscomplexobj(y_sorted):
                sum_r = np.bincount(inv, weights=np.real(y_sorted))
                sum_i = np.bincount(inv, weights=np.imag(y_sorted))
                y_new = (sum_r + 1j * sum_i) / counts
                y_new = y_new.astype(np.complex128, copy=False)
            else:
                sum_y = np.bincount(inv, weights=y_sorted.astype(float))
                y_new = (sum_y / counts).astype(np.float64, copy=False)

        return np.asarray(uniq_f, dtype=float), np.asarray(y_new, dtype=y_dt)

# =============================================================================
# SmoothPointsTransform — сглаживание по точкам (оконное усреднение по NS точкам)
# =============================================================================

@register_transform(("smooth_points", "smooth", "SmoothPointsTransform"))
class SmoothPointsTransform(BaseTransform):
    """
    Сглаживание по числу точек (оконное среднее по индексу).

    Этот Transform не зависит от единиц частоты: окно определяется в количестве точек.
    Параметры
    ---------
    n_pts : int
        Размер окна по числу точек.
    on_empty : {"raise","ok"}
        Политика при пустом входе.
    validate : bool, optional
        Включать ли структурные проверки входной кривой. По умолчанию True.
    """

    def __init__(self, n_pts: int, *, on_empty: str = "raise", validate: bool = True):
        self.n_pts = int(n_pts)
        if self.n_pts <= 0:
            raise ValueError("SmoothPointsTransform.n_pts должен быть положительным")

        self.on_empty: OnEmpty = norm_on_empty(on_empty, "SmoothPointsTransform")
        self.validate = bool(validate)
        self.name = f"SmoothPoints(n={self.n_pts})"

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "SmoothPointsTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if y.size == 0:
            return handle_empty_curve(self.on_empty, "SmoothPointsTransform", y_dtype=y_dt)

        if self.n_pts <= 1 or y.size == 1:
            return np.asarray(f, dtype=float), np.asarray(y, dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        NS = self.n_pts
        N = y.size

        SL = int(np.floor((NS - 1) / 2))
        SR = int(np.ceil((NS - 1) / 2))

        idx = np.arange(N, dtype=int)
        L = np.maximum(0, idx - SL)
        R = np.minimum(N - 1, idx + SR)
        cnt = (R - L + 1).astype(float)

        def _smooth_real(yr: np.ndarray) -> np.ndarray:
            yr = np.asarray(yr, dtype=float)
            cs = np.cumsum(yr, dtype=float)
            cs0 = np.concatenate([[0.0], cs])
            s = cs0[R + 1] - cs0[L]
            return s / cnt

        if np.iscomplexobj(y):
            yr = _smooth_real(np.real(y))
            yi = _smooth_real(np.imag(y))
            ys = (yr + 1j * yi).astype(np.complex128)
        else:
            ys = _smooth_real(y).astype(np.float64)

        return f, np.asarray(ys, dtype=y_dt)

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = (freq_unit, value_unit)
        return self(freq, vals)


# =============================================================================
# SmoothApertureTransform — сглаживание по частотной апертуре FW
# =============================================================================

@register_transform(("smooth_aperture", "smooth_fw", "SmoothApertureTransform"))
class SmoothApertureTransform(BaseTransform):
    """
    Сглаживание по частотной апертуре FW (окно фиксированной ширины по частоте).

    Усреднение выполняется как интеграл по окну, делённый на ширину окна:
      y_s(i) = (1/(fr_i - fl_i)) * ∫_{fl_i}^{fr_i} y(f) df

    Интеграл считается по кусочно-линейной интерполяции y(f), что корректно учитывает
    вклад границ окна, даже если они лежат между узлами сетки.

    Параметры
    ---------
    fw : float
        Ширина окна.
    fw_unit : str | None
        Единицы, в которых задан fw. Если None, fw считается заданным в тех же единицах,
        что и входной freq.
    use_effective_fw : bool
        Если True, делить на фактическую ширину окна (fr-fl) у края диапазона.
        Если False, делить на номинальную fw.
    validate : bool, optional
        Включать ли структурные проверки (1-D, сортировка частоты). По умолчанию True;
        установите False, если кривая уже подготовлена и порядок частот гарантирован.
    """

    def __init__(
        self,
        fw: float,
        *,
        fw_unit: Optional[str] = None,
        use_effective_fw: bool = True,
        on_empty: str = "raise",
        validate: bool = True,
    ):
        self.fw = float(fw)
        if self.fw <= 0.0:
            raise ValueError("SmoothApertureTransform.fw должен быть > 0")

        self.fw_unit = None if fw_unit is None else normalize_freq_unit(fw_unit)
        self.use_effective_fw = bool(use_effective_fw)
        self.on_empty: OnEmpty = norm_on_empty(on_empty, "SmoothApertureTransform")
        self.validate = bool(validate)
        self.name = f"SmoothAperture(FW={self.fw})"

    @staticmethod
    def _cumtrapz(f: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Кумулятивный интеграл по трапециям:
          I[k] = ∫_{f[0]}^{f[k]} y(f) df
        """
        f = np.asarray(f, dtype=float)
        y = np.asarray(y)

        out_dt = np.complex128 if np.iscomplexobj(y) else np.float64
        I = np.zeros_like(y, dtype=out_dt)
        if f.size < 2:
            return I

        df = np.diff(f)
        seg = 0.5 * (y[1:] + y[:-1]) * df
        I[1:] = np.cumsum(seg)
        return I

    @staticmethod
    def _cumtrapz_eval(f: np.ndarray, y: np.ndarray, I: np.ndarray, xq: np.ndarray) -> np.ndarray:
        """
        Вычислить I(xq) = ∫_{f[0]}^{xq} y(f) df для xq внутри [f[0], f[-1]].

        Реализация:
        - индексируем сегмент, содержащий xq, через searchsorted;
        - используем устойчивое сравнение xq с узлами сетки через atol,
          согласованный с минимальным шагом частоты.
        """
        f = np.asarray(f, dtype=float)
        y = np.asarray(y)
        I = np.asarray(I)

        xq = np.asarray(xq, dtype=float)

        out_dt = np.complex128 if np.iscomplexobj(y) else np.float64
        out = np.zeros_like(xq, dtype=out_dt)

        idx = np.searchsorted(f, xq, side="right") - 1
        idx = np.clip(idx, 0, f.size - 1)

        df = np.diff(f)
        df = df[np.isfinite(df) & (df > 0.0)]
        min_df = float(np.min(df)) if df.size else 1.0

        eps = np.finfo(float).eps
        atol = max(16.0 * eps * min_df, 1e-15)

        at_node = np.isclose(xq, f[idx], rtol=0.0, atol=atol)
        if np.any(at_node):
            out[at_node] = I[idx][at_node]

        mid = ~at_node
        if np.any(mid):
            j = idx[mid]
            j = np.clip(j, 0, f.size - 2)

            f0 = f[j]
            f1 = f[j + 1]
            y0 = y[j]
            y1 = y[j + 1]
            dx = (f1 - f0)
            dt = (xq[mid] - f0)

            with np.errstate(divide="ignore", invalid="ignore"):
                slope = (y1 - y0) / dx
                seg_int = y0 * dt + 0.5 * slope * (dt * dt)

            out[mid] = I[j] + seg_int

        return out

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = value_unit
        fw_local = _convert_fw(self.fw, fw_unit=self.fw_unit, freq_unit=freq_unit)
        return self._apply_fw(freq, vals, fw_local)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fw_local = float(self.fw)
        return self._apply_fw(freq, vals, fw_local)

    def _apply_fw(self, freq: np.ndarray, vals: np.ndarray, fw_local: float) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "SmoothApertureTransform", validate=self.validate)

        y_dt = dtype_out(y)

        if f.size == 0:
            return handle_empty_curve(self.on_empty, "SmoothApertureTransform", y_dtype=y_dt)
        if f.size == 1:
            # Для одиночной точки апертурное усреднение вырождается в тождественное отображение.
            return np.asarray(f, dtype=float), np.asarray(y, dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        fmin = float(f[0])
        fmax = float(f[-1])

        half = 0.5 * fw_local
        fl = np.clip(f - half, fmin, fmax)
        fr = np.clip(f + half, fmin, fmax)

        I = self._cumtrapz(f, y)
        I_fl = self._cumtrapz_eval(f, y, I, fl)
        I_fr = self._cumtrapz_eval(f, y, I, fr)

        denom = (fr - fl) if self.use_effective_fw else np.full_like(f, fw_local, dtype=float)
        denom_ok = np.isfinite(denom) & (denom != 0.0)

        fill = (np.nan + 1j * np.nan) if y_dt == np.complex128 else np.nan
        ys = np.full_like(y, fill, dtype=y_dt)

        with np.errstate(divide="ignore", invalid="ignore"):
            avg = (I_fr - I_fl) / denom

        ys[denom_ok] = avg[denom_ok].astype(y_dt, copy=False)
        return f, ys


# =============================================================================
# ResampleTransform — ресэмплинг на равномерную сетку из num_points
# =============================================================================

@register_transform(("resample", "num_points", "ResampleTransform"))
class ResampleTransform(BaseTransform):
    """
    Ресэмплинг кривой на равномерную сетку по частоте (между f[0] и f[-1]).

    Параметры
    ---------
    num_points : int
        Число точек на новой равномерной сетке.
    on_empty : {"raise","ok"}
        Политика при пустом входе.
    validate : bool, optional
        Включать ли структурные проверки входной кривой. По умолчанию True.
    """

    def __init__(self, num_points: int, *, on_empty: str = "raise", validate: bool = True):
        self.num_points = int(num_points)
        if self.num_points < 2:
            raise ValueError("ResampleTransform.num_points должен быть >= 2")

        self.on_empty: OnEmpty = norm_on_empty(on_empty, "ResampleTransform")
        self.validate = bool(validate)
        self.name = f"Resample(M={self.num_points})"

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "ResampleTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if f.size < 2:
            return handle_empty_curve(self.on_empty, "ResampleTransform", y_dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        f_new = np.linspace(float(f[0]), float(f[-1]), self.num_points, dtype=float)
        y_new = interp_linear(f, y, f_new).astype(y_dt, copy=False)
        return f_new, y_new

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = (freq_unit, value_unit)
        return self(freq, vals)


# =============================================================================
# Вспомогательный тип для shift
# =============================================================================

ShiftRefMode = Literal["mean", "median", "min", "max", "first", "last", "value_at"]
ShiftRef = Union[float, complex, ShiftRefMode]


def _compute_shift_ref(f: np.ndarray, y: np.ndarray, ref: ShiftRef, f0: Optional[float]) -> Union[float, complex]:
    """
    Унифицированное вычисление опорного значения y_ref для Shift-операций.

    ref может быть:
      - числом (float/complex),
      - строковым режимом: mean/median/min/max/first/last/value_at.
    """
    # Защита от случайной передачи функции/лямбды (несериализуемо в YAML/JSON).
    if callable(ref):  # type: ignore[arg-type]
        raise TypeError(
            "Shift*: параметр ref не поддерживает callable. "
            "Используйте число или строковый режим ('mean'|'median'|'min'|'max'|'first'|'last'|'value_at'). "
            "Для произвольной логики создайте отдельный Transform и зарегистрируйте его по alias."
        )

    if isinstance(ref, (int, float, complex, np.number)):
        return ref  # type: ignore[return-value]

    mode = str(ref).strip().lower()
    if mode == "mean":
        return np.nanmean(y).item()
    if mode == "median":
        if np.iscomplexobj(y):
            return (np.nanmedian(np.real(y)) + 1j * np.nanmedian(np.imag(y))).item()
        return float(np.nanmedian(y))
    if mode == "min":
        return np.nanmin(y).item()
    if mode == "max":
        return np.nanmax(y).item()
    if mode == "first":
        return y[0].item()
    if mode == "last":
        return y[-1].item()
    if mode == "value_at":
        if f0 is None:
            raise ValueError("Shift*: для ref='value_at' нужно задать f0")
        return interp_at_scalar(f, y, float(f0))

    raise ValueError(
        "ref должен быть float|complex|"
        "'mean'|'median'|'min'|'max'|'first'|'last'|'value_at'"
    )


# =============================================================================
# ShiftTransform — сдвиг уровня: y'(f) = y(f) - y_ref
# =============================================================================

@register_transform(("shift", "yshifter", "ShiftTransform"))
class ShiftTransform(BaseTransform):
    """
    Сдвиг уровня: y'(f) = y(f) - y_ref.

    Единицы values сохраняются.

    Параметры
    ---------
    ref : ShiftRef
        Опорное значение или режим вычисления опоры.
    validate : bool, optional
        Включать ли структурные проверки входной кривой. По умолчанию True.
    """

    def __init__(self, ref: ShiftRef, *, f0: Optional[float] = None, on_empty: str = "raise", validate: bool = True):
        self.ref = ref
        self.f0 = None if f0 is None else float(f0)
        self.on_empty: OnEmpty = norm_on_empty(on_empty, "ShiftTransform")
        self.validate = bool(validate)
        self.name = "Shift"

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "ShiftTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if y.size == 0:
            return handle_empty_curve(self.on_empty, "ShiftTransform", y_dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        y_ref = _compute_shift_ref(f, y, self.ref, self.f0)
        return f, np.asarray(y - y_ref, dtype=y_dt)

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = (freq_unit, value_unit)
        return self(freq, vals)


# =============================================================================
# ShiftByRefInBandTransform — ref вычисляется только по band, shift применяется ко всей кривой
# =============================================================================

@register_transform(("shift_in_band", "shift_by_ref_in_band", "ShiftByRefInBandTransform"))
class ShiftByRefInBandTransform(BaseTransform):
    """
    Сдвиг уровня, где опорное значение y_ref вычисляется только в заданном диапазоне band,
    а затем применяется ко всей кривой.

    Параметры
    ---------
    band : (float,float)
        Диапазон, на котором вычисляется ref.
    band_unit : str | None
        Единицы band. Если None, band задан в единицах входной freq.
    include_edges : bool
        Если True, границы band учитываются с линейной интерполяцией.
    validate : bool, optional
        Включать ли структурные проверки (1-D, сортировка частоты) до вычисления
        опорного значения. По умолчанию True.
    """

    def __init__(
        self,
        band: Tuple[float, float],
        ref: ShiftRef,
        *,
        band_unit: Optional[str] = None,
        f0: Optional[float] = None,
        include_edges: bool = True,
        tol: float = 0.0,
        rtol: float = 0.0,
        on_empty: str = "raise",
        validate: bool = True,
    ):
        self.band = (float(band[0]), float(band[1]))
        self.band_unit = None if band_unit is None else normalize_freq_unit(band_unit)

        self.ref = ref
        self.f0 = None if f0 is None else float(f0)

        self.include_edges = bool(include_edges)
        self.tol = float(max(0.0, tol))
        self.rtol = float(max(0.0, rtol))
        self.on_empty: OnEmpty = norm_on_empty(on_empty, "ShiftByRefInBandTransform")
        self.validate = bool(validate)
        self.name = f"ShiftByRefInBand({self.band[0]}..{self.band[1]})"

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = value_unit
        band_local = _convert_band(self.band, band_unit=self.band_unit, freq_unit=freq_unit)
        return self._apply(freq, vals, band_local)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        band_local = self.band
        return self._apply(freq, vals, band_local)

    def _apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        band_local: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "ShiftByRefInBandTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if y.size == 0:
            return handle_empty_curve(self.on_empty, "ShiftByRefInBandTransform", y_dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        band_tr = BandTransform(
            band_local,
            include_edges=self.include_edges,
            tol=self.tol,
            rtol=self.rtol,
            on_empty=self.on_empty,
            # Кривая уже нормализована выше; повторная валидация здесь не нужна.
            validate = False,
        )
        fb, yb = band_tr(f, y)
        if fb.size == 0:
            return handle_empty_curve(self.on_empty, "ShiftByRefInBandTransform", y_dtype=y_dt)

        y_ref = _compute_shift_ref(fb, yb, self.ref, self.f0)
        return f, np.asarray(y - y_ref, dtype=y_dt)


# =============================================================================
# SignTransform — явное изменение знака
# =============================================================================

@register_transform(("sign", "SignTransform"))
class SignTransform(BaseTransform):
    """
    Явное изменение знака: y -> sign * y, где sign ∈ {+1, -1}.

    Единицы values сохраняются.

    Параметры
    ---------
    validate : bool, optional
        Включать ли структурные проверки входной кривой. По умолчанию True.
    """

    def __init__(self, sign: int, *, validate: bool = True):
        s = int(sign)
        if s not in (-1, +1):
            raise ValueError("SignTransform.sign должен быть +1 или -1")
        self.sign = s
        self.validate = bool(validate)
        self.name = f"Sign({self.sign:+d})"

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "SignTransform", validate=self.validate)

        y_dt = dtype_out(y)
        return np.asarray(f, dtype=float), (np.asarray(y, dtype=y_dt) * self.sign)

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = (freq_unit, value_unit)
        return self(freq, vals)


# =============================================================================
# FiniteTransform — обработка NaN/Inf
# =============================================================================

@register_transform(("finite", "FiniteTransform"))
class FiniteTransform(BaseTransform):
    """
    Политика обработки невалидных значений (NaN/Inf) в кривой.

    - Частотная ось является “координатой”, поэтому невалидные значения freq
      не могут быть корректно “заполнены”: такие точки всегда удаляются.
    - Для vals доступна политика drop/raise/fill.

    Параметры
    ---------
    validate : bool, optional
        Включать ли структурные проверки входной кривой (1-D, сортировка).
        По умолчанию True.
    """

    def __init__(
        self,
        mode: Literal["drop", "raise", "fill"] = "drop",
        *,
        fill_value: Union[float, complex] = 0.0,
        on_empty: str = "raise",
        validate: bool = True,
    ):
        self.mode = str(mode).lower()
        if self.mode not in ("drop", "raise", "fill"):
            raise ValueError("FiniteTransform.mode должен быть 'drop'|'raise'|'fill'")

        self.fill_value = fill_value
        self.on_empty: OnEmpty = norm_on_empty(on_empty, "FiniteTransform")
        self.validate = bool(validate)
        self.name = f"Finite({self.mode})"

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "FiniteTransform", validate=self.validate)

        y_dt = dtype_out(y)
        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        ok_f = np.isfinite(f)
        ok_y = np.isfinite(y)
        ok = ok_f & ok_y

        if self.mode == "raise":
            if not bool(np.all(ok)):
                raise ValueError("FiniteTransform: обнаружены NaN/Inf в данных")
            return f, y

        # Всегда удаляем точки с невалидной частотой
        if not bool(np.all(ok_f)):
            f = f[ok_f]
            y = y[ok_f]
            ok_y = np.isfinite(y)
            ok = ok_y  # теперь ok относится только к vals

        if f.size == 0:
            return handle_empty_curve(self.on_empty, "FiniteTransform", y_dtype=y_dt)

        if self.mode == "fill":
            y2 = y.copy()
            y2[~ok] = np.asarray(self.fill_value, dtype=y_dt)
            return f, y2

        # mode == "drop"
        f2 = f[ok]
        y2 = y[ok]
        if f2.size == 0:
            return handle_empty_curve(self.on_empty, "FiniteTransform", y_dtype=y_dt)
        return f2, y2

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = (freq_unit, value_unit)
        return self(freq, vals)


# =============================================================================
# DerivativeTransform — численная производная dy/df
# =============================================================================

@register_transform(("derivative", "diff", "DerivativeTransform"))
class DerivativeTransform(BaseTransform):
    """
    Численная производная dy/df по частоте.

    Параметр basis задаёт “базу” единиц частоты, в которых вычисляется производная:
    - basis="native": df берётся в единицах входного freq (по умолчанию).
    - basis="Hz"    : df берётся в Герцах, независимо от freq_unit входа
                      (требуется вызывать через apply(...) с корректным freq_unit).

    Это удобно, когда требуется физически сопоставимая производная при разных freq_unit.
    Параметры
    ---------
    validate : bool, optional — структурные проверки входной кривой. По умолчанию True.
    """

    def __init__(
        self,
        *,
        method: Literal["diff", "gradient"] = "diff",
        basis: Literal["native", "Hz"] = "native",
        abs_value: bool = False,
        on_empty: str = "raise",
        validate: bool = True,
    ):
        self.method = str(method).lower()
        if self.method not in ("diff", "gradient"):
            raise ValueError("DerivativeTransform.method должен быть 'diff' или 'gradient'")

        self.basis = str(basis)
        if self.basis not in ("native", "Hz"):
            raise ValueError("DerivativeTransform.basis должен быть 'native' или 'Hz'")

        self.abs_value = bool(abs_value)
        self.on_empty: OnEmpty = norm_on_empty(on_empty, "DerivativeTransform")
        self.validate = bool(validate)
        self.name = f"Derivative({self.method}, basis={self.basis})"

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        u = str(in_unit or "").strip()
        if self.basis == "Hz":
            return f"{u}/Hz" if u else "1/Hz"
        fu = normalize_freq_unit(freq_unit)
        return f"{u}/{fu}" if u else f"1/{fu}"

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = value_unit
        return self._apply(freq, vals, freq_unit=normalize_freq_unit(freq_unit))

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Совместимость: basis="Hz" в __call__ интерпретирует freq как Hz.
        return self._apply(freq, vals, freq_unit="Hz")

    def _apply(self, freq: np.ndarray, vals: np.ndarray, *, freq_unit: str) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "DerivativeTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if f.size < 2:
            return handle_empty_curve(self.on_empty, "DerivativeTransform", y_dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        f_work = f
        if self.basis == "Hz":
            f_work = f * _freq_scale(freq_unit)

        if self.method == "diff":
            df = np.diff(f_work)
            dy = np.diff(y)

            ok = np.isfinite(df) & (df != 0.0) & np.isfinite(dy)
            if not np.any(ok):
                return handle_empty_curve(self.on_empty, "DerivativeTransform", y_dtype=y_dt)

            f_mid = 0.5 * (f[1:] + f[:-1])
            f_mid = f_mid[ok]
            d = (dy[ok] / df[ok])

            if self.abs_value:
                return np.asarray(f_mid, dtype=float), np.abs(d).astype(np.float64)

            out_dt = np.complex128 if np.iscomplexobj(d) else np.float64
            return np.asarray(f_mid, dtype=float), np.asarray(d, dtype=out_dt)

        # method == "gradient"
        if np.iscomplexobj(y):
            dr = np.gradient(np.real(y).astype(float), f_work)
            di = np.gradient(np.imag(y).astype(float), f_work)
            d = (dr + 1j * di).astype(np.complex128)
        else:
            d = np.gradient(y.astype(float), f_work).astype(np.float64)

        if self.abs_value:
            d = np.abs(d).astype(np.float64)

        return f, d


# =============================================================================
# GroupDelayTransform — ГВЗ как производная фазы по частоте в Гц
# =============================================================================

@register_transform(("group_delay", "gd", "GroupDelayTransform"))
class GroupDelayTransform(BaseTransform):
    """
    Групповое время запаздывания (ГВЗ) по фазовой характеристике.

    Определение:
      GD(f) = -(1/(2*pi)) * d(phi)/df

    где:
      - phi — фаза (rad или deg),
      - f   — частота в Гц.

    Требования к данным
    -------------------
    - Фаза должна быть непрерывной по частоте (unwrap выполняется до этого Transform).
    - Частотная ось может быть в любых единицах, но для физически корректной производной
      всегда используется df в Гц.

    Выход
    -----
    - Частотная ось: midpoints исходной сетки (в исходных единицах входного freq).
    - Значения: время задержки в "s" или "ns" (задаётся out_unit).

    Параметры
    ---------
    validate : bool, optional — структурные проверки (1-D, сортировка). По умолчанию True.
    """

    def __init__(
        self,
        *,
        out_unit: Literal["s", "ns"] = "ns",
        on_empty: str = "raise",
        validate: bool = True,
    ):
        ou = str(out_unit).strip().lower()
        if ou not in ("s", "ns"):
            raise ValueError("GroupDelayTransform.out_unit должен быть 's' или 'ns'")
        self.out_unit: Literal["s", "ns"] = ou  # type: ignore[assignment]

        self.on_empty: OnEmpty = norm_on_empty(on_empty, "GroupDelayTransform")
        self.validate = bool(validate)
        self.name = f"GroupDelay({self.out_unit})"

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        _ = (in_unit, freq_unit)
        return "ns" if self.out_unit == "ns" else "s"

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fu = normalize_freq_unit(freq_unit)
        vu = str(value_unit or "").strip().lower()
        return self._apply(freq, vals, freq_unit=fu, value_unit=vu)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Совместимость: в __call__ считаем freq в Hz и фазу в rad.
        return self._apply(freq, vals, freq_unit="Hz", value_unit="rad")

    def _apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        f, phi = _prepare_curve(freq, vals, "GroupDelayTransform", validate=self.validate)

        if f.size < 2:
            return handle_empty_curve(self.on_empty, "GroupDelayTransform", y_dtype=np.float64)

        f = np.asarray(f, dtype=float)

        if np.iscomplexobj(phi):
            raise TypeError("GroupDelayTransform: ожидается вещественная фаза (rad/deg), а не complex")
        phi = np.asarray(phi, dtype=float)

        if value_unit not in ("rad", "deg"):
            raise ValueError("GroupDelayTransform: value_unit должен быть 'rad' или 'deg'")

        if value_unit == "deg":
            phi = np.deg2rad(phi)

        f_hz = f * _freq_scale(freq_unit)

        df = np.diff(f_hz)
        dphi = np.diff(phi)

        ok = np.isfinite(df) & (df != 0.0) & np.isfinite(dphi)
        if not np.any(ok):
            return handle_empty_curve(self.on_empty, "GroupDelayTransform", y_dtype=np.float64)

        f_mid = 0.5 * (f[1:] + f[:-1])
        f_mid = f_mid[ok]

        with np.errstate(divide="ignore", invalid="ignore"):
            gd_s = -(dphi[ok] / df[ok]) / (2.0 * np.pi)  # seconds

        gd = gd_s * 1e9 if self.out_unit == "ns" else gd_s
        gd = np.asarray(gd, dtype=np.float64)

        ok2 = np.isfinite(f_mid) & np.isfinite(gd)
        return np.asarray(f_mid[ok2], dtype=float), np.asarray(gd[ok2], dtype=np.float64)


# =============================================================================
# ApertureSlopeTransform — «крутизна» через апертуру FW
# =============================================================================

@register_transform(("aperture_slope", "slope_fw", "ApertureSlopeTransform"))
class ApertureSlopeTransform(BaseTransform):
    """
    «Крутизна» через апертуру FW: (y(fr) - y(fl)) / (fr - fl).

    Окно сдвигается так, чтобы не выходить за границы диапазона.
    Интерполяция значений на концах окна выполняется линейно.

    Параметры
    ---------
    fw : float
        Ширина окна.
    fw_unit : str | None
        Единицы fw. Если None, fw задан в единицах входного freq.
    use_effective_fw : bool
        Если True, делить на фактическую ширину (fr-fl) у края диапазона.
        Если False, делить на номинальную fw.
    abs_value : bool
        Если True, возвращать модуль крутизны (float64).
    validate : bool, optional
        Включать ли структурные проверки входной кривой (1-D, сортировка).
        По умолчанию True.
    """

    def __init__(
        self,
        fw: float,
        *,
        fw_unit: Optional[str] = None,
        use_effective_fw: bool = True,
        abs_value: bool = True,
        on_empty: str = "raise",
        validate: bool = True,
    ):
        self.fw = float(fw)
        if self.fw <= 0.0:
            raise ValueError("ApertureSlopeTransform.fw должен быть > 0")

        self.fw_unit = None if fw_unit is None else normalize_freq_unit(fw_unit)

        self.use_effective_fw = bool(use_effective_fw)
        self.abs_value = bool(abs_value)
        self.on_empty: OnEmpty = norm_on_empty(on_empty, "ApertureSlopeTransform")
        self.validate = bool(validate)
        self.name = f"ApertureSlope(FW={self.fw})"

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        u = str(in_unit or "").strip()
        fu = normalize_freq_unit(freq_unit)
        return f"{u}/{fu}" if u else f"1/{fu}"

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _ = value_unit
        fw_local = _convert_fw(self.fw, fw_unit=self.fw_unit, freq_unit=freq_unit)
        return self._apply_fw(freq, vals, fw_local)

    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fw_local = float(self.fw)
        return self._apply_fw(freq, vals, fw_local)

    def _apply_fw(self, freq: np.ndarray, vals: np.ndarray, fw_local: float) -> Tuple[np.ndarray, np.ndarray]:
        f, y = _prepare_curve(freq, vals, "ApertureSlopeTransform", validate=self.validate)

        y_dt = dtype_out(y)
        if f.size < 2:
            return handle_empty_curve(self.on_empty, "ApertureSlopeTransform", y_dtype=y_dt)

        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=y_dt)

        fmin = float(f[0])
        fmax = float(f[-1])
        if fmax <= fmin:
            return handle_empty_curve(self.on_empty, "ApertureSlopeTransform", y_dtype=y_dt)

        # Нормированная координата 0..1 для управления сдвигом окна у границ
        S = (f - fmin) / (fmax - fmin)
        S = np.clip(S, 0.0, 1.0)

        fl = f - fw_local * S
        fr = fl + fw_local

        fl = np.clip(fl, fmin, fmax)
        fr = np.clip(fr, fmin, fmax)

        y_fl = interp_linear(f, y, fl)
        y_fr = interp_linear(f, y, fr)

        denom = (fr - fl) if self.use_effective_fw else np.full_like(fl, fw_local, dtype=float)
        denom_ok = np.isfinite(denom) & (denom != 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (y_fr - y_fl) / denom

        if self.abs_value:
            out_dt = np.float64
            fill = np.nan
            slope_out = np.abs(slope).astype(out_dt)
        else:
            out_dt = np.complex128 if np.iscomplexobj(slope) else np.float64
            fill = (np.nan + 1j * np.nan) if out_dt == np.complex128 else np.nan
            slope_out = np.asarray(slope, dtype=out_dt)

        z = np.full(f.shape, fill, dtype=out_dt)
        z[denom_ok] = slope_out[denom_ok]
        return f, z


# =============================================================================
# Публичный экспорт
# =============================================================================

__all__ = [
    # композиция
    "Compose",
    # диапазоны/сетка
    "BandTransform",
    "DedupFreqTransform",
    "ResampleTransform",
    # сглаживание
    "SmoothPointsTransform",
    "SmoothApertureTransform",
    # арифметика/преобразования значения
    "ShiftTransform",
    "ShiftByRefInBandTransform",
    "SignTransform",
    "FiniteTransform",
    # производные/ГВЗ/крутизна
    "DerivativeTransform",
    "GroupDelayTransform",
    "ApertureSlopeTransform",
]
