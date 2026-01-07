# mwlab/opt/objectives/selectors.py
"""
mwlab.opt.objectives.selectors
==============================

Модуль селекторов (Selector) подсистемы целей/ограничений mwlab.

Зачем нужен этот модуль
-----------------------
Selector отвечает на вопрос: «**какую именно** частотную зависимость нужно извлечь из
`NetworkLike`?». Он является **источником данных** для вычисления критериев.

Каждый Selector возвращает пару 1-D массивов одинаковой длины:

    (freq, vals)

где:
- `freq` — частотная ось **в единицах `selector.freq_unit`** (канонический вид:
  "Hz", "kHz", "MHz", "GHz"),
- `vals` — значения кривой **в семантических единицах `selector.value_unit`**
  (например: "db", "lin", "rad", "deg", "complex").

Далее в пайплайне:
    Selector -> Transform -> Aggregator -> Comparator

Важно: «прозрачность» семантики
-------------------------------
Selector **не должен** внедрять инженерные соглашения ТЗ, которые меняют физический
смысл величины. Например, если в ТЗ потери описаны как `-20*log10(|S|)`, то:
- Selector возвращает стандартное `20*log10(|S|)` (как принято в scikit-rf),
- смена знака выполняется явным Transform (например, SignTransform(-1)).

Это делает цепочки:
- воспроизводимыми,
- читаемыми,
- переиспользуемыми в разных ТЗ.

Соглашения по частоте
---------------------
- Внутри `NetworkLike` частота хранится в Гц: `net.frequency.f`.
- На выходе Selector переводит частоту в `freq_unit` и по умолчанию приводит
  ось к монотонно возрастающей, чтобы downstream-компоненты могли полагаться на
  корректный порядок. Для «доверенных» сетей эти проверки можно отключить,
  передав `validate=False` в конструктор селектора.

Параметр band
-------------
Во многих селекторах есть `band=(f1, f2)`:
- band задаётся в **тех же единицах**, что и `freq_unit`;
- реализуется как маска по существующим точкам сетки (без добавления граничных
  точек через интерполяцию).
Если требуется строгая обрезка с включением границ — используйте Transform
`BandTransform(..., include_edges=True)`.

Численные особенности
---------------------
- Для величин в db при |S|=0 возможны `-inf`. Это считается нормальным: политику
  обработки NaN/Inf задавайте Transform-ом (например, FiniteTransform).
- Для производных и derived-метрик (ГВЗ, «крутизна») обычно применяют сначала
  сглаживание/ресэмплинг/finite-policy, и только затем дифференцирование.

Состав модуля
-------------
1) Внутренние утилиты:
   - проверка/конвертация портов,
   - проверка доступности портов в сети,
   - применение band,
   - согласованная сортировка одной частотной оси и нескольких массивов.
2) Базовые S-селекторы:
   - SComplexSelector: комплексный S_mn(f)
   - SMagSelector: |S_mn| в lin или db
   - PhaseSelector: фаза arg(S_mn) в rad/deg (unwrap опционально)
3) Пример derived-селектора:
   - AxialRatioSelector: демонстрационный расчёт Axial Ratio по S31/S41

Примечание о единицах и валидации
--------------------------------
В подсистеме objectives валидация совместимости единиц выполняется на уровне
Criterion/Transform/Aggregator. Поэтому селекторы должны строго и однозначно
задавать:
- `freq_unit` (канонический вид),
- `value_unit` (семантический ярлык).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .network_like import NetworkLike
from .registry import register_selector
from .base import (
    BaseSelector,
    convert_freq_from_hz,
    ensure_1d,
    normalize_freq_unit,
    sort_by_freq,
)

# =============================================================================
# Внутренние утилиты
# =============================================================================

def _ports_to_0based(m: int, n: int) -> Tuple[int, int]:
    """
    Перевести номера портов из 1-based инженерной записи (S21: m=2, n=1)
    в 0-based индексацию NumPy/scikit-rf.

    Параметры
    ---------
    m, n : int
        Номера портов, начиная с 1.

    Возвращает
    ----------
    (m0, n0) : (int, int)
        Индексы портов в 0-based индексации.

    Исключения
    ----------
    TypeError
        Если m или n не являются int.
    ValueError
        Если m <= 0 или n <= 0.
    """
    if not isinstance(m, int) or not isinstance(n, int):
        raise TypeError("Номера портов должны быть целыми числами (int)")
    if m <= 0 or n <= 0:
        raise ValueError("Порты должны задаваться с 1 (например, S11 -> m=1, n=1)")
    return m - 1, n - 1


def _ensure_port_exists(net: NetworkLike, m0: int, n0: int, who: str) -> None:
    """
        Проверить, что в сети достаточно портов для доступа к S[m0, n0].

        Это даёт понятную диагностику вместо низкоуровневого IndexError.

        Параметры
        ---------
        net : NetworkLike
            Сеть, совместимая со scikit-rf.
        m0, n0 : int
            Порты в 0-based индексации.
        who : str
            Имя вызывающего селектора для сообщения ошибки.
        """
    nports = getattr(net, "nports", None)

    if nports is None:
        # fallback: пытаемся вычислить число портов из shape массива S
        try:
            s = getattr(net, "s", None)
            if s is not None:
                s = np.asarray(s)
                if s.ndim >= 3:
                    nports = int(s.shape[1])
        except Exception:
            nports = None

    if nports is None:
        # не удалось определить — не мешаем работе
        return

    nports_i = int(nports)
    if m0 < 0 or n0 < 0 or m0 >= nports_i or n0 >= nports_i:
        raise ValueError(
            f"{who}: сеть имеет nports={nports_i}, но запрошен элемент S[{m0},{n0}] "
            f"(порты в 0-based индексации). Проверьте номера портов."
        )


def _apply_band(
    freq: np.ndarray,
    vals: np.ndarray,
    band: Tuple[float, float] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Применить band=(f1, f2) как маску по текущей сетке.

    Замечания
    ---------
    - band задаётся в тех же единицах, что и freq (то есть в freq_unit селектора).
    - Границы не интерполируются: выбираются только существующие точки сетки.
    """
    f = np.asarray(freq)
    y = np.asarray(vals)
    if band is None:
        return f, y

    if len(band) != 2:
        raise ValueError("параметр `band` должен содержать ровно два числа: [f1, f2]")

    f1, f2 = float(band[0]), float(band[1])
    if f2 < f1:
        f1, f2 = f2, f1

    mask = (f >= f1) & (f <= f2)
    return f[mask], y[mask]


def _sort_freq_and_apply(
    f: np.ndarray,
    arrays: Sequence[np.ndarray],
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Отсортировать частотную ось один раз и применить ту же перестановку к нескольким массивам.

    Это важно для derived-метрик, где несколько кривых должны оставаться согласованными:
    сортировка каждой кривой "по отдельности" способна привести к скрытому рассогласованию.

    Параметры
    ---------
    f : np.ndarray
        Частотная ось (обычно в Гц из skrf).
    arrays : Sequence[np.ndarray]
        Массивы значений, которые нужно переставить так же, как и f.

    Возвращает
    ----------
    f_sorted : np.ndarray
        Отсортированная частотная ось (float).
    arrays_sorted : tuple[np.ndarray, ...]
        Массивы, переставленные согласованно с f_sorted.
    """
    f = np.asarray(f, dtype=float)

    if len(arrays) == 0:
        f2, _ = sort_by_freq(f, np.asarray([], dtype=float))
        return f2, tuple()

    # Используем sort_by_freq с return_idx=True на первом массиве,
    # затем применяем idx ко всем остальным массивам.
    f2, a0, idx = sort_by_freq(f, arrays[0], return_idx=True)

    out = [np.asarray(a0)]
    for a in arrays[1:]:
        out.append(np.asarray(a)[idx])

    return np.asarray(f2, dtype=float), tuple(out)

# =============================================================================
# 1) SComplexSelector — комплексный S-параметр
# =============================================================================

@register_selector(("SComplexSelector", "s_complex", "s", "sparam"))
class SComplexSelector(BaseSelector):
    """
    Комплексный S-параметр S_mn(f).

    Это «низкоуровневый» селектор: на его основе Transform-ами можно построить
    амплитуду/фазу/unwrap/производные/пользовательские метрики.

    Parameters
    ----------
    m, n : int
        Номера портов, начиная с 1 (S11 -> 1,1).
    band : (float, float) | None
        Диапазон частот в единицах freq_unit. None -> вся сетка.
    freq_unit : str
        "Hz"/"kHz"/"MHz"/"GHz" (регистр не важен).
    name : str | None
        Человекочитаемое имя кривой.
    validate : bool, optional
        Включать ли проверки портов, сортировку оси частот и проверку
        размерностей (ensure_1d). По умолчанию True; установите False для
        «доверенных» сетей в горячих циклах оптимизации.
    """

    def __init__(
        self,
        m: int,
        n: int,
        *,
        band: Tuple[float, float] | None = None,
        freq_unit: str = "GHz",
        name: Optional[str] = None,
        validate: bool = True,
    ):
        # ВАЖНО для serde:
        # - m/n храним в 1-based виде (канонический инженерный формат для YAML/JSON)
        # - 0-based индексы используем только внутренне
        self.m = int(m)
        self.n = int(n)
        self._m0, self._n0 = _ports_to_0based(self.m, self.n)
        self.band = band

        self.freq_unit = normalize_freq_unit(freq_unit)
        self.value_unit = "complex"
        self.name = name or f"S{m}{n}_complex"
        self.validate = bool(validate)

    def __call__(self, net: NetworkLike) -> Tuple[np.ndarray, np.ndarray]:
        if self.validate:
            _ensure_port_exists(net, self._m0, self._n0, "SComplexSelector")

        f_hz = np.asarray(net.frequency.f, dtype=float)
        s_mn = np.asarray(net.s[:, self._m0, self._n0], dtype=np.complex128)

        if self.validate:
            f_hz, (s_mn,) = _sort_freq_and_apply(f_hz, [s_mn])

        freq = convert_freq_from_hz(f_hz, self.freq_unit)
        freq, s_mn = _apply_band(freq, s_mn, self.band)

        if self.validate:
            freq, s_mn = ensure_1d(freq, s_mn, who="SComplexSelector")

        return freq, s_mn

# =============================================================================
# 2) SMagSelector — |S_mn| в lin или в db (стандартное определение)
# =============================================================================

@register_selector(("SMagSelector", "s_mag", "sdb", "smag"))
class SMagSelector(BaseSelector):
    """
    Магнитуда |S_mn| в линейном масштабе или в db.

    Если db=True, возвращается стандартное:
        20*log10(|S|)
    (как в scikit-rf: net.s_db).

    В этом селекторе нет скрытых инженерных соглашений. Если нужна величина
    вида -20*log10(|S|), используйте Transform SignTransform(-1).

    Parameters
    ----------
    m, n : int
        Порты, начиная с 1 (S21 -> 2,1).
    band : (float, float) | None
        Диапазон частот (в единицах freq_unit). Маска по сетке.
    db : bool
        True -> 20*log10(|S|), False -> |S|.
    freq_unit : str
        Единицы частоты на выходе.
    name : str | None
        Имя кривой.
    validate : bool, optional
        Включать ли проверки портов, сортировку оси частот и проверку
        размерностей (ensure_1d). По умолчанию True; установите False для
        «доверенных» сетей в горячих циклах оптимизации.
    """

    def __init__(
        self,
        m: int,
        n: int,
        *,
        band: Tuple[float, float] | None = None,
        db: bool = True,
        freq_unit: str = "GHz",
        name: Optional[str] = None,
        validate: bool = True,
    ):
        # serde: m/n храним 1-based, вычисления делаем через внутренние 0-based индексы
        self.m = int(m)
        self.n = int(n)
        self._m0, self._n0 = _ports_to_0based(self.m, self.n)
        self.band = band
        self.db = bool(db)

        self.freq_unit = normalize_freq_unit(freq_unit)
        self.value_unit = "db" if self.db else "lin"
        self.name = name or f"S{m}{n}_{'db' if self.db else 'mag'}"
        self.validate = bool(validate)

    def __call__(self, net: NetworkLike) -> Tuple[np.ndarray, np.ndarray]:
        if self.validate:
            _ensure_port_exists(net, self._m0, self._n0, "SMagSelector")

        f_hz = np.asarray(net.frequency.f, dtype=float)

        # scikit-rf:
        # - net.s_mag -> |S| (lin)
        # - net.s_db  -> 20 log10 |S|
        y = net.s_db[:, self._m0, self._n0] if self.db else net.s_mag[:, self._m0, self._n0]
        y = np.asarray(y, dtype=np.float64)

        if self.validate:
            f_hz, (y,) = _sort_freq_and_apply(f_hz, [y])

        freq = convert_freq_from_hz(f_hz, self.freq_unit)
        freq, y = _apply_band(freq, y, self.band)

        if self.validate:
            freq, y = ensure_1d(freq, y, who="SMagSelector")

        return freq, y

# =============================================================================
# 3) PhaseSelector — фаза S_mn(f)
# =============================================================================

@register_selector(("PhaseSelector", "phase", "phase_s"))
class PhaseSelector(BaseSelector):
    """
    Фазовая характеристика arg(S_mn(f)).

    По умолчанию unwrap=True, поскольку производные по фазе (ГВЗ и т.п.)
    требуют непрерывной фазы.

    Важно
    -----
    ГВЗ следует вычислять Transform-ом, например:
      PhaseSelector(..., unwrap=True, unit="rad") -> GroupDelayTransform(...)

    Parameters
    ----------
    m, n : int
        Порты, начиная с 1.
    band : (float, float) | None
        Диапазон частот в единицах freq_unit.
    unwrap : bool
        Разворачивать ли фазу вдоль частоты.
    unit : {"rad","deg"}
        Единицы фазы на выходе.
    freq_unit : str
        Единицы частоты на выходе.
    name : str | None
        Имя кривой.
    validate : bool, optional
        Включать ли проверки портов, сортировку оси частот и проверку
        размерностей (ensure_1d). По умолчанию True; установите False для
        «доверенных» сетей в горячих циклах оптимизации.
    """

    def __init__(
        self,
        m: int,
        n: int,
        *,
        band: Tuple[float, float] | None = None,
        unwrap: bool = True,
        unit: str = "rad",
        freq_unit: str = "GHz",
        name: Optional[str] = None,
        validate: bool = True,
    ):
        # serde: m/n храним 1-based; для net.s используем внутренние индексы 0-based
        self.m = int(m)
        self.n = int(n)
        self._m0, self._n0 = _ports_to_0based(self.m, self.n)
        self.band = band
        self.unwrap = bool(unwrap)

        u = str(unit).strip().lower()
        if u not in ("rad", "deg"):
            raise ValueError("PhaseSelector.unit должен быть 'rad' или 'deg'")
        self.unit = u

        self.freq_unit = normalize_freq_unit(freq_unit)
        self.value_unit = self.unit
        self.name = name or f"Phase_S{m}{n}_{self.unit}"
        self.validate = bool(validate)

    def __call__(self, net: NetworkLike) -> Tuple[np.ndarray, np.ndarray]:
        if self.validate:
            _ensure_port_exists(net, self._m0, self._n0, "PhaseSelector")

        f_hz = np.asarray(net.frequency.f, dtype=float)
        s_mn = np.asarray(net.s[:, self._m0, self._n0], dtype=np.complex128)

        if self.validate:
            f_hz, (s_mn,) = _sort_freq_and_apply(f_hz, [s_mn])

        phi = np.angle(s_mn)  # radians in [-pi, pi]
        if self.unwrap:
            phi = np.unwrap(phi)
        if self.unit == "deg":
            phi = np.degrees(phi)

        freq = convert_freq_from_hz(f_hz, self.freq_unit)
        freq, phi = _apply_band(freq, phi.astype(np.float64), self.band)

        if self.validate:
            freq, phi = ensure_1d(freq, phi, who="PhaseSelector")

        return freq, phi

# =============================================================================
# 4) AxialRatioSelector — демонстрационный derived-селектор
# =============================================================================

@register_selector(("AxialRatioSelector", "axial_ratio", "ar"))
class AxialRatioSelector(BaseSelector):
    """
    Демонстрационный derived-селектор: Axial Ratio по S31/S41 (например, RHCP/LHCP).

    Селектор показан как пример добавления derived-метрик на уровне Selector-а.
    Для строгого «ядра» СВЧ-фильтров этот класс не является обязательным.

    Формула (линейная):
      C = |S31|, D = |S41|
      A = arg(S31), B = arg(S41)
      AR = |tan( 0.5 * asin( 2*C*D/(C^2 + D^2) * sin(A - B) ) )|

    Возвращает:
      - AR_lin (если db=False)
      - AR_db  = 20 log10(AR_lin) (если db=True)

    Parameters
    ----------
    band : (float, float) | None
        Диапазон частот в единицах freq_unit.
    db : bool
        Возвращать ли AR в db.
    freq_unit : str
        Единицы частоты на выходе.
    name : str | None
        Имя кривой.
    validate : bool, optional
        Включать ли проверку числа портов, сортировку оси частот и проверку
        размерностей (ensure_1d). По умолчанию True; установите False для
        «доверенных» сетей в горячих циклах оптимизации.
    """

    def __init__(
        self,
        *,
        band: Tuple[float, float] | None = None,
        db: bool = True,
        freq_unit: str = "GHz",
        name: Optional[str] = None,
        validate: bool = True,
    ):
        self.band = band
        self.db = bool(db)

        self.freq_unit = normalize_freq_unit(freq_unit)
        self.value_unit = "db" if self.db else "lin"
        self.name = name or f"AxialRatio_{'db' if self.db else 'lin'}"
        self.validate = bool(validate)

    def __call__(self, net: NetworkLike) -> Tuple[np.ndarray, np.ndarray]:
        # AR использует S31 и S41 => требуется сеть как минимум с 4 портами.
        if self.validate:
            nports = getattr(net, "nports", None)
            if nports is not None and int(nports) < 4:
                raise ValueError(
                    f"AxialRatioSelector: требуется сеть с >=4 портами, получено nports={int(nports)}"
                )

        f_hz = np.asarray(net.frequency.f, dtype=float)
        s31 = np.asarray(net.s[:, 2, 0], dtype=np.complex128)
        s41 = np.asarray(net.s[:, 3, 0], dtype=np.complex128)

        # Сортируем частоту один раз и применяем порядок к обоим каналам.
        if self.validate:
            f_hz, (s31, s41) = _sort_freq_and_apply(f_hz, [s31, s41])

        freq = convert_freq_from_hz(f_hz, self.freq_unit)

        C, D = np.abs(s31), np.abs(s41)
        A, B = np.angle(s31), np.angle(s41)

        num = 2.0 * C * D * np.sin(A - B)
        den = (C * C + D * D)

        with np.errstate(divide="ignore", invalid="ignore"):
            x = num / den

        # Домен arcsin: [-1, 1]
        x = np.clip(x, -1.0, 1.0)

        with np.errstate(invalid="ignore"):
            ar_lin = np.abs(np.tan(0.5 * np.arcsin(x)))

        if self.db:
            # Логарифм от нуля даёт -inf; это допускается как диагностическая информация.
            # При необходимости политика обработки задаётся Transform-ом (FiniteTransform).
            vals = 20.0 * np.log10(ar_lin)
        else:
            vals = ar_lin

        # Здесь сознательно НЕ удаляем -inf в db (это тоже валидная информация).
        # Фильтруем только явные NaN по частоте/значениям; Inf оставляем как есть.
        ok = np.isfinite(freq) & ~np.isnan(vals)
        freq = np.asarray(freq[ok], dtype=float)
        vals = np.asarray(vals[ok], dtype=np.float64)

        freq, vals = _apply_band(freq, vals, self.band)
        if self.validate:
            freq, vals = ensure_1d(freq, vals, who="AxialRatioSelector")

        return freq, vals


# =============================================================================
# Публичный экспорт
# =============================================================================

__all__ = [
    "SComplexSelector",
    "SMagSelector",
    "PhaseSelector",
    "AxialRatioSelector",
]
