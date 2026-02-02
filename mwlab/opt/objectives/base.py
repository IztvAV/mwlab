# mwlab/opt/objectives/base.py
"""
mwlab.opt.objectives.base
=========================

Базовая архитектура подсистемы целей/ограничений (objectives) библиотеки **mwlab**.

Назначение
----------
Подсистема предназначена для формализации требований к СВЧ-устройствам и расчёта:

- **ограничений** (constraints): проверка выполнения требований (pass/fail),
- **целевых функций** (objectives): вычисление штрафов/метрик для оптимизации,

на основе S-параметров (`NetworkLike`, совместимый с `skrf.Network`) и производных характеристик
(АЧХ, ФЧХ, ГВЗ, производные, «крутизна», интегральные метрики и т.п.).

Ключевая идея: ортогональные компоненты
--------------------------------------
Архитектура строится на композиции независимых компонентов:

1) **Selector**    — извлекает из `NetworkLike` частотную зависимость: `(freq, values)`
2) **Transform**   — преобразует кривую: `(freq, vals) -> (freq2, vals2)`
3) **Aggregator**  — сворачивает кривую в скаляр: `(freq, vals) -> float`
4) **Comparator**  — интерпретирует скаляр как pass/fail и вычисляет метрики:
                     - `penalty(value)` : неотрицательный штраф за нарушение,
                     - `reward(value)`  : неотрицательное вознаграждение за запас (опционально).

Композиция:
    Selector ∘ Transform ∘ Aggregator ∘ Comparator
образует **Criterion** (критерий).

Набор Criterion, объединённый логическим AND, образует **Specification** (спецификацию).

Единицы частоты и контекст единиц
--------------------------------
Часть производных метрик физически зависит от единиц частоты (например, ГВЗ определяется через df в Гц).
Чтобы избежать «тихих» ошибок масштаба, вычислительный проход критерия передаёт компонентам контекст:

- `freq_unit`  — единицы оси частоты, используемые в цепочке,
- `value_unit` — единицы (или семантика) значений текущей кривой.

По умолчанию Transform/Aggregator игнорируют контекст и ведут себя так же, как `__call__`,
но компоненты, которым важно знать единицы, могут переопределить:

- `BaseTransform.apply(freq, vals, *, freq_unit, value_unit)`
- `BaseAggregator.aggregate(freq, vals, *, freq_unit, value_unit)`

Валидация совместимости единиц
------------------------------
Чтобы исключить «молчаливое» неверное масштабирование, Criterion выполняет строгую проверку
совместимости единиц на границе компонентов. Компонент может объявить ожидания:

- `expects_freq_unit`  (строка или набор строк частотных единиц)
- `expects_value_unit` (строка или набор строк единиц/семантики значений)

Также поддерживаются распространённые атрибуты для ожиданий:
- `freq_unit`  (если значение похоже на частотную единицу)
- `phase_unit` (для трансформов, работающих с фазой)

Registry-pattern (авторегистрация по alias)
-------------------------------------------
Компоненты могут регистрироваться по alias для сборки пайплайнов из JSON/YAML и интроспекции.

Утилиты для 1-D кривых
----------------------
Модуль содержит единый набор низкоуровневых утилит:
- приведение к 1-D и проверка длины,
- сортировка по частоте,
- линейная интерполяция (real/complex),
- стандартизация dtype,
- единая семантика обработки пустой кривой у Transform-ов.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import contextmanager
from functools import lru_cache
import inspect
from typing import (
    Any,
    Dict,
    Iterable,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .network_like import NetworkLike

# ---------------------------------------------------------------------
# Serde-critical exception: structural unit mismatch
# ---------------------------------------------------------------------
class UnitMismatchError(ValueError):
    """
    Структурная ошибка несовместимости единиц в цепочке компонентов.

    ВАЖНО:
    - Наследуется от ValueError (обратная совместимость).
    - Используется serde для надёжной классификации ошибки как UnitMismatch.
    """
    kind: str = "UnitMismatch"
    __mwlab_kind__: str = "UnitMismatch"


# =============================================================================
# Частотные единицы: нормализация и конвертеры
# =============================================================================
#
# Внутри `NetworkLike` частота хранится в Гц (Hz) и доступна как net.frequency.f.
# В подсистеме objectives допускаются инженерные единицы: Hz/kHz/MHz/GHz.
#

_FREQ_SCALE_FROM_HZ: Dict[str, float] = {
    "Hz": 1.0,
    "kHz": 1e-3,
    "MHz": 1e-6,
    "GHz": 1e-9,
}

_FREQ_UNIT_CANONICAL: Dict[str, str] = {
    "hz": "Hz",
    "khz": "kHz",
    "mhz": "MHz",
    "ghz": "GHz",
}


def normalize_freq_unit(unit: str) -> str:
    """
    Нормализовать строку единиц частоты до канонического вида.

    Поддерживаются: "Hz", "kHz", "MHz", "GHz" (без чувствительности к регистру).
    """
    u = str(unit).strip()
    if not u:
        raise ValueError("freq_unit не должен быть пустой строкой")
    key = u.lower()
    if key not in _FREQ_UNIT_CANONICAL:
        supported = list(_FREQ_UNIT_CANONICAL.values())
        raise ValueError(f"Unsupported freq unit '{unit}'. Supported: {supported}")
    return _FREQ_UNIT_CANONICAL[key]


def convert_freq_from_hz(freq_hz: np.ndarray, unit: str) -> np.ndarray:
    """
    Преобразовать массив частот из Гц в заданные единицы.

    Возвращает массив float64.
    """
    u = normalize_freq_unit(unit)
    return np.asarray(freq_hz, dtype=float) * _FREQ_SCALE_FROM_HZ[u]


def freq_unit_scale_to_hz(unit: str) -> float:
    """
    Возвращает множитель для перевода частоты из `unit` в Гц:

        f_hz = f_unit * scale_to_hz(unit)
    """
    u = normalize_freq_unit(unit)
    return 1.0 / _FREQ_SCALE_FROM_HZ[u]


# =============================================================================
# Утилиты для 1-D кривых: проверки, сортировка, dtype, интерполяция
# =============================================================================

def dtype_out(vals: np.ndarray) -> np.dtype:
    """
    Стандартизация dtype:

    - вещественные данные -> float64
    - комплексные данные  -> complex128
    """
    return np.dtype(np.complex128) if np.iscomplexobj(vals) else np.dtype(np.float64)


def ensure_1d(freq: np.ndarray, vals: np.ndarray, who: str = "curve") -> Tuple[np.ndarray, np.ndarray]:
    """
    Привести вход к 1-D массивам и проверить согласованность длины.
    """
    f = np.asarray(freq)
    y = np.asarray(vals)
    if f.ndim != 1 or y.ndim != 1:
        raise ValueError(f"{who}: ожидаются 1-D массивы freq и vals")
    if f.size != y.size:
        raise ValueError(f"{who}: freq и vals должны иметь одинаковую длину")
    return f, y

def sort_by_freq(
    freq: np.ndarray,
    vals: np.ndarray,
    *,
    return_idx: bool = False,
):
    """
    Обеспечить монотонно возрастающую ось частоты.

    Поведение:
    - если freq возрастает: возвращается как есть;
    - если freq убывает: массивы разворачиваются;
    - если freq не монотонна: выполняется сортировка по freq.

    Проверка finite-ности freq сознательно не выполняется:
    политику обработки NaN/Inf следует задавать на уровне Transform/Aggregator.
    """
    f = np.asarray(freq, dtype=float)
    y = np.asarray(vals)

    if f.size <= 1:
        if return_idx:
            return f, y, np.arange(f.size, dtype=int)
        return f, y

    df = np.diff(f)
    if np.all(df >= 0):
        if return_idx:
            return f, y, np.arange(f.size, dtype=int)
        return f, y

    if np.all(df <= 0):
        idx = np.arange(f.size - 1, -1, -1, dtype=int)
        if return_idx:
            return f[::-1], y[::-1], idx
        return f[::-1], y[::-1]

    idx = np.argsort(f)
    if return_idx:
        return f[idx], y[idx], idx
    return f[idx], y[idx]


def interp_linear(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    Линейная интерполяция y(x) в точках xq.

    Поддерживается:
    - вещественный y
    - комплексный y (интерполяция Re/Im отдельно)

    Требования:
    - x должен быть отсортирован по возрастанию.
    - xq предполагается внутри [x[0], x[-1]].
    """
    x = np.asarray(x, dtype=float)
    xq = np.asarray(xq, dtype=float)
    y = np.asarray(y)

    if y.size == 0:
        return np.asarray([], dtype=dtype_out(y))

    if np.iscomplexobj(y):
        yr = np.interp(xq, x, np.real(y).astype(float))
        yi = np.interp(xq, x, np.imag(y).astype(float))
        return (yr + 1j * yi).astype(np.complex128)

    return np.interp(xq, x, y.astype(float)).astype(np.float64)

def interp_at_scalar(x: np.ndarray, y: np.ndarray, x0: float, *, atol: float = 0.0, rtol: float = 0.0) -> Union[float, complex]:
    """
    Интерполировать y(x) в одной точке x0 (линейно).

    Экстраполяция запрещена: если x0 вне диапазона [x[0], x[-1]] — ValueError.

    Численная устойчивость на границах
    ---------------------------------
    В реальных пайплайнах x0 может отличаться от границы диапазона на величину
    порядка машинного эпсилона (например, после конверсии единиц и клипования).
    Чтобы не получать «ложные» ошибки, допускается мягкая коррекция x0 на границу,
    если x0 близко к x[0]/x[-1] согласно atol/rtol.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("interp_at_scalar: пустой массив x")
    x0 = float(x0)

    x_lo = float(x[0])
    x_hi = float(x[-1])

    if x0 < x_lo and np.isclose(x0, x_lo, atol=float(atol), rtol=float(rtol)):
        x0 = x_lo
    elif x0 > x_hi and np.isclose(x0, x_hi, atol=float(atol), rtol=float(rtol)):
        x0 = x_hi
    elif x0 < x_lo or x0 > x_hi:
        raise ValueError(f"interp_at_scalar: x0={x0} вне диапазона [{x_lo}, {x_hi}]")

    v = interp_linear(x, y, np.asarray([x0], dtype=float))
    return v[0].item()

# =============================================================================
# Empty-curve policy helper (единая семантика пустого результата Transform-ов)
# =============================================================================

OnEmpty = Literal["raise", "ok"]

def norm_on_empty(on_empty: str, who: str) -> OnEmpty:
    """
    Нормализация политики on_empty и единая диагностика.
    """
    v = str(on_empty).strip().lower()
    if v not in ("raise", "ok"):
        raise ValueError(f"{who}: on_empty должен быть 'raise' или 'ok'")
    return v  # type: ignore[return-value]


def handle_empty_curve(
    on_empty: OnEmpty,
    who: str,
    *,
    y_dtype: np.dtype = np.dtype(np.float64),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Унифицированная обработка пустой кривой для Transform-ов.

    on_empty:
      - "raise": ValueError
      - "ok"   : вернуть пустые массивы (freq=float64, vals=y_dtype)
    """
    mode = str(on_empty).strip().lower()
    if mode not in ("raise", "ok"):
        raise ValueError(f"{who}: on_empty must be 'raise'|'ok'")

    if mode == "raise":
        raise ValueError(f"{who}: empty curve")

    return np.asarray([], dtype=float), np.asarray([], dtype=np.dtype(y_dtype))


@contextmanager
def temporarily_disable_validate(
    components: Union[object, Iterable[object]],
    assume_prepared: bool,
):
    """
    Временно отключить validate у компонента(ов), если assume_prepared=True.
    """
    if not assume_prepared:
        yield
        return

    items: Iterable[object]
    if isinstance(components, (list, tuple, set)):
        items = components
    else:
        items = (components,)

    prev: List[Tuple[object, object]] = []
    for component in items:
        # Отключаем только если validate — именно булев флаг.
        # Это защищает от случаев, когда validate является методом или property без setter.
        val = getattr(component, "validate", None)
        if isinstance(val, (bool, np.bool_)):
            prev.append((component, val))
            try:
                setattr(component, "validate", False)
            except Exception:
                # Если атрибут нельзя перезаписать (например, read-only property),
                # не ломаем вычисление.
                prev.pop()
    try:
        yield
    finally:
        for component, value in prev:
            setattr(component, "validate", value)


# =============================================================================
# Serde helpers (MVP-friendly): introspection + params extraction
# =============================================================================

@lru_cache(maxsize=256)
def constructor_kw_params(cls: type) -> Tuple[str, ...]:
    """
    Вернуть имена параметров конструктора (__init__), которые допустимы как kwargs.

    Используется serde-лоадером для строгой проверки params:
    - unknown params -> ошибка
    - missing required params -> ошибка

    Правила:
    - исключаем 'self'
    - исключаем *args/**kwargs (VAR_POSITIONAL/VAR_KEYWORD)
    - positional-only параметры считаются НЕ поддержанными для serde (игнорируются)
      (в mwlab рекомендуется использовать keyword-friendly __init__).
    """
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return tuple()

    names: List[str] = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            # Для serde мы не поддерживаем positional-only параметры.
            continue
        names.append(p.name)
    return tuple(names)


def _serde_params_from_obj(obj: object) -> Dict[str, Any]:
    """
    Базовая стратегия получения параметров для dump.

    Приоритет:
    1) obj.__serde_fields__ (если задано явно)
    2) constructor_kw_params(obj.__class__) (по подписи __init__)

    Переименования:
    - obj.__serde_rename__ : dict[param_name -> attr_name]

    Исключения:
    - obj.__serde_exclude__ : iterable[attr_name or param_name]

    Если параметр из списка ожидается, но атрибут отсутствует -> ValueError,
    чтобы разработчик компонента явно зафиксировал mapping (или сохранил атрибут).
    """
    cls = obj.__class__

    fields = getattr(obj, "__serde_fields__", None)
    if fields is None:
        names = list(constructor_kw_params(cls))
    else:
        names = [str(x) for x in fields]

    rename: Dict[str, str] = dict(getattr(obj, "__serde_rename__", {}) or {})
    exclude = set(getattr(obj, "__serde_exclude__", ()) or ())

    out: Dict[str, Any] = {}
    for name in names:
        if name in exclude:
            continue
        attr = rename.get(name, name)
        if attr in exclude:
            continue
        try:
            out[name] = getattr(obj, attr)
        except AttributeError as e:
            raise ValueError(
                f"{cls.__name__}: невозможно сформировать serde_params: "
                f"не найден атрибут '{attr}' для параметра '{name}'. "
                f"Сохраните параметр как self.{attr} или задайте __serde_rename__/__serde_fields__."
            ) from e
    return out

# =============================================================================
# Base-классы: Selector / Transform / Aggregator / Comparator
# =============================================================================

class BaseSelector(ABC):
    """
    Selector извлекает из `NetworkLike` частотную зависимость: (freq, values).

    Соглашения:
    - freq   : 1-D массив частот в единицах `self.freq_unit`
    - values : 1-D массив значений той же длины

    Метаданные:
    - name       : человекочитаемое имя кривой
    - freq_unit  : единицы частоты на выходе (по умолчанию "GHz")
    - value_unit : единицы/семантика values (например, "dB", "rad", "ns", "lin")
    """

    name: str = ""
    freq_unit: str = "GHz"
    value_unit: str = ""

    # serde metadata (используется dumper'ом/loader'ом)
    __mwlab_kind__: str = "selector"
    __serde_fields__: Optional[Tuple[str, ...]] = None
    __serde_rename__: Dict[str, str] = {}
    __serde_exclude__: Tuple[str, ...] = ()

    @abstractmethod
    def __call__(self, net: NetworkLike) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def serde_params(self) -> Dict[str, Any]:
        """Параметры для сериализации (dict из python-примитивов / JSON-friendly значений)."""
        return _serde_params_from_obj(self)


ExpectedUnit = Union[str, Sequence[str], None]

class BaseTransform(ABC):
    """
    Transform преобразует кривую (freq, vals) -> (freq2, vals2).

    Transform работает только с массивами и не обращается к `NetworkLike`.

    Контекст единиц
    --------------
    Для операций, зависящих от единиц (например, физически корректные производные),
    Transform может переопределить метод:

        apply(freq, vals, *, freq_unit: str, value_unit: str)

    По умолчанию `apply()` игнорирует контекст и вызывает `__call__()`.

    Ожидания по единицам
    --------------------
    Transform может объявить ожидания для входных единиц, чтобы Criterion мог
    выполнить строгую проверку до вычислений:

    - expects_freq_unit  : str | Sequence[str] | None
    - expects_value_unit : str | Sequence[str] | None

    Также допускаются распространённые атрибуты:
    - freq_unit  (если это частотная единица из {Hz,kHz,MHz,GHz})
    - phase_unit (например, "rad"/"deg")
    """

    name: str = ""
    expects_freq_unit: ExpectedUnit = None
    expects_value_unit: ExpectedUnit = None

    __mwlab_kind__: str = "transform"
    __serde_fields__: Optional[Tuple[str, ...]] = None
    __serde_rename__: Dict[str, str] = {}
    __serde_exclude__: Tuple[str, ...] = ()

    @abstractmethod
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def apply(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнить преобразование с учётом контекста единиц.

        Базовая реализация не использует контекст и вызывает `__call__()`.
        """
        _ = freq_unit
        _ = value_unit
        return self(freq, vals)

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        """
        Вернуть единицы values на выходе Transform-а.

        По умолчанию Transform не меняет единицы values.
        """
        _ = freq_unit
        return str(in_unit)

    def iter_transforms(self) -> Iterable["BaseTransform"]:
        """
        Итератор по внутренней цепочке трансформов.

        По умолчанию Transform считается одиночным и возвращает себя.
        ComposeTransform должен переопределить и отдавать элементы цепочки.
        """
        yield self

    def serde_params(self) -> Dict[str, Any]:
        return _serde_params_from_obj(self)


class BaseAggregator(ABC):
    """
    Aggregator сворачивает 1-D кривую → скаляр.

    Контекст единиц
    --------------
    Агрегатор может зависеть от единиц частоты (например, нормировки по ширине полосы).
    Для таких случаев агрегатор может переопределить:

        aggregate(freq, vals, *, freq_unit: str, value_unit: str)

    По умолчанию `aggregate()` игнорирует контекст и вызывает `__call__()`.

    Ожидания по единицам
    --------------------
    При необходимости агрегатор может объявить:

    - expects_freq_unit
    - expects_value_unit
    """
    expects_freq_unit: ExpectedUnit = None
    expects_value_unit: ExpectedUnit = None

    __mwlab_kind__: str = "aggregator"
    __serde_fields__: Optional[Tuple[str, ...]] = None
    __serde_rename__: Dict[str, str] = {}
    __serde_exclude__: Tuple[str, ...] = ()

    @abstractmethod
    def __call__(self, freq: np.ndarray, vals: np.ndarray) -> float:
        ...

    def aggregate(
        self,
        freq: np.ndarray,
        vals: np.ndarray,
        *,
        freq_unit: str,
        value_unit: str,
    ) -> float:
        """
        Свернуть кривую в скаляр с учётом контекста единиц.

        Базовая реализация не использует контекст и вызывает `__call__()`.
        """
        _ = freq_unit
        _ = value_unit
        return float(self(freq, vals))

    def out_value_unit(self, in_unit: str, freq_unit: str) -> str:
        """
        Вернуть единицы результата агрегатора.

        По умолчанию предполагается, что агрегатор не меняет размерность
        и возвращает величину в тех же единицах, что входной values.
        """
        _ = freq_unit
        return str(in_unit or "")

    def serde_params(self) -> Dict[str, Any]:
        return _serde_params_from_obj(self)


class BaseComparator(ABC):
    """
    Comparator интерпретирует скаляр: «норма выполнена?» и/или штраф penalty().

    Comparator видит только скаляр и не зависит от частотной оси и массивов.

    Для отчётности компаратор может хранить атрибут `unit` (например, "dB"),
    который используется только для читаемости.
    """

    __mwlab_kind__: str = "comparator"
    __serde_fields__: Optional[Tuple[str, ...]] = None
    __serde_rename__: Dict[str, str] = {}
    __serde_exclude__: Tuple[str, ...] = ()

    @abstractmethod
    def is_ok(self, value: float) -> bool:
        ...

    def penalty(self, value: float) -> float:
        """
        Возвращает неотрицательный штраф.

        По умолчанию — бинарная ступень:
          0.0, если is_ok(value) == True
          1.0, иначе
        """
        return 0.0 if self.is_ok(value) else 1.0

    def reward(self, value: float) -> float:
        """
        Вознаграждение за запас (>= 0).

        По умолчанию компаратор не начисляет reward и возвращает 0.0.
        Конкретные компараторы могут переопределять reward и делать его нормированным.
        """
        _ = value
        return 0.0

    def serde_params(self) -> Dict[str, Any]:
        return _serde_params_from_obj(self)


# =============================================================================
# Результат вычисления критерия (для отчётов, логов и отладки)
# =============================================================================

@dataclass(frozen=True)
class CriterionResult:
    """
    Структурированный результат вычисления одного критерия.

    Поля единиц:
    - freq_unit        : единицы частоты (ось X), используемые в цепочке
    - value_unit       : единицы итогового значения (после Transform и Aggregator)
    - comparator_unit  : единицы требования (если компаратор их хранит)

    Поля цепочки:
    - selector/transform/aggregator/comparator : имена классов
    - transform_chain : подробное представление Transform-а (repr), удобно для Compose
    """
    name: str
    value: float
    ok: bool
    raw_penalty: float
    weighted_penalty: float
    weight: float
    raw_reward: float
    weighted_reward: float
    reward_weight: float
    freq_unit: str
    value_unit: str

    comparator_unit: str = ""

    selector: str = ""
    transform: str = ""
    aggregator: str = ""
    comparator: str = ""

    transform_chain: str = ""


# =============================================================================
# Внутренние утилиты: ожидания единиц и проверка совместимости
# =============================================================================

def normalize_value_unit(u: str) -> str:
    """
    Нормализация "единиц/семантики" значений.

    Value-unit в mwlab — строковый ярлык для отчётности и валидации.
    Поэтому здесь выполняется только мягкая нормализация:
    - trim
    - lower
    """
    return str(u or "").strip().lower()


def parse_expected_freq_units(x: ExpectedUnit) -> Optional[Tuple[str, ...]]:
    """
    Преобразовать ожидание единиц частоты к кортежу канонических значений.

    Возвращает None, если ожидание не задано.
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        return (normalize_freq_unit(s),)
    # Iterable[str]
    items = [str(it).strip() for it in x]
    items = [it for it in items if it]
    if not items:
        return None
    out: List[str] = []
    for it in items:
        try:
            out.append(normalize_freq_unit(it))
        except ValueError:
            # Если разработчик компонента передал не частотную строку,
            # считаем, что это не ожидание частотных единиц.
            return None
    return tuple(sorted(set(out)))


def parse_expected_value_units(x: ExpectedUnit) -> Optional[Tuple[str, ...]]:
    """
    Преобразовать ожидание единиц/семантики значений к кортежу нормализованных строк.

    Возвращает None, если ожидание не задано.
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = normalize_value_unit(x)
        return (s,) if s else None
    items = [normalize_value_unit(it) for it in x]
    items = [it for it in items if it]
    return tuple(sorted(set(items))) if items else None


def _infer_expected_freq_units_from_attr(obj: object) -> Optional[Tuple[str, ...]]:
    """
    Попытаться извлечь ожидание частотных единиц из распространённых атрибутов.

    Поддержка:
    - expects_freq_unit
    - freq_unit (если похоже на Hz/kHz/MHz/GHz)
    """
    if hasattr(obj, "expects_freq_unit"):
        return parse_expected_freq_units(getattr(obj, "expects_freq_unit"))
    if hasattr(obj, "freq_unit"):
        try:
            return parse_expected_freq_units(str(getattr(obj, "freq_unit")))
        except Exception:
            return None
    return None


def _infer_expected_value_units_from_attr(obj: object) -> Optional[Tuple[str, ...]]:
    """
    Попытаться извлечь ожидание value_unit из распространённых атрибутов.

    Поддержка:
    - expects_value_unit
    - phase_unit (типичные значения: "rad", "deg")
    """
    if hasattr(obj, "expects_value_unit"):
        return parse_expected_value_units(getattr(obj, "expects_value_unit"))
    if hasattr(obj, "phase_unit"):
        return parse_expected_value_units(str(getattr(obj, "phase_unit")))
    return None


def validate_expected_units(
    *,
    who: str,
    component_name: str,
    expected_freq: Optional[Tuple[str, ...]],
    expected_value: Optional[Tuple[str, ...]],
    actual_freq_unit: str,
    actual_value_unit: str,
) -> None:
    """
    Проверить соответствие фактических единиц ожиданиям компонента.

    Валидация выполняется строго: несовпадение -> ValueError с понятной диагностикой.
    """
    fu = normalize_freq_unit(actual_freq_unit)
    vu = normalize_value_unit(actual_value_unit)

    if expected_freq is not None and fu not in expected_freq:
        raise UnitMismatchError(
            f"{who}: несовместимые единицы частоты для {component_name}: "
            f"получено freq_unit='{fu}', ожидается одно из {list(expected_freq)}"
        )

    if expected_value is not None and vu not in expected_value:
        raise UnitMismatchError(
            f"{who}: несовместимые единицы/семантика значений для {component_name}: "
            f"получено value_unit='{actual_value_unit}', ожидается одно из {list(expected_value)}"
        )


# =============================================================================
# Criterion = Selector ∘ Transform ∘ Aggregator ∘ Comparator
# =============================================================================

class BaseCriterion:
    """
    Criterion инкапсулирует цепочку:
        selector   -> (freq, vals)
        transform  -> (freq, vals)   [опционально]
        aggregator -> scalar
        comparator -> ok/penalty

    Вычислительный проход:
    - поддерживает контекст единиц (freq_unit/value_unit),
    - выполняет строгую валидацию ожиданий по единицам на границах компонентов,
    - возвращает CriterionResult для отчётности и диагностики.

    Валидация совместимости единиц выполняется один раз при создании экземпляра Criterion и не влияет на скорость последующих вычислений.

    Параметр assume_prepared=True отключает повторные проверки/сортировки
    в Transform/Aggregator с атрибутом validate, предполагая что Selector
    уже подготовил кривую (1-D, sorted).
    """

    def __init__(
        self,
        selector: BaseSelector,
        aggregator: BaseAggregator,
        comparator: BaseComparator,
        *,
        transform: Optional[BaseTransform] = None,
        weight: float = 1.0,
        reward_weight: float = 1.0,
        name: str = "",
        assume_prepared: bool = False,
    ):
        self.selector = selector
        self.transform = transform
        self.aggregator = aggregator
        self.comparator = comparator

        self.weight = float(weight)
        if self.weight < 0:
            raise ValueError("weight must be >= 0")

        self.reward_weight = float(reward_weight)
        if self.reward_weight < 0:
            raise ValueError("reward_weight must be >= 0")

        # ------------------------------------------------------------------
        # Auto-disable reward if comparator doesn't support it.
        #
        # Support is detected structurally: comparator overrides BaseComparator.reward.
        # If not overridden, reward() is always 0.0 → such criteria must NOT participate
        # in min/softmin aggregation, otherwise they "kill" the global reward.
        #
        # You can still force-disable reward explicitly with reward_weight=0.
        # ------------------------------------------------------------------
        try:
            reward_overridden = (self.comparator.__class__.reward is not BaseComparator.reward)
        except Exception:
            # Conservative fallback: if introspection fails, do not auto-disable
            reward_overridden = True

        if (self.reward_weight > 0.0) and (not reward_overridden):
            self.reward_weight = 0.0

        self.name = name or getattr(selector, "name", "crit")
        self.assume_prepared = bool(assume_prepared)

        #Получаем начальные единицы от селектора
        freq_unit, value_unit = self._initial_units()
        self._freq_unit = freq_unit
        self._selector_value_unit = value_unit

        #Обрабатываем Transform (если есть)
        if self.transform is not None:
            def _iter_transforms(tr: BaseTransform):
                fn_iter = getattr(tr, "iter_transforms", None)
                if callable(fn_iter):
                    yield from fn_iter()
                else:
                    yield tr

            current_value_unit = self._selector_value_unit
            for tr in _iter_transforms(self.transform):
                exp_f_t = _infer_expected_freq_units_from_attr(tr)
                exp_v_t = _infer_expected_value_units_from_attr(tr)

                validate_expected_units(
                    who="Criterion",
                    component_name=f"Transform({tr.__class__.__name__})",
                    expected_freq=exp_f_t,
                    expected_value=exp_v_t,
                    actual_freq_unit=self._freq_unit,
                    actual_value_unit=current_value_unit,
                )

                current_value_unit = str(tr.out_value_unit(current_value_unit, self._freq_unit) or "")

            value_unit_after_transform = str(
                self.transform.out_value_unit(self._selector_value_unit, self._freq_unit) or ""
            )
        else:
            value_unit_after_transform = self._selector_value_unit

        self._value_unit_after_transform = value_unit_after_transform

        #Обрабатываем Aggregator (один раз)
        exp_f_a = _infer_expected_freq_units_from_attr(self.aggregator)
        exp_v_a = _infer_expected_value_units_from_attr(self.aggregator)

        validate_expected_units(
            who="Criterion",
            component_name=f"Aggregator({self.aggregator.__class__.__name__})",
            expected_freq=exp_f_a,
            expected_value=exp_v_a,
            actual_freq_unit=self._freq_unit,
            actual_value_unit=self._value_unit_after_transform,
        )

        final_value_unit = str(
            self.aggregator.out_value_unit(self._value_unit_after_transform, self._freq_unit) or ""
        )
        self._final_value_unit = final_value_unit

    def _initial_units(self) -> Tuple[str, str]:
        """
        Получить начальные единицы (freq_unit, value_unit) сразу после Selector.

        freq_unit нормализуется до канонического вида.
        value_unit — строковый ярлык (может быть пустым).
        """
        fu_raw = getattr(self.selector, "freq_unit", "GHz")
        fu = normalize_freq_unit(str(fu_raw))
        vu = str(getattr(self.selector, "value_unit", "") or "")
        return fu, vu

    def _curve_with_units(self, net: NetworkLike) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
            Вернуть кривую после selector (+ transform, если он задан) и текущие единицы.
        """
        freq, vals = self.selector(net)

        if self.transform is not None:
            with temporarily_disable_validate(self.transform, self.assume_prepared):
                freq, vals = self.transform.apply(
                    freq,
                    vals,
                    freq_unit=self._freq_unit,
                    value_unit=self._selector_value_unit,
                )

        return freq, vals, self._freq_unit, self._value_unit_after_transform

    def curve(self, net: NetworkLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вернуть кривую после selector (+ transform, если он задан).

        Метод полезен для отладки и визуализации. Единицы можно получить через
        `evaluate()` (в составе CriterionResult) или через `_resolved_units()`.
        """
        freq, vals, _, _ = self._curve_with_units(net)
        return freq, vals

    def value(self, net: NetworkLike) -> float:
        """
        Скалярное значение критерия (до сравнения с порогом).

        Примечание:
        Ожидания агрегатора по единицам проверяются при создании критерия;
        во время вычисления дополнительных проверок не выполняется.
        """
        freq, vals, freq_unit, value_unit = self._curve_with_units(net)
        with temporarily_disable_validate(self.aggregator, self.assume_prepared):
            return float(self.aggregator.aggregate(freq, vals, freq_unit=freq_unit, value_unit=value_unit))

    def _resolved_units(self) -> Tuple[str, str]:
        """
        Разрешить (freq_unit, value_unit) для отчёта.

        value_unit вычисляется последовательной протяжкой единиц через компоненты:
        selector.value_unit -> transform.out_value_unit -> aggregator.out_value_unit
        """
        return self._freq_unit, self._final_value_unit

    def evaluate(self, net: NetworkLike) -> CriterionResult:
        """
        Единый проход:
        1) посчитать value
        2) проверить is_ok
        3) посчитать raw penalty
        4) учесть weight

        Метод предназначен для отчётности и отладки. Для быстрых вычислений штрафа
        в оптимизационных задачах рекомендуется использовать penalty(net), который
        не создаёт CriterionResult.
        """
        freq_unit, value_unit = self._resolved_units()

        v = float(self.value(net))
        ok = bool(self.comparator.is_ok(v))
        raw_p = float(self.comparator.penalty(v))
        raw_r = float(self.comparator.reward(v))
        wpen = self.weight * raw_p
        wrew = self.reward_weight * raw_r

        t_class = self.transform.__class__.__name__ if self.transform is not None else "None"
        t_chain = repr(self.transform) if self.transform is not None else "None"
        c_unit = str(getattr(self.comparator, "unit", "") or "")

        return CriterionResult(
            name=self.name,
            value=v,
            ok=ok,
            raw_penalty=raw_p,
            weighted_penalty=wpen,
            weight = self.weight,
            raw_reward = raw_r,
            weighted_reward = wrew,
            reward_weight = self.reward_weight,
            freq_unit=freq_unit,
            value_unit=value_unit,
            comparator_unit=c_unit,
            selector=self.selector.__class__.__name__,
            transform=t_class,
            aggregator=self.aggregator.__class__.__name__,
            comparator=self.comparator.__class__.__name__,
            transform_chain=t_chain,
        )

    def is_ok(self, net: NetworkLike) -> bool:
        v = float(self.value(net))
        return bool(self.comparator.is_ok(v))

    def penalty(self, net: NetworkLike) -> float:
        v = float(self.value(net))
        raw = float(self.comparator.penalty(v))
        return self.weight * raw

    def reward(self, net: NetworkLike) -> float:
        v = float(self.value(net))
        raw = float(self.comparator.reward(v))
        return self.reward_weight * raw

    def __repr__(self):  # pragma: no cover
        return f"Criterion({self.name})"


# =============================================================================
# Specification — AND-набор критериев
# =============================================================================
Reduction = Literal["sum", "mean", "max"]
RewardReduction = Literal["min", "softmin", "mean", "sum"]

class BaseSpecification:
    """
    Specification — набор критериев, объединённых логическим AND.

    is_ok(net)
      True, если все критерии выполнены.

    penalty(net)
      Сводит штрафы критериев в одну величину:
        - reduction="sum"  : сумма weighted_penalty
        - reduction="mean" : среднее по критериям
        - reduction="max"  : худший критерий (minimax)

    reward(net)
      Сводит вознаграждения критериев в одну величину:
        - reduction="min"     : минимальный weighted_reward (bottleneck / maximin)
        - reduction="softmin" : гладкая аппроксимация min (управляется tau)
        - reduction="sum"/"mean"/"max" : опционально для экспериментов/диагностики
    """

    def __init__(self, criteria: Sequence[BaseCriterion], *, name: str = "spec"):
        self.criteria: Tuple[BaseCriterion, ...] = tuple(criteria)
        self.name = str(name)

    def evaluate(self, net: NetworkLike) -> List[CriterionResult]:
        return [c.evaluate(net) for c in self.criteria]

    def is_ok(self, net: NetworkLike) -> bool:
        return all(c.is_ok(net) for c in self.criteria)

    def penalty(self, net: NetworkLike, *, reduction: Reduction = "sum") -> float:
        """
        Метод использует Criterion.penalty(net) и не создаёт CriterionResult
        для каждого критерия. Для получения подробной информации по каждому
        критерию используйте evaluate(net).
        """
        if not self.criteria:
            return 0.0

        if reduction == "sum":
            total = 0.0
            for c in self.criteria:
                total += c.penalty(net)
            return total

        if reduction == "mean":
            total = 0.0
            n = 0
            for c in self.criteria:
                total += c.penalty(net)
                n += 1
            return total / n if n else 0.0

        if reduction == "max":
            max_val = 0.0
            first = True
            for c in self.criteria:
                p = c.penalty(net)
                if first or p > max_val:
                    max_val = p
                    first = False
            return max_val if not first else 0.0

        raise ValueError("reduction must be 'sum'|'mean'|'max'")

    def reward(
        self,
        net: NetworkLike,
        *,
        reduction: RewardReduction = "min",
        tau: float = 0.1,
    ) -> float:
        """
        Сводный reward по спецификации.

        reduction:
          - "min"     : минимальный reward среди критериев (maximin по запасу)
          - "softmin" : гладкая аппроксимация min (через log-sum-exp), параметр tau>0
          - "mean"    : среднее reward
          - "sum"     : сумма reward
          - "max"     : максимальный reward среди критериев

        ВАЖНО:
        - по умолчанию критерии с reward_weight==0 исключаются из агрегации reward,
          чтобы они не «обнуляли» min/softmin.
        """
        vals: List[float] = []
        for c in self.criteria:
            rw = float(getattr(c, "reward_weight", 1.0))
            if rw <= 0.0:
                continue
            vals.append(float(c.reward(net)))

        if not vals:
            return 0.0

        red = str(reduction).strip().lower()

        if red == "sum":
            return float(np.sum(np.asarray(vals, dtype=float)))

        if red == "mean":
            return float(np.mean(np.asarray(vals, dtype=float)))

        if red == "min":
            return float(np.min(np.asarray(vals, dtype=float)))

        if red == "softmin":
            t = float(tau)
            if t <= 0.0:
                raise ValueError("tau must be > 0 for softmin")
            a = np.asarray(vals, dtype=float)
            m = float(np.min(a))
            # softmin(a) = -t*log(mean(exp(-a/t)))  (стабильная форма через сдвиг на m)
            z = np.exp(-(a - m) / t)
            return float(m - t * np.log(np.mean(z)))

        if red == "max":
            return float(np.max(np.asarray(vals, dtype=float)))

        raise ValueError("reduction must be 'min'|'softmin'|'mean'|'sum'|'max'")

    def __len__(self) -> int:  # pragma: no cover
        return len(self.criteria)

    def __iter__(self):  # pragma: no cover
        return iter(self.criteria)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Specification({self.name}, n={len(self.criteria)})"


# =============================================================================
# __all__ – публичный экспорт
# =============================================================================

__all__ = [
    # freq helpers
    "normalize_freq_unit",
    "convert_freq_from_hz",
    "freq_unit_scale_to_hz",
    # curve utils
    "dtype_out",
    "ensure_1d",
    "sort_by_freq",
    "interp_linear",
    "interp_at_scalar",
    # serde helper (public for mwlab.opt.objectives.serde)
    "constructor_kw_params",
    # unit helpers (public, shared across modules)
    "normalize_value_unit",
    "parse_expected_freq_units",
    "parse_expected_value_units",
    "validate_expected_units",
    "UnitMismatchError",
    # empty curve helper
    "OnEmpty",
    "norm_on_empty",
    "handle_empty_curve",
    "temporarily_disable_validate",
    # bases
    "BaseSelector",
    "BaseTransform",
    "BaseAggregator",
    "BaseComparator",
    "BaseCriterion",
    "CriterionResult",
    "BaseSpecification",
    "RewardReduction",
]