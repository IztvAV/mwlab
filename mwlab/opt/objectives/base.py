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

на основе S-параметров (`skrf.Network`) и производных характеристик
(АЧХ, ФЧХ, ГВЗ, производные, «крутизна», интегральные метрики и т.п.).

Ключевая идея: ортогональные компоненты
--------------------------------------
Архитектура строится на композиции независимых компонентов:

1) **Selector**    — извлекает из `skrf.Network` частотную зависимость: `(freq, values)`
2) **Transform**   — преобразует кривую: `(freq, vals) -> (freq2, vals2)`
3) **Aggregator**  — сворачивает кривую в скаляр: `(freq, vals) -> float`
4) **Comparator**  — интерпретирует скаляр как pass/fail и/или штраф: `is_ok(value)`, `penalty(value)`

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
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import skrf as rf


# =============================================================================
# Частотные единицы: нормализация и конвертеры
# =============================================================================
#
# Внутри `skrf.Network` частота хранится в Гц (Hz) и доступна как net.frequency.f.
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
# Registry helpers — регистрация классов по alias
# =============================================================================

T = TypeVar("T")

def normalize_alias(alias: str) -> str:
    """
    Нормализует alias для registry:
      - trim
      - пробелы внутри -> '_'
     Важно: алиасы/типы считаем case-sensitive (канонический вид задаёт registry).
    """
    s = str(alias).strip()
    if not s:
        raise ValueError("alias не должен быть пустой строкой")
    return s.replace(" ", "_")

@dataclass
class _Registry(Generic[T]):
    """
    Внутренний реестр компонентов (Selector/Transform/Aggregator/Comparator).

    Зачем нужен отдельный класс, а не просто dict?
    ---------------------------------------------
    Для сериализации/десериализации (serde) нам важно:
      1) alias -> class  (чтобы собрать объект из YAML/JSON)
      2) class -> canonical alias (чтобы стабильно дампить обратно в YAML/JSON)
      3) class -> все алиасы (для диагностики и обратной совместимости)

    Канонический alias
    ------------------
    Каноническим считается ПЕРВЫЙ alias, указанный в декораторе @register_*().
    Например:
        @register_selector(("s_mag", "sdb", "smag"))
    => canonical = "s_mag"
    Именно его следует использовать при dump().
    """

    kind: str
    alias_to_cls: Dict[str, Type[T]] = field(default_factory=dict)
    cls_to_aliases: Dict[Type[T], Tuple[str, ...]] = field(default_factory=dict)
    cls_to_canonical: Dict[Type[T], str] = field(default_factory=dict)

    def register(self, aliases: Union[str, Iterable[str]]) -> Callable[[Type[T]], Type[T]]:
        """
        Зарегистрировать класс по одному alias или по списку alias-ов.

        Важно:
        - все alias нормализуются через normalize_alias()
        - первый alias становится каноническим (для dump)
        - конфликт alias (alias уже занят другим классом) -> ошибка при импорте
        """
        raw = [aliases] if isinstance(aliases, str) else list(aliases)
        if not raw:
            raise ValueError(f"{self.kind}: список aliases пуст")

        norm = tuple(normalize_alias(a) for a in raw)
        canonical = norm[0]

        def _wrap(cls: Type[T]) -> Type[T]:
            # 1) Проверяем коллизии alias -> другой класс
            for a in norm:
                if a in self.alias_to_cls and self.alias_to_cls[a] is not cls:
                    other = self.alias_to_cls[a]
                    raise KeyError(
                        f"{self.kind.capitalize()} alias '{a}' already exists "
                        f"(registered for {other.__name__}), cannot reuse for {cls.__name__}"
                    )

            # 2) alias -> class
            for a in norm:
                self.alias_to_cls[a] = cls

            # 3) class -> aliases (на случай повторной регистрации/расширения алиасов)
            prev = self.cls_to_aliases.get(cls, tuple())
            merged = tuple(dict.fromkeys(prev + norm))  # сохраняем порядок, убираем дубли
            self.cls_to_aliases[cls] = merged

            # 4) class -> canonical alias (фиксируем один раз)
            self.cls_to_canonical.setdefault(cls, canonical)

            # 5) Для удобства интроспекции сохраняем информацию на самом классе
            #    (совместимо с текущей практикой в mwlab).
            setattr(cls, "_aliases", list(merged))
            setattr(cls, "_canonical_alias", self.cls_to_canonical[cls])

            return cls

        return _wrap

    def resolve_cls(self, alias: str) -> Type[T]:
        """Получить класс по alias (без создания экземпляра)."""
        key = normalize_alias(alias)
        if key not in self.alias_to_cls:
            available = sorted(self.alias_to_cls.keys())
            raise KeyError(f"{self.kind.capitalize()} '{alias}' not found. Available: {available}")
        return self.alias_to_cls[key]

    def get(self, alias: str, **kw) -> T:
        """
        Создать экземпляр по alias.

        **kw передаются в конструктор класса.
        """
        cls = self.resolve_cls(alias)
        return cls(**kw)  # type: ignore[misc]

    def canonical(self, obj_or_cls: Union[T, Type[T]]) -> str:
        """Вернуть канонический alias для класса/экземпляра."""
        cls: Type[T] = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
        if cls not in self.cls_to_canonical:
            raise KeyError(f"{self.kind}: class {cls.__name__} is not registered")
        return self.cls_to_canonical[cls]

    def aliases_of(self, obj_or_cls: Union[T, Type[T]]) -> Tuple[str, ...]:
        """Вернуть все алиасы, зарегистрированные для класса/экземпляра."""
        cls: Type[T] = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
        return self.cls_to_aliases.get(cls, tuple())

    def list_aliases(self) -> List[str]:
        """Список всех известных alias (отсортированный)."""
        return sorted(self.alias_to_cls.keys())


# --- Конкретные реестры для четырёх видов компонентов ---
_SELECTOR_REG = _Registry["BaseSelector"]("selector")
_TRANSFORM_REG = _Registry["BaseTransform"]("transform")
_AGGREGATOR_REG = _Registry["BaseAggregator"]("aggregator")
_COMPARATOR_REG = _Registry["BaseComparator"]("comparator")


# --- Декораторы регистрации (совместимы с текущим API) ---
register_selector = _SELECTOR_REG.register
register_transform = _TRANSFORM_REG.register
register_aggregator = _AGGREGATOR_REG.register
register_comparator = _COMPARATOR_REG.register


# --- Фабрики создания экземпляров по alias (совместимы с текущим API) ---
def get_selector(alias: str, **kw) -> "BaseSelector":
    return _SELECTOR_REG.get(alias, **kw)


def get_transform(alias: str, **kw) -> "BaseTransform":
    return _TRANSFORM_REG.get(alias, **kw)


def get_aggregator(alias: str, **kw) -> "BaseAggregator":
    return _AGGREGATOR_REG.get(alias, **kw)


def get_comparator(alias: str, **kw) -> "BaseComparator":
    return _COMPARATOR_REG.get(alias, **kw)


def list_selectors() -> List[str]:
    return _SELECTOR_REG.list_aliases()


def list_transforms() -> List[str]:
    return _TRANSFORM_REG.list_aliases()


def list_aggregators() -> List[str]:
    return _AGGREGATOR_REG.list_aliases()


def list_comparators() -> List[str]:
    return _COMPARATOR_REG.list_aliases()


# --- Дополнительный API для serde (дамп/лоад) и диагностики ---
def resolve_type(kind: str, alias: str) -> Type[Any]:
    """
    Вернуть класс компонента по (kind, alias), не создавая экземпляр.

    kind: selector|transform|aggregator|comparator
    """
    k = str(kind).strip().lower()
    if k == "selector":
        return _SELECTOR_REG.resolve_cls(alias)
    if k == "transform":
        return _TRANSFORM_REG.resolve_cls(alias)
    if k == "aggregator":
        return _AGGREGATOR_REG.resolve_cls(alias)
    if k == "comparator":
        return _COMPARATOR_REG.resolve_cls(alias)
    raise ValueError("kind должен быть selector|transform|aggregator|comparator")


def canonical_type(kind: str, obj_or_cls: Any) -> str:
    """
    Вернуть канонический alias для дампа (стабильный `type` в YAML/JSON).
    """
    k = str(kind).strip().lower()
    if k == "selector":
        return _SELECTOR_REG.canonical(obj_or_cls)
    if k == "transform":
        return _TRANSFORM_REG.canonical(obj_or_cls)
    if k == "aggregator":
        return _AGGREGATOR_REG.canonical(obj_or_cls)
    if k == "comparator":
        return _COMPARATOR_REG.canonical(obj_or_cls)
    raise ValueError("kind должен быть selector|transform|aggregator|comparator")


def aliases_of(kind: str, obj_or_cls: Any) -> Tuple[str, ...]:
    """
    Вернуть все alias-ы, зарегистрированные для компонента.
    Полезно для отчётов, подсказок и обратной совместимости.
    """
    k = str(kind).strip().lower()
    if k == "selector":
        return _SELECTOR_REG.aliases_of(obj_or_cls)
    if k == "transform":
        return _TRANSFORM_REG.aliases_of(obj_or_cls)
    if k == "aggregator":
        return _AGGREGATOR_REG.aliases_of(obj_or_cls)
    if k == "comparator":
        return _COMPARATOR_REG.aliases_of(obj_or_cls)
    raise ValueError("kind должен быть selector|transform|aggregator|comparator")


def known_types(kind: str) -> Tuple[str, ...]:
    """
    Список всех известных `type` (alias) для данного kind.
    """
    k = str(kind).strip().lower()
    if k == "selector":
        return tuple(_SELECTOR_REG.list_aliases())
    if k == "transform":
        return tuple(_TRANSFORM_REG.list_aliases())
    if k == "aggregator":
        return tuple(_AGGREGATOR_REG.list_aliases())
    if k == "comparator":
        return tuple(_COMPARATOR_REG.list_aliases())
    raise ValueError("kind должен быть selector|transform|aggregator|comparator")



# =============================================================================
# Base-классы: Selector / Transform / Aggregator / Comparator
# =============================================================================

class BaseSelector(ABC):
    """
    Selector извлекает из `rf.Network` частотную зависимость: (freq, values).

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

    @abstractmethod
    def __call__(self, net: rf.Network) -> Tuple[np.ndarray, np.ndarray]:
        ...


ExpectedUnit = Union[str, Sequence[str], None]


class BaseTransform(ABC):
    """
    Transform преобразует кривую (freq, vals) -> (freq2, vals2).

    Transform работает только с массивами и не обращается к `rf.Network`.

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


class BaseComparator(ABC):
    """
    Comparator интерпретирует скаляр: «норма выполнена?» и/или штраф penalty().

    Comparator видит только скаляр и не зависит от частотной оси и массивов.

    Для отчётности компаратор может хранить атрибут `unit` (например, "dB"),
    который используется только для читаемости.
    """

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
        raise ValueError(
            f"{who}: несовместимые единицы частоты для {component_name}: "
            f"получено freq_unit='{fu}', ожидается одно из {list(expected_freq)}"
        )

    if expected_value is not None and vu not in expected_value:
        raise ValueError(
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
        name: str = "",
        assume_prepared: bool = False,
    ):
        self.selector = selector
        self.transform = transform
        self.agg = aggregator
        self.comp = comparator

        self.weight = float(weight)
        if self.weight < 0:
            raise ValueError("weight must be >= 0")

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
        exp_f_a = _infer_expected_freq_units_from_attr(self.agg)
        exp_v_a = _infer_expected_value_units_from_attr(self.agg)

        validate_expected_units(
            who="Criterion",
            component_name=f"Aggregator({self.agg.__class__.__name__})",
            expected_freq=exp_f_a,
            expected_value=exp_v_a,
            actual_freq_unit=self._freq_unit,
            actual_value_unit=self._value_unit_after_transform,
        )

        final_value_unit = str(
            self.agg.out_value_unit(self._value_unit_after_transform, self._freq_unit) or ""
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

    def _curve_with_units(self, net: rf.Network) -> Tuple[np.ndarray, np.ndarray, str, str]:
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

    def curve(self, net: rf.Network) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вернуть кривую после selector (+ transform, если он задан).

        Метод полезен для отладки и визуализации. Единицы можно получить через
        `evaluate()` (в составе CriterionResult) или через `_resolved_units()`.
        """
        freq, vals, _, _ = self._curve_with_units(net)
        return freq, vals

    def value(self, net: rf.Network) -> float:
        """
        Скалярное значение критерия (до сравнения с порогом).

        Примечание:
        Ожидания агрегатора по единицам проверяются при создании критерия;
        во время вычисления дополнительных проверок не выполняется.
        """
        freq, vals, freq_unit, value_unit = self._curve_with_units(net)
        with temporarily_disable_validate(self.agg, self.assume_prepared):
            return float(self.agg.aggregate(freq, vals, freq_unit=freq_unit, value_unit=value_unit))

    def _resolved_units(self) -> Tuple[str, str]:
        """
        Разрешить (freq_unit, value_unit) для отчёта.

        value_unit вычисляется последовательной протяжкой единиц через компоненты:
        selector.value_unit -> transform.out_value_unit -> aggregator.out_value_unit
        """
        return self._freq_unit, self._final_value_unit

    def evaluate(self, net: rf.Network) -> CriterionResult:
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
        ok = bool(self.comp.is_ok(v))
        raw = float(self.comp.penalty(v))
        wpen = self.weight * raw

        t_class = self.transform.__class__.__name__ if self.transform is not None else "None"
        t_chain = repr(self.transform) if self.transform is not None else "None"
        c_unit = str(getattr(self.comp, "unit", "") or "")

        return CriterionResult(
            name=self.name,
            value=v,
            ok=ok,
            raw_penalty=raw,
            weighted_penalty=wpen,
            weight=self.weight,
            freq_unit=freq_unit,
            value_unit=value_unit,
            comparator_unit=c_unit,
            selector=self.selector.__class__.__name__,
            transform=t_class,
            aggregator=self.agg.__class__.__name__,
            comparator=self.comp.__class__.__name__,
            transform_chain=t_chain,
        )

    def is_ok(self, net: rf.Network) -> bool:
        v = float(self.value(net))
        return bool(self.comp.is_ok(v))

    def penalty(self, net: rf.Network) -> float:
        v = float(self.value(net))
        raw = float(self.comp.penalty(v))
        return self.weight * raw

    def __repr__(self):  # pragma: no cover
        return f"Criterion({self.name})"


# =============================================================================
# Specification — AND-набор критериев
# =============================================================================

Reduction = Literal["sum", "mean", "max"]


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
    """

    def __init__(self, criteria: Sequence[BaseCriterion], *, name: str = "spec"):
        self.criteria: Tuple[BaseCriterion, ...] = tuple(criteria)
        self.name = str(name)

    def evaluate(self, net: rf.Network) -> List[CriterionResult]:
        return [c.evaluate(net) for c in self.criteria]

    def is_ok(self, net: rf.Network) -> bool:
        return all(c.is_ok(net) for c in self.criteria)

    def penalty(self, net: rf.Network, *, reduction: Reduction = "sum") -> float:
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
    # unit helpers (public, shared across modules)
    "normalize_value_unit",
    "parse_expected_freq_units",
    "parse_expected_value_units",
    "validate_expected_units",
    # empty curve helper
    "OnEmpty",
    "norm_on_empty",
    "handle_empty_curve",
    "temporarily_disable_validate",
    # registry
    "normalize_alias",
    "register_selector",
    "register_transform",
    "register_aggregator",
    "register_comparator",
    "resolve_type",
    "canonical_type",
    "aliases_of",
    "known_types",
    "get_selector",
    "get_transform",
    "get_aggregator",
    "get_comparator",
    "list_selectors",
    "list_transforms",
    "list_aggregators",
    "list_comparators",
    # bases
    "BaseSelector",
    "BaseTransform",
    "BaseAggregator",
    "BaseComparator",
    "BaseCriterion",
    "CriterionResult",
    "BaseSpecification",
]
