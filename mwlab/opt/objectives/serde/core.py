# mwlab/opt/objectives/serde/core.py
"""
mwlab.opt.objectives.serde.core
==============================

Полноценное (MVP) ядро сериализации/десериализации (serde) спецификаций mwlab
в YAML/JSON по контракту v1.

Главная идея MVP
----------------
- На входе принимаем YAML/JSON (строки/файлы) или уже готовый Python-dict.
- Строго валидируем структуру + запреты (NaN/Inf, несериализуемые типы, ключи не-строки).
- Собираем объекты через registry:
    Selector / Transform / Aggregator / Comparator
- Строим BaseCriterion (или совместимый Criterion) и Specification.
- При сборке BaseCriterion выполняется "dry" unit-aware валидация цепочки (без rf.Network),
  что соответствует контракту (UnitMismatch ловим заранее).

Ограничения MVP (осознанно)
---------------------------
- Мы не реализуем миграции версий (кроме проверки version==1).
- "diff-from-default" режим dump реализован в *простом* виде (по умолчанию explicit=True).
- Мы не делаем глубокую семантическую проверку отдельных params (например, LineSpec-покрытие частот),
  потому что это зависит от конкретных реализаций агрегаторов/трансформов.
  Но мы строго проверяем *форму* данных, запреты NaN/Inf и соответствие сигнатуре __init__.

Зависимости
-----------
- JSON: стандартная библиотека json
- YAML: PyYAML (yaml.safe_load / safe_dump). Если PyYAML не установлен — загрузка/дамп YAML недоступны.

Важно: модуль сознательно не "раздувается" — всё в одном файле.
"""

from __future__ import annotations

from dataclasses import is_dataclass, asdict
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, cast

# --- mwlab imports (ядро подсистемы objectives) ---
from ..base import BaseCriterion, constructor_kw_params, normalize_freq_unit, UnitMismatchError
from ..registry import (
    canonical_type,
    canonicalize_alias,
    resolve_type,
    suggest_types,
    has_type,
)
from ..specification import Specification  # ожидается высокоуровневый класс спецификации

# --- serde errors ---
from .errors import (
    SerdeError,
    SerdePath,
    SchemaError,
    UnknownType,
    UnknownParam,
    MissingParam,
    InvalidValue,
    NonFiniteParam,
    UnitMismatch,
    DuplicateName,
)


# =============================================================================
# YAML support (optional)
# =============================================================================

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # PyYAML не обязателен для JSON

if yaml is not None:
    # Сделать YAML ближе к JSON-подмножеству:
    # отключаем неявный резолвер timestamp, чтобы даты не превращались в datetime/date.
    class _MWLabSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
        pass

    # Важно: yaml_implicit_resolvers — class-level dict, делаем deepcopy чтобы не менять глобально SafeLoader.
    _MWLabSafeLoader.yaml_implicit_resolvers = copy.deepcopy(getattr(yaml.SafeLoader, "yaml_implicit_resolvers", {}))
    for ch, resolvers in list(_MWLabSafeLoader.yaml_implicit_resolvers.items()):
        _MWLabSafeLoader.yaml_implicit_resolvers[ch] = [
            r for r in resolvers if r and r[0] != "tag:yaml.org,2002:timestamp"
        ]

# =============================================================================
# Константы контракта v1
# =============================================================================

FORMAT_ID = "mwlab.spec"
FORMAT_VERSION = 1

# Разрешённые поля верхнего уровня (кроме x-*)
_TOP_LEVEL_ALLOWED = {"format", "version", "name", "criteria", "defaults", "meta"}

# Разрешённые секции defaults
_DEFAULTS_ALLOWED = {"criterion", "selector", "transform", "aggregator", "comparator"}

# Разрешённые поля Criterion (кроме x-*)
_CRITERION_ALLOWED = {"name", "weight", "selector", "transform", "aggregator", "comparator", "meta", "assume_prepared"}

# Разрешённые поля ComponentSpec (кроме x-*)
_COMPONENTSPEC_ALLOWED = {"type", "params", "meta"}


# =============================================================================
# Низкоуровневые утилиты валидации и преобразования
# =============================================================================

def _is_extension_key(k: str) -> bool:
    """x-* поля допускаются как расширения (forward-compat)."""
    return str(k).startswith("x-")


def _is_number(x: Any) -> bool:
    """True для int/float (но не для bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite_number(x: Any) -> bool:
    """True, если x — конечное число (int/float) без NaN/Inf. bool считаем допустимым."""
    if isinstance(x, bool):
        return True
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return math.isfinite(x)
    return False


def _raise_non_finite(path: SerdePath, value: Any) -> None:
    raise NonFiniteParam(
        path=path,
        message=f"Недопустимое не-конечное число (NaN/Inf): {value!r}",
    )


def _ensure_mapping(node: Any, path: SerdePath, *, what: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise SchemaError(path=path, message=f"{what} должен быть словарём (dict)")
    # ключи должны быть строками
    for k in node.keys():
        if not isinstance(k, str):
            raise SchemaError(path=path, message=f"{what}: ключи словаря должны быть строками, получено {type(k).__name__}")
    return cast(Mapping[str, Any], node)


def _ensure_list(node: Any, path: SerdePath, *, what: str) -> List[Any]:
    if not isinstance(node, list):
        raise SchemaError(path=path, message=f"{what} должен быть списком (list)")
    return node


def _ensure_str(node: Any, path: SerdePath, *, what: str) -> str:
    if not isinstance(node, str):
        raise SchemaError(path=path, message=f"{what} должен быть строкой (str)")
    s = node.strip()
    if not s:
        raise SchemaError(path=path, message=f"{what} не должен быть пустой строкой")
    return s


def _ensure_optional_str(node: Any, path: SerdePath, *, what: str, default: str = "") -> str:
    if node is None:
        return default
    if not isinstance(node, str):
        raise SchemaError(path=path, message=f"{what} должен быть строкой (str) или null")
    return node


def _ensure_bool(node: Any, path: SerdePath, *, what: str) -> bool:
    if not isinstance(node, bool):
        raise SchemaError(path=path, message=f"{what} должен быть bool")
    return node


def _ensure_int(node: Any, path: SerdePath, *, what: str) -> int:
    if isinstance(node, bool) or not isinstance(node, int):
        raise SchemaError(path=path, message=f"{what} должен быть int")
    return int(node)


def _ensure_float(node: Any, path: SerdePath, *, what: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise SchemaError(path=path, message=f"{what} должен быть числом (int/float)")
    v = float(node)
    if not math.isfinite(v):
        _raise_non_finite(path, v)
    return v


def _check_unknown_keys(
    mapping: Mapping[str, Any],
    path: SerdePath,
    *,
    allowed: set,
    what: str,
    strict: bool,
) -> None:
    """
    Проверить неизвестные ключи в mapping.
    - В strict режиме: неизвестные ключи (кроме x-*) -> ошибка.
    - В нестрогом: пропускаем.
    """
    if not strict:
        return
    for k in mapping.keys():
        if k in allowed or _is_extension_key(k):
            continue
        raise SchemaError(path=path.key(k), message=f"{what}: неизвестное поле '{k}'")


def _decode_complex_marker(node: Any, path: SerdePath) -> Any:
    """
    Декодировать канонический complex marker:
        {"__complex__": [re, im]} -> complex(re, im)

    Если node не является таким маркером — вернуть как есть.
    """
    if not isinstance(node, Mapping):
        return node
    if "__complex__" not in node:
        return node
    # Разрешаем только строго эту форму (для предсказуемости)
    if set(node.keys()) != {"__complex__"}:
        raise InvalidValue(path=path, message="Маркер __complex__ не должен содержать других ключей")

    arr = node["__complex__"]
    if not isinstance(arr, list) or len(arr) != 2:
        raise InvalidValue(path=path.key("__complex__"), message="__complex__ должен быть списком длины 2: [re, im]")

    re_v = arr[0]
    im_v = arr[1]
    if not isinstance(re_v, (int, float)) or isinstance(re_v, bool):
        raise InvalidValue(path=path.key("__complex__").index(0), message="Re комплексного числа должен быть числом")
    if not isinstance(im_v, (int, float)) or isinstance(im_v, bool):
        raise InvalidValue(path=path.key("__complex__").index(1), message="Im комплексного числа должен быть числом")

    re_f = float(re_v)
    im_f = float(im_v)
    if not (math.isfinite(re_f) and math.isfinite(im_f)):
        _raise_non_finite(path.key("__complex__"), [re_f, im_f])

    return complex(re_f, im_f)


def _walk_and_validate_primitives(
    node: Any,
    path: SerdePath,
    *,
    decode_complex: bool,
) -> Any:
    """
    Рекурсивно пройти по node и:
    - запретить NaN/Inf в числах
    - запретить ключи dict не-строки
    - запретить нестандартные типы (set/tuple/object/etc)
    - (опционально) декодировать complex marker

    Возвращает преобразованное значение (например, complex маркеры -> complex).
    """
    if decode_complex:
        node2 = _decode_complex_marker(node, path)
        if node2 is not node:
            return node2

    # Примитивы
    if node is None:
        return None
    if isinstance(node, bool):
        return node
    if isinstance(node, int) and not isinstance(node, bool):
        return node
    if isinstance(node, float):
        if not math.isfinite(node):
            _raise_non_finite(path, node)
        return node
    # complex может появиться после декодирования __complex__ маркера
    if isinstance(node, complex):
        re_v = float(node.real)
        im_v = float(node.imag)
        if not (math.isfinite(re_v) and math.isfinite(im_v)):
            _raise_non_finite(path, node)
        return node
    if isinstance(node, str):
        return node

    # Списки
    if isinstance(node, list):
        out_list: List[Any] = []
        for i, item in enumerate(node):
            out_list.append(_walk_and_validate_primitives(item, path.index(i), decode_complex=decode_complex))
        return out_list

    # Словари
    if isinstance(node, Mapping):
        for k in node.keys():
            if not isinstance(k, str):
                raise SchemaError(path=path, message=f"Ключи словаря должны быть строками, получено {type(k).__name__}")
        out: Dict[str, Any] = {}
        for k, v in node.items():
            out[k] = _walk_and_validate_primitives(v, path.key(k), decode_complex=decode_complex)
        return out

    # Остальные типы запрещены контрактом
    raise InvalidValue(
        path=path,
        message=f"Недопустимый тип в документе/params: {type(node).__name__}. Разрешены только JSON/YAML примитивы.",
    )


def _normalize_unit_like_strings(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Мягкая нормализация unit-полей в params.

    Контракт требует нормализовать freq_unit через normalize_freq_unit().
    Кроме того, некоторые параметры вида *_unit (band_unit, f0_unit и т.п.)
    могут принимать частотные единицы из {Hz,kHz,MHz,GHz}. Для MVP делаем так:

    - Если ключ оканчивается на "_unit" или равен "freq_unit",
      и значение — строка, пытаемся normalize_freq_unit().
      Если normalize_freq_unit() не подходит (например, "rad", "ns") — оставляем как есть.

    Это безопасно: мы НЕ трогаем нечастотные единицы, потому что normalize_freq_unit()
    их отвергнет, и мы просто оставим исходную строку.
    """
    out = dict(params)
    for k, v in list(out.items()):
        if not isinstance(k, str):
            continue
        if not (k == "freq_unit" or k.endswith("_unit")):
            continue
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        try:
            out[k] = normalize_freq_unit(s)
        except Exception:
            # не частотная единица — оставляем как есть
            out[k] = v
    return out


def _canonical_compose_type() -> Optional[str]:
    """
    Вернуть канонический type для transform.Compose (с учётом алиасов/реестра),
    либо None, если Compose не зарегистрирован.
    """
    if not has_type("transform", "Compose"):
        return None
    return canonicalize_alias("transform", "Compose")

def _is_compose_type(type_str: str) -> bool:
    """
    True, если type_str (возможно алиас) соответствует transform.Compose.
    """
    canon_compose = _canonical_compose_type()
    if canon_compose is None:
        return False
    try:
        return canonicalize_alias("transform", type_str) == canon_compose
    except Exception:
        return False


# =============================================================================
# Работа с defaults
# =============================================================================

def _parse_defaults(node: Any, path: SerdePath, *, strict: bool) -> Dict[str, Dict[str, Any]]:
    """
    Разобрать defaults по контракту:

        defaults:
          criterion: { weight: 1.0 }
          selector: { ... }
          transform: { ... }
          aggregator: { ... }
          comparator: { ... }

    Возвращает нормализованную структуру с 5 секциями, каждая — dict.
    """
    if node is None:
        return {k: {} for k in _DEFAULTS_ALLOWED}

    mp = _ensure_mapping(node, path, what="defaults")
    _check_unknown_keys(mp, path, allowed=_DEFAULTS_ALLOWED, what="defaults", strict=strict)

    out: Dict[str, Dict[str, Any]] = {k: {} for k in _DEFAULTS_ALLOWED}
    for section in _DEFAULTS_ALLOWED:
        if section not in mp:
            continue
        sec_node = mp[section]
        sec_path = path.key(section)
        sec_map = _ensure_mapping(sec_node, sec_path, what=f"defaults.{section}")

        # Рекурсивная проверка примитивности + запреты NaN/Inf + decode_complex
        # (complex в defaults тоже допустим по контракту, в канонической форме)
        validated = _walk_and_validate_primitives(sec_map, sec_path, decode_complex=True)
        assert isinstance(validated, dict)
        out[section] = _normalize_unit_like_strings(validated)
    return out


def _merge_defaults(defaults: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Merge по правилу контракта: подмешиваем default'ы только по отсутствующим ключам.
    Явные params всегда сильнее.
    """
    out = dict(defaults)
    for k, v in params.items():
        out[k] = v
    return out


def _filter_params_for_cls(cls: type, params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Оставить только те параметры, которые принимает конструктор cls (kwargs).
    Используется для безопасного применения defaults к "контейнерам" (например Compose),
    чтобы defaults.transform не ломал сборку цепочек из-за UnknownParam.
    """
    allowed = set(_signature_expected_params(cls))
    return {k: v for k, v in params.items() if k in allowed}


# =============================================================================
# Строгая проверка params по сигнатуре конструктора
# =============================================================================

def _signature_expected_params(cls: type) -> Tuple[str, ...]:
    """
    Разрешённые kwargs-имена для cls.

    В mwlab уже есть constructor_kw_params(cls) — используем его.
    """
    return constructor_kw_params(cls)


def _signature_required_params(cls: type) -> Tuple[str, ...]:
    """
    Попробовать определить обязательные kwargs-параметры (без default).

    Для MVP используем inspect.signature косвенно через constructor_kw_params + попытку
    "восстановить required" не делаем здесь вручную; но MissingParam по контракту важен.
    Поэтому реализуем required через inspect.signature напрямую.
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return tuple()

    required: List[str] = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            continue
        if p.default is inspect._empty:
            required.append(p.name)
    return tuple(required)


def _validate_params_against_signature(
    cls: type,
    params: Mapping[str, Any],
    path: SerdePath,
) -> None:
    """
    Строгая проверка:
    - неизвестные параметры -> UnknownParam
    - отсутствующие обязательные -> MissingParam
    """
    allowed = set(_signature_expected_params(cls))
    required = set(_signature_required_params(cls))

    # UnknownParam
    for k in params.keys():
        if k not in allowed:
            raise UnknownParam(
                path=path.key(k),
                message=f"{cls.__name__}: неизвестный параметр '{k}'. Разрешённые: {sorted(allowed)}",
            )

    # MissingParam
    missing = sorted([k for k in required if k not in params])
    if missing:
        raise MissingParam(
            path=path,
            message=f"{cls.__name__}: отсутствуют обязательные параметры: {missing}",
        )


# =============================================================================
# ComponentSpec / TransformSpec сборка
# =============================================================================

def _parse_componentspec(
    node: Any,
    path: SerdePath,
    *,
    strict: bool,
) -> Tuple[str, Dict[str, Any], Optional[Any]]:
    """
    Прочитать ComponentSpec:
        { type: "...", params: { ... } }

    Возвращает:
        (type_str, params_dict, meta)

    meta в MVP не влияет на построение компонента, но можно сохранить наружу.
    """
    mp = _ensure_mapping(node, path, what="ComponentSpec")
    _check_unknown_keys(mp, path, allowed=_COMPONENTSPEC_ALLOWED, what="ComponentSpec", strict=strict)

    t = _ensure_str(mp.get("type"), path.key("type"), what="ComponentSpec.type")

    params_node = mp.get("params", {})
    if params_node is None:
        params_node = {}
    params_map = _ensure_mapping(params_node, path.key("params"), what="ComponentSpec.params")

    # validate primitives + decode complex marker
    validated = _walk_and_validate_primitives(params_map, path.key("params"), decode_complex=True)
    assert isinstance(validated, dict)
    params = _normalize_unit_like_strings(validated)

    meta = mp.get("meta", None)
    if meta is not None:
        meta = _walk_and_validate_primitives(meta, path.key("meta"), decode_complex=True)

    return t, params, meta


def _build_component(
    kind: str,
    node: Any,
    path: SerdePath,
    *,
    defaults: Mapping[str, Any],
    strict: bool,
    require_canonical_types: bool,
) -> Any:
    """
    Построить компонент одного из видов: selector/transform/aggregator/comparator.

    - node: ComponentSpec dict
    - defaults: defaults для соответствующего вида (selector/transform/...)
    """
    type_str, params, meta = _parse_componentspec(node, path, strict=strict)

    # Проверка существования type в registry
    if not has_type(kind, type_str):
        hints = suggest_types(kind, type_str, canonical_only=True)
        raise UnknownType(
            path=path.key("type"),
            message=f"Неизвестный {kind}.type='{type_str}'.",
            hints=hints,
        )

    # Поддержка алиасов (backward-compat):
    canonical = canonicalize_alias(kind, type_str)
    if require_canonical_types and canonical != type_str:
        # Строгий режим "только канон" — даём понятную ошибку.
        raise InvalidValue(
            path=path.key("type"),
            message=f"{kind}.type='{type_str}' не является каноническим. Используйте '{canonical}'.",
        )

    # merge defaults (только по отсутствующим ключам)
    params_final = _merge_defaults(defaults, params)

    # Опционально можно повторно нормализовать unit-поля после merge
    params_final = _normalize_unit_like_strings(params_final)

    # resolve class
    cls = resolve_type(kind, canonical)

    # строгая проверка по сигнатуре
    _validate_params_against_signature(cls, params_final, path=path.key("params"))

    # создание экземпляра
    try:
        obj = cls(**params_final)
    except SerdeError:
        raise
    except (TypeError, ValueError) as e:
        raise InvalidValue(
            path=path.key("params"),
            message=f"{kind}({cls.__name__}): ошибка создания экземпляра: {e}",
            cause=e,
        ) from e
    except Exception as e:
        # Любая другая ошибка конструктора — тоже InvalidValue (MVP)
        raise InvalidValue(
            path=path.key("params"),
            message=f"{kind}({cls.__name__}): неожиданная ошибка создания экземпляра: {e}",
            cause=e,
        ) from e

    # meta (ComponentSpec.meta): сохраняем в объекте, если возможно (round-trip-friendly)
    if meta is not None:
        try:
            setattr(obj, "meta", meta)
        except Exception:
            # если объект "слотовый/замороженный" — игнорируем (MVP)
            pass

    return obj

def _build_transform(
    node: Any,
    path: SerdePath,
    *,
    defaults_transform: Mapping[str, Any],
    strict: bool,
    require_canonical_types: bool,
) -> Optional[Any]:
    """
    Построить transform по TransformSpec (контракт §8):
    - отсутствует / null -> None (identity)
    - ComponentSpec dict
    - Compose dict (type=Compose, params.transforms=[ComponentSpec...])
    - список ComponentSpec -> sugar, канонизируем в Compose
    """
    if node is None:
        return None

    # Синтаксический сахар: список transforms
    if isinstance(node, list):
        items = _ensure_list(node, path, what="Criterion.transform")
        # transforms: каждый элемент — ComponentSpec
        transforms: List[Any] = []
        for i, it in enumerate(items):
            tr = _build_component(
                "transform",
                it,
                path.index(i),
                defaults=defaults_transform,
                strict=strict,
                require_canonical_types=require_canonical_types,
            )
            transforms.append(tr)

        # Канонизируем в Compose (создаём объект напрямую из transforms-объектов)
        return _build_compose_from_objects(
            transforms,
            path,
            defaults_transform=defaults_transform,
            strict=strict,
            require_canonical_types=require_canonical_types,
        )

    # Обычный dict ComponentSpec
    mp = _ensure_mapping(node, path, what="Criterion.transform")
    # Если это Compose (в т.ч. алиас) с transforms как ComponentSpec list — собираем аккуратно
    t_val = mp.get("type", None)
    if isinstance(t_val, str) and _is_compose_type(t_val.strip()):
        # Разбираем как ComponentSpec, но transforms — особый ключ
        type_str, params, meta = _parse_componentspec(mp, path, strict=strict)

        # transforms поле обязательно
        if "transforms" not in params:
            raise SchemaError(path=path.key("params").key("transforms"), message="Compose.params.transforms обязателен")
        tr_list_node = params["transforms"]
        if tr_list_node is None:
            raise SchemaError(path=path.key("params").key("transforms"), message="Compose.params.transforms не должен быть null")
        if not isinstance(tr_list_node, list):
            raise SchemaError(path=path.key("params").key("transforms"), message="Compose.params.transforms должен быть списком")

        transforms: List[Any] = []
        for i, it in enumerate(tr_list_node):
            tr = _build_component(
                "transform",
                it,
                path.key("params").key("transforms").index(i),
                defaults=defaults_transform,
                strict=strict,
                require_canonical_types=require_canonical_types,
            )
            transforms.append(tr)

        # Остальные params Compose (кроме transforms) тоже передаём в конструктор
        compose_params = dict(params)
        compose_params.pop("transforms", None)

        obj = _build_compose_from_objects(
            transforms,
            path,
            defaults_transform=defaults_transform,
            strict=strict,
            require_canonical_types=require_canonical_types,
            extra_params=compose_params,
        )
        # meta из ComponentSpec для Compose
        if meta is not None:
            try:
                setattr(obj, "meta", meta)
            except Exception:
                pass
        return obj

    # Иначе — одиночный transform как компонент
    return _build_component(
        "transform",
        mp,
        path,
        defaults=defaults_transform,
        strict=strict,
        require_canonical_types=require_canonical_types,
    )


def _build_compose_from_objects(
    transforms: Sequence[Any],
    path: SerdePath,
    *,
    defaults_transform: Mapping[str, Any],
    strict: bool,
    require_canonical_types: bool,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Построить Compose(transform_chain) напрямую из уже созданных transform-объектов.

    Требование MVP:
    - В registry должен существовать transform type "Compose" (или его алиас),
      иначе невозможно создать Compose-класс.

    extra_params: дополнительные params для Compose (например, validate=True).
    """
    canon_compose = _canonical_compose_type()
    if canon_compose is None:
        hints = suggest_types("transform", "Compose", canonical_only=True)
        raise UnknownType(
            path=path.key("type"),
            message="Для канонического представления цепочек требуется зарегистрированный transform 'Compose'.",
            hints=hints,
        )

    cls = resolve_type("transform", canon_compose)

    # Сформируем параметры Compose: defaults.transform + extra_params + transforms
    # ВАЖНО: defaults.transform предназначены для leaf-трансформов и могут содержать
    # параметры, которые Compose не принимает. Поэтому:
    # - defaults применяем только по пересечению с сигнатурой Compose (без ошибок),
    # - user-supplied extra_params проверяем строго (ошибки должны быть видны).
    params: Dict[str, Any] = {}
    params.update(_filter_params_for_cls(cls, defaults_transform))
    if extra_params:
        params.update(dict(extra_params))

    # Важно: параметр transforms должен быть реальными объектами, а не dict-ами.
    params["transforms"] = list(transforms)

    # Проверка сигнатуры: Compose должен принимать transforms
    _validate_params_against_signature(cls, params, path=path.key("params"))

    try:
        obj = cls(**params)
    except SerdeError:
        raise
    except (TypeError, ValueError) as e:
        raise InvalidValue(
            path=path,
            message=f"transform(Compose): ошибка создания экземпляра: {e}",
            cause=e,
        ) from e
    except Exception as e:
        raise InvalidValue(
            path=path,
            message=f"transform(Compose): неожиданная ошибка создания экземпляра: {e}",
            cause=e,
        ) from e

    return obj


# =============================================================================
# Criterion / Specification build (loader)
# =============================================================================

def _build_criterion(
    node: Any,
    path: SerdePath,
    *,
    defaults: Dict[str, Dict[str, Any]],
    strict: bool,
    require_canonical_types: bool,
) -> BaseCriterion:
    """
    Собрать один Criterion из dict-узла.

    Поля по контракту:
      - name (обяз.)
      - weight (опц, default из defaults.criterion.weight, иначе 1.0)
      - selector (обяз.)
      - transform (опц)
      - aggregator (обяз.)
      - comparator (обяз.)
      - meta (опц)
      - assume_prepared (опц, не в контракте как must, но полезно для mwlab)
    """
    mp = _ensure_mapping(node, path, what="Criterion")
    _check_unknown_keys(mp, path, allowed=_CRITERION_ALLOWED, what="Criterion", strict=strict)

    name = _ensure_str(mp.get("name"), path.key("name"), what="Criterion.name")

    # weight: merge defaults.criterion.weight
    crit_defaults = defaults.get("criterion", {})
    w_node = mp.get("weight", crit_defaults.get("weight", 1.0))
    weight = _ensure_float(w_node, path.key("weight"), what="Criterion.weight")

    # Контракт: SHOULD требовать weight > 0 (MVP делаем строго > 0)
    if weight <= 0.0:
        raise InvalidValue(path=path.key("weight"), message="Criterion.weight должен быть > 0")

    assume_node = mp.get("assume_prepared", crit_defaults.get("assume_prepared", False))
    if assume_node is None:
        assume_prepared = False
    else:
        assume_prepared = bool(assume_node) if isinstance(assume_node, (bool, int)) else None
        if assume_prepared is None:
            raise SchemaError(path=path.key("assume_prepared"), message="assume_prepared должен быть bool")
        assume_prepared = bool(assume_prepared)

    # Компоненты
    if "selector" not in mp:
        raise SchemaError(path=path.key("selector"), message="Criterion.selector обязателен")
    if "aggregator" not in mp:
        raise SchemaError(path=path.key("aggregator"), message="Criterion.aggregator обязателен")
    if "comparator" not in mp:
        raise SchemaError(path=path.key("comparator"), message="Criterion.comparator обязателен")

    selector = _build_component(
        "selector",
        mp["selector"],
        path.key("selector"),
        defaults=defaults.get("selector", {}),
        strict=strict,
        require_canonical_types=require_canonical_types,
    )

    transform = None
    if "transform" in mp:
        transform = _build_transform(
            mp.get("transform"),
            path.key("transform"),
            defaults_transform=defaults.get("transform", {}),
            strict=strict,
            require_canonical_types=require_canonical_types,
        )

    aggregator = _build_component(
        "aggregator",
        mp["aggregator"],
        path.key("aggregator"),
        defaults=defaults.get("aggregator", {}),
        strict=strict,
        require_canonical_types=require_canonical_types,
    )

    comparator = _build_component(
        "comparator",
        mp["comparator"],
        path.key("comparator"),
        defaults=defaults.get("comparator", {}),
        strict=strict,
        require_canonical_types=require_canonical_types,
    )

    # Сборка Criterion + dry unit-aware валидация внутри __init__
    try:
        crit = BaseCriterion(
            selector=selector,
            transform=transform,
            aggregator=aggregator,
            comparator=comparator,
            weight=weight,
            name=name,
            assume_prepared=assume_prepared,
        )
    except ValueError as e:
        # Надёжная классификация через UnitMismatchError (подкласс ValueError из base.py).
        if (
                isinstance(e, UnitMismatchError)
                or getattr(e, "kind", None) == "UnitMismatch"
                or getattr(e, "__mwlab_kind__", None) == "UnitMismatch"
        ):
            raise UnitMismatch(path=path, message=f"Несовместимость единиц в Criterion '{name}': {e}", cause=e) from e
        raise InvalidValue(path=path, message=f"Ошибка построения Criterion '{name}': {e}", cause=e) from e
    except Exception as e:
        raise InvalidValue(path=path, message=f"Неожиданная ошибка построения Criterion '{name}': {e}", cause=e) from e

    # meta (MVP): сохраняем, если возможно
    if "meta" in mp:
        meta = _walk_and_validate_primitives(mp["meta"], path.key("meta"), decode_complex=True)
        try:
            setattr(crit, "meta", meta)
        except Exception:
            # Если объект "заморожен" — просто игнорируем (MVP)
            pass

    return crit


def load_spec_dict(
    doc: Mapping[str, Any],
    *,
    strict: bool = True,
    require_canonical_types: bool = False,
) -> Specification:
    """
    Загрузить Specification из Python-dict (уже распарсенного YAML/JSON).

    Параметры:
    - strict:
        True  -> неизвестные поля (кроме x-*) вызывают ошибку.
        False -> неизвестные поля игнорируются (кроме грубых нарушений типов).
    - require_canonical_types:
        True  -> запрещать неканонические алиасы в поле type (требовать только канон).
        False -> разрешить алиасы и приводить к канону через canonicalize_alias().

    Возвращает:
        Specification
    """
    root = SerdePath.root()
    mp = _ensure_mapping(doc, root, what="Документ спецификации")

    # Сначала строгая проверка примитивности и запретов NaN/Inf во всём документе.
    # ВАЖНО: здесь НЕ декодируем __complex__ (иначе complex "протечёт" в следующий проход валидации params).
    validated_doc = _walk_and_validate_primitives(mp, root, decode_complex=False)

    assert isinstance(validated_doc, dict)
    mp = validated_doc

    # Проверка заголовка
    fmt = mp.get("format", None)
    ver = mp.get("version", None)

    if fmt != FORMAT_ID:
        raise SchemaError(path=root.key("format"), message=f"format должен быть '{FORMAT_ID}', получено {fmt!r}")
    if ver != FORMAT_VERSION:
        raise SchemaError(path=root.key("version"), message=f"version должен быть {FORMAT_VERSION}, получено {ver!r}")

    # strict: неизвестные верхнеуровневые поля
    _check_unknown_keys(mp, root, allowed=_TOP_LEVEL_ALLOWED, what="Документ", strict=strict)

    name = mp.get("name", "spec")
    if name is None:
        name = "spec"
    if not isinstance(name, str):
        raise SchemaError(path=root.key("name"), message="name должен быть строкой")
    name = name or "spec"

    # defaults
    defaults = _parse_defaults(mp.get("defaults", None), root.key("defaults"), strict=strict)

    # criteria
    if "criteria" not in mp:
        raise SchemaError(path=root.key("criteria"), message="criteria обязательно и должно быть непустым списком")
    criteria_node = mp["criteria"]
    criteria_list = _ensure_list(criteria_node, root.key("criteria"), what="criteria")
    if not criteria_list:
        raise SchemaError(path=root.key("criteria"), message="criteria должно быть непустым списком")

    criteria: List[BaseCriterion] = []
    names_seen: Dict[str, int] = {}

    for i, item in enumerate(criteria_list):
        crit_path = root.key("criteria").index(i)
        crit = _build_criterion(
            item,
            crit_path,
            defaults=defaults,
            strict=strict,
            require_canonical_types=require_canonical_types,
        )
        # уникальность имён
        if crit.name in names_seen:
            j = names_seen[crit.name]
            raise DuplicateName(
                path=crit_path.key("name"),
                message=f"Имя критерия '{crit.name}' повторяется (первое в criteria[{j}])",
            )
        names_seen[crit.name] = i
        criteria.append(crit)

    # Создание Specification
    try:
        spec = Specification(criteria=criteria, name=name)
    except TypeError:
        # Если сигнатура Specification другая, пробуем позиционный вариант (MVP-friendly)
        try:
            spec = Specification(criteria, name=name)  # type: ignore[misc]
        except Exception as e:
            raise InvalidValue(path=root, message=f"Не удалось создать Specification: {e}", cause=e) from e
    except Exception as e:
        raise InvalidValue(path=root, message=f"Не удалось создать Specification: {e}", cause=e) from e

    # meta (MVP): сохраняем, если возможно
    if "meta" in mp:
        meta = _walk_and_validate_primitives(mp["meta"], root.key("meta"), decode_complex=True)
        try:
            setattr(spec, "meta", meta)
        except Exception:
            pass

    return spec


# =============================================================================
# Dump helpers (to dict / yaml / json)
# =============================================================================

def _encode_complex_marker(value: complex) -> Dict[str, Any]:
    """complex -> {"__complex__":[re, im]} с проверкой finite."""
    re_v = float(value.real)
    im_v = float(value.imag)
    if not (math.isfinite(re_v) and math.isfinite(im_v)):
        raise NonFiniteParam(path=SerdePath.root(), message=f"Недопустимое complex с NaN/Inf: {value!r}")
    return {"__complex__": [re_v, im_v]}


def _to_json_friendly(
    node: Any,
    path: SerdePath,
) -> Any:
    """
    Привести произвольный python-объект к JSON/YAML-friendly виду согласно контракту.

    Разрешаем:
    - None/bool/int/float/str
    - list/tuple -> list
    - dict (ключи str)
    - complex -> {"__complex__":[re,im]}
    - dataclass -> asdict (если это простая структура)
    - numpy scalar -> .item() (без импорта numpy: проверяем по duck-typing)

    Запрещаем:
    - numpy arrays / любые объекты со сложной структурой (без явного преобразования)
    - set, bytes, callable, class, etc.
    """
    # complex
    if isinstance(node, complex):
        return _encode_complex_marker(node)

    # простые примитивы
    if node is None or isinstance(node, (bool, str)):
        return node
    if isinstance(node, int) and not isinstance(node, bool):
        return node
    if isinstance(node, float):
        if not math.isfinite(node):
            _raise_non_finite(path, node)
        return node

    # numpy scalar: hasattr(item) and not a list/dict
    # (осторожно: некоторые объекты тоже имеют item(), но это приемлемо для MVP)
    if hasattr(node, "item") and callable(getattr(node, "item")) and not isinstance(node, (list, tuple, dict, Mapping)):
        try:
            v = node.item()  # type: ignore[attr-defined]
            return _to_json_friendly(v, path)
        except Exception:
            # если item() не помогает — запрещаем
            raise InvalidValue(path=path, message=f"Недопустимый тип (не удалось привести через .item()): {type(node).__name__}")

    # dataclass -> dict
    if is_dataclass(node):
        return _to_json_friendly(asdict(node), path)

    # list/tuple
    if isinstance(node, (list, tuple)):
        out: List[Any] = []
        for i, it in enumerate(node):
            out.append(_to_json_friendly(it, path.index(i)))
        return out

    # dict
    if isinstance(node, Mapping):
        out_d: Dict[str, Any] = {}
        for k, v in node.items():
            if not isinstance(k, str):
                raise InvalidValue(path=path, message="Ключи dict должны быть строками для JSON/YAML")
            out_d[k] = _to_json_friendly(v, path.key(k))
        return out_d

    # Остальное запрещаем
    raise InvalidValue(path=path, message=f"Недопустимый тип для сериализации: {type(node).__name__}")


def _dump_component_dict(
    obj: Any,
    *,
    explicit: bool,
    canonical_transforms: bool,
) -> Dict[str, Any]:
    """
    Dump одного компонента в ComponentSpec dict:
        {"type": "...", "params": {...}}

    explicit:
        True  -> пишем все serde_params()
        False -> пробуем исключить параметры, равные default в подписи __init__ (простая реализация)
    """
    kind = getattr(obj, "__mwlab_kind__", None)
    if not isinstance(kind, str) or not kind:
        raise InvalidValue(path=SerdePath.root(), message=f"Объект {obj!r} не имеет __mwlab_kind__ и не сериализуем как компонент")

    type_name = canonical_type(kind, obj)

    # Параметры из компонента
    if not hasattr(obj, "serde_params") or not callable(getattr(obj, "serde_params")):
        raise InvalidValue(path=SerdePath.root(), message=f"{type(obj).__name__} не поддерживает serde_params()")

    params_raw = obj.serde_params()
    if params_raw is None:
        params_raw = {}
    if not isinstance(params_raw, dict):
        raise InvalidValue(path=SerdePath.root(), message=f"{type(obj).__name__}.serde_params() должен возвращать dict")

    # Приводим к JSON-friendly и запрещаем NaN/Inf
    params_full = cast(Dict[str, Any], _to_json_friendly(params_raw, SerdePath.root().key("params")))
    # Мягкая нормализация unit-строк для стабилизации dump (не обязательна, но полезна)
    params_full = _normalize_unit_like_strings(params_full)

    if not explicit:
        # Простой diff-from-default: если параметр есть в подписи __init__ и у него есть default,
        # и значение совпадает — убираем.
        import inspect

        try:
            sig = inspect.signature(obj.__class__.__init__)
            defaults: Dict[str, Any] = {}
            for p in sig.parameters.values():
                if p.name == "self":
                    continue
                if p.default is not inspect._empty:
                    defaults[p.name] = p.default
            params_compact = {}
            for k, v in params_full.items():
                if k in defaults and v == _to_json_friendly(defaults[k], SerdePath.root().key("defaults").key(k)):
                    continue
                params_compact[k] = v
            params_full = params_compact
        except Exception:
            # Если не смогли — оставляем explicit
            pass

    out: Dict[str, Any] = {"type": type_name}
    if params_full:
        out["params"] = params_full
    else:
        out["params"] = {}  # канонично хранить params (по контракту может отсутствовать, но так проще)

    # meta (ComponentSpec.meta): round-trip-friendly
    if hasattr(obj, "meta"):
        meta_val = getattr(obj, "meta")
        if meta_val is not None:
            out["meta"] = _to_json_friendly(meta_val, SerdePath.root().key("meta"))

    return out


def _dump_transform_field(
    transform_obj: Any,
    *,
    explicit: bool,
    canonical_transforms: bool,
) -> Any:
    """
    Dump поля transform.

    canonical_transforms=True:
      - None -> null
      - одиночный transform -> Compose(...) (если доступен) или ошибка
      - Compose/цепочка -> Compose(...)

    canonical_transforms=False:
      - None -> null
      - transform -> ComponentSpec
        (если это Compose, то он должен корректно сериализоваться сам через serde_params())
    """
    if transform_obj is None:
        return None

    if not canonical_transforms:
        # Особый случай: Compose часто хранит transforms как объекты, и может
        # не уметь отдавать их в serde_params(). Чтобы dump по умолчанию был надёжным,
        # делаем устойчивый dump Compose-цепочки, если объект действительно Compose.
        try:
            kind = getattr(transform_obj, "__mwlab_kind__", None)
            if kind == "transform":
                canon_compose = _canonical_compose_type()
                tname = canonical_type("transform", transform_obj)
                if canon_compose is not None and tname == canon_compose:
                    chain: List[Any]
                    if hasattr(transform_obj, "iter_transforms") and callable(getattr(transform_obj, "iter_transforms")):
                        chain = list(transform_obj.iter_transforms())
                    elif hasattr(transform_obj, "transforms"):
                        chain = list(getattr(transform_obj, "transforms"))
                    else:
                        raise InvalidValue(path=SerdePath.root().key("transform"), message="Compose не предоставляет iter_transforms()/transforms")

                    transforms_specs = [_dump_component_dict(t, explicit=explicit, canonical_transforms=False) for t in chain]

                    # Доп. параметры Compose (если доступны) — сохраняем, но убираем transforms.
                    extra_params: Dict[str, Any] = {}
                    if hasattr(transform_obj, "serde_params") and callable(getattr(transform_obj, "serde_params")):
                        raw = transform_obj.serde_params() or {}
                        if not isinstance(raw, dict):
                            raise InvalidValue(path=SerdePath.root().key("transform").key("params"), message="Compose.serde_params() должен возвращать dict")
                        raw = dict(raw)
                        raw.pop("transforms", None)
                        extra_params = cast(Dict[str, Any], _to_json_friendly(raw, SerdePath.root().key("transform").key("params")))
                        extra_params = _normalize_unit_like_strings(extra_params)

                    params = dict(extra_params)
                    params["transforms"] = transforms_specs
                    out: Dict[str, Any] = {"type": tname, "params": params}

                    # meta
                    if hasattr(transform_obj, "meta"):
                        mv = getattr(transform_obj, "meta")
                        if mv is not None:
                            out["meta"] = _to_json_friendly(mv, SerdePath.root().key("transform").key("meta"))
                    return out
        except SerdeError:
            raise
        except Exception:
            # fallback to generic dump below
            pass

        return _dump_component_dict(transform_obj, explicit=explicit, canonical_transforms=canonical_transforms)


    # canonical_transforms=True: всегда Compose
    # Требуется, чтобы Compose был зарегистрирован (иначе невозможно гарантировать канон).
    canon_compose = _canonical_compose_type()
    if canon_compose is None:
        hints = suggest_types("transform", "Compose", canonical_only=True)
        raise UnknownType(
            path=SerdePath.root().key("transform").key("type"),
            message="canonical_transforms=True требует зарегистрированный transform 'Compose'.",
            hints=hints,
        )

    # Получаем цепочку transforms через iter_transforms()
    if hasattr(transform_obj, "iter_transforms") and callable(getattr(transform_obj, "iter_transforms")):
        chain = list(transform_obj.iter_transforms())
    else:
        chain = [transform_obj]

    transforms_specs = [_dump_component_dict(t, explicit=explicit, canonical_transforms=False) for t in chain]

    # ВАЖНО: canonical_transforms не должен приводить к потере информации.
    # Если исходный объект реально Compose, сохраняем его дополнительные params и meta.
    extra_params: Dict[str, Any] = {}
    is_compose_obj = False
    try:
        is_compose_obj = (getattr(transform_obj, "__mwlab_kind__", None) == "transform" and canonical_type("transform", transform_obj) == canon_compose)
    except Exception:
        is_compose_obj = False

    if is_compose_obj and hasattr(transform_obj, "serde_params") and callable(getattr(transform_obj, "serde_params")):
        raw = transform_obj.serde_params() or {}
        if not isinstance(raw, dict):
            raise InvalidValue(path=SerdePath.root().key("transform").key("params"), message="Compose.serde_params() должен возвращать dict")
        raw = dict(raw)
        raw.pop("transforms", None)
        extra_params = cast(Dict[str, Any], _to_json_friendly(raw, SerdePath.root().key("transform").key("params")))
        extra_params = _normalize_unit_like_strings(extra_params)

    params = dict(extra_params)
    params["transforms"] = transforms_specs

    out: Dict[str, Any] = {"type": canon_compose, "params": params}
    if is_compose_obj and hasattr(transform_obj, "meta"):
        mv = getattr(transform_obj, "meta")
        if mv is not None:
            out["meta"] = _to_json_friendly(mv, SerdePath.root().key("transform").key("meta"))
    return out

def dump_spec_dict(
    spec: Any,
    *,
    explicit: bool = True,
    canonical_transforms: bool = False,
) -> Dict[str, Any]:
    """
    Сериализовать Specification в dict по контракту v1.

    Параметры:
    - explicit=True: писать все параметры компонентов
    - canonical_transforms=False: писать transform как есть; True -> всегда Compose
    """
    # Извлекаем имя и criteria
    name = getattr(spec, "name", "spec")
    if not isinstance(name, str):
        name = str(name)

    criteria = getattr(spec, "criteria", None)
    if criteria is None:
        raise InvalidValue(path=SerdePath.root(), message="Specification должен иметь атрибут criteria")
    crit_list = list(criteria)
    if not crit_list:
        # контракт говорит "criteria должно быть непустым", но иногда удобно дампить пустую.
        # Для MVP: считаем это ошибкой, чтобы формат был валиден.
        raise InvalidValue(path=SerdePath.root().key("criteria"), message="Нельзя сериализовать Specification с пустым criteria")

    out: Dict[str, Any] = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "name": name or "spec",
        "criteria": [],
    }

    # meta (если есть)
    if hasattr(spec, "meta"):
        try:
            out["meta"] = _to_json_friendly(getattr(spec, "meta"), SerdePath.root().key("meta"))
        except SerdeError:
            raise
        except Exception as e:
            raise InvalidValue(path=SerdePath.root().key("meta"), message=f"meta не сериализуемо: {e}", cause=e) from e

    # Критерии
    names_seen: set = set()
    for i, c in enumerate(crit_list):
        p = SerdePath.root().key("criteria").index(i)

        c_name = getattr(c, "name", None) or f"crit_{i}"
        if not isinstance(c_name, str):
            c_name = str(c_name)
        if c_name in names_seen:
            raise DuplicateName(path=p.key("name"), message=f"Имя критерия '{c_name}' повторяется при dump")
        names_seen.add(c_name)

        weight = getattr(c, "weight", 1.0)
        # приводим к float и запрещаем NaN/Inf
        weight_f = float(weight)
        if not math.isfinite(weight_f):
            _raise_non_finite(p.key("weight"), weight_f)

        item: Dict[str, Any] = {
            "name": c_name,
            "weight": weight_f,
        }

        # selector/transform/aggregator/comparator
        selector = getattr(c, "selector", None)
        aggregator = getattr(c, "aggregator", None)
        comparator = getattr(c, "comparator", None)
        transform = getattr(c, "transform", None)

        if selector is None or aggregator is None or comparator is None:
            raise InvalidValue(path=p, message="Criterion должен содержать selector/aggregator/comparator для dump")

        item["selector"] = _dump_component_dict(selector, explicit=explicit, canonical_transforms=canonical_transforms)
        item["aggregator"] = _dump_component_dict(aggregator, explicit=explicit, canonical_transforms=canonical_transforms)
        item["comparator"] = _dump_component_dict(comparator, explicit=explicit, canonical_transforms=canonical_transforms)
        item["transform"] = _dump_transform_field(transform, explicit=explicit, canonical_transforms=canonical_transforms)

        # meta (если есть)
        if hasattr(c, "meta"):
            try:
                item["meta"] = _to_json_friendly(getattr(c, "meta"), p.key("meta"))
            except SerdeError:
                raise
            except Exception as e:
                raise InvalidValue(path=p.key("meta"), message=f"meta критерия не сериализуемо: {e}", cause=e) from e

        out["criteria"].append(item)

    # Финальная проверка "документ только из разрешённых типов" (страховка)
    out = cast(Dict[str, Any], _walk_and_validate_primitives(out, SerdePath.root(), decode_complex=False))
    return out


# =============================================================================
# YAML/JSON: load/dump из файлов и строк
# =============================================================================

def _detect_format_from_path(path: Union[str, Path]) -> str:
    """
    Вернуть "json" или "yaml" по расширению.
    Допустимые: .json, .yaml, .yml
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".json":
        return "json"
    if ext in (".yaml", ".yml"):
        return "yaml"
    raise ValueError(f"Неизвестное расширение файла '{p.name}'. Ожидается .json/.yaml/.yml")


def loads_json(text: str, *, strict: bool = True, require_canonical_types: bool = False) -> Specification:
    """
    Загрузить Specification из JSON-строки.

    JSON loader MUST запрещать NaN/Infinity (не-RFC).
    """
    def _forbid_constants(x: str) -> Any:
        raise NonFiniteParam(path=SerdePath.root(), message=f"JSON содержит запрещённую константу: {x}")

    try:
        data = json.loads(text, parse_constant=_forbid_constants)
    except SerdeError:
        raise
    except Exception as e:
        raise SchemaError(path=SerdePath.root(), message=f"Ошибка парсинга JSON: {e}", cause=e) from e

    mp = _ensure_mapping(data, SerdePath.root(), what="JSON документ")
    return load_spec_dict(mp, strict=strict, require_canonical_types=require_canonical_types)


def dumps_json(
    spec: Any,
    *,
    explicit: bool = True,
    canonical_transforms: bool = False,
    indent: int = 2,
) -> str:
    """
    Сериализовать Specification в JSON-строку.

    allow_nan=False гарантирует запрет NaN/Inf при записи.
    """
    data = dump_spec_dict(spec, explicit=explicit, canonical_transforms=canonical_transforms)
    try:
        return json.dumps(data, ensure_ascii=False, indent=indent, allow_nan=False)
    except ValueError as e:
        # allow_nan=False может бросить ValueError
        raise NonFiniteParam(path=SerdePath.root(), message=f"Нельзя сериализовать JSON из-за NaN/Inf: {e}", cause=e) from e
    except Exception as e:
        raise InvalidValue(path=SerdePath.root(), message=f"Ошибка сериализации JSON: {e}", cause=e) from e


def loads_yaml(text: str, *, strict: bool = True, require_canonical_types: bool = False) -> Specification:
    """
    Загрузить Specification из YAML-строки.

    Используется yaml.safe_load (без python-тегов).
    """
    if yaml is None:
        raise ImportError("PyYAML не установлен: YAML serde недоступен. Установите пакет 'PyYAML'.")

    try:
        # Используем loader без timestamp-resolver: YAML остаётся JSON-подмножеством по типам.
        data = yaml.load(text, Loader=_MWLabSafeLoader)  # type: ignore[arg-type]
    except Exception as e:
        raise SchemaError(path=SerdePath.root(), message=f"Ошибка парсинга YAML: {e}", cause=e) from e

    mp = _ensure_mapping(data, SerdePath.root(), what="YAML документ")
    return load_spec_dict(mp, strict=strict, require_canonical_types=require_canonical_types)


def dumps_yaml(
    spec: Any,
    *,
    explicit: bool = True,
    canonical_transforms: bool = False,
    sort_keys: bool = False,
) -> str:
    """
    Сериализовать Specification в YAML-строку.

    Используется yaml.safe_dump, без python-типов.
    """
    if yaml is None:
        raise ImportError("PyYAML не установлен: YAML serde недоступен. Установите пакет 'PyYAML'.")

    data = dump_spec_dict(spec, explicit=explicit, canonical_transforms=canonical_transforms)
    try:
        return yaml.safe_dump(data, sort_keys=sort_keys, allow_unicode=True)
    except Exception as e:
        raise InvalidValue(path=SerdePath.root(), message=f"Ошибка сериализации YAML: {e}", cause=e) from e


def load_spec(
    path: Union[str, Path],
    *,
    strict: bool = True,
    require_canonical_types: bool = False,
    encoding: str = "utf-8",
) -> Specification:
    """
    Загрузить Specification из файла (.yaml/.yml/.json).
    """
    fmt = _detect_format_from_path(path)
    text = Path(path).read_text(encoding=encoding)

    if fmt == "json":
        return loads_json(text, strict=strict, require_canonical_types=require_canonical_types)
    return loads_yaml(text, strict=strict, require_canonical_types=require_canonical_types)


def dump_spec(
    spec: Any,
    path: Union[str, Path],
    *,
    explicit: bool = True,
    canonical_transforms: bool = False,
    encoding: str = "utf-8",
) -> None:
    """
    Сериализовать Specification в файл (.yaml/.yml/.json).
    """
    fmt = _detect_format_from_path(path)
    p = Path(path)

    if fmt == "json":
        text = dumps_json(spec, explicit=explicit, canonical_transforms=canonical_transforms)
        p.write_text(text + "\n", encoding=encoding)
        return

    text = dumps_yaml(spec, explicit=explicit, canonical_transforms=canonical_transforms)
    p.write_text(text, encoding=encoding)


# =============================================================================
# Публичный экспорт
# =============================================================================

__all__ = [
    # dict-level
    "load_spec_dict",
    "dump_spec_dict",
    # json
    "loads_json",
    "dumps_json",
    # yaml
    "loads_yaml",
    "dumps_yaml",
    # files
    "load_spec",
    "dump_spec",
]
