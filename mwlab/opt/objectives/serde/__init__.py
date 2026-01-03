# mwlab/opt/objectives/serde/__init__.py
"""
mwlab.opt.objectives.serde
=========================

Публичный API подсистемы сериализации/десериализации (serde) для objectives.

MVP-цель
--------
Дать минимальный и удобный интерфейс для:
- загрузки спецификаций из YAML/JSON (строка/файл/словарь),
- дампа спецификаций в YAML/JSON (строка/файл/словарь),
- получения структурированных ошибок (SerdeError и подклассы).

Формат
------
Контракт: MWLAB Objectives Serde Contract (v1)
- format:  "mwlab.spec"
- version: 1

Принцип использования
---------------------
Рекомендуемый импорт (стабильный фасад):

    from mwlab.opt.objectives.serde import load_spec, dump_spec

    spec = load_spec("spec.yaml")
    dump_spec(spec, "out.json")

Или работа со строками:

    spec = loads_yaml(yaml_text)
    json_text = dumps_json(spec)

Ошибки
------
Все ошибки loader/dumper поднимают исключения SerdeError (или подклассы),
в которых есть:
- path  : путь до проблемного поля
- kind  : категория ошибки (UnknownType, MissingParam, ...)
- message, hints

Это удобно для GUI/логов/CI.
"""

from __future__ import annotations

# --- core API ---
from .core import (
    load_spec_dict,
    dump_spec_dict,
    loads_json,
    dumps_json,
    loads_yaml,
    dumps_yaml,
    load_spec,
    dump_spec,
)

# --- errors (re-export) ---
from .errors import (
    PathSegment,
    SerdePath,
    path_to_str,
    SerdeError,
    SchemaError,
    UnknownType,
    UnknownParam,
    MissingParam,
    InvalidValue,
    NonFiniteParam,
    UnitMismatch,
    DuplicateName,
)

__all__ = [
    # core
    "load_spec_dict",
    "dump_spec_dict",
    "loads_json",
    "dumps_json",
    "loads_yaml",
    "dumps_yaml",
    "load_spec",
    "dump_spec",
    # errors
    "PathSegment",
    "SerdePath",
    "path_to_str",
    "SerdeError",
    "SchemaError",
    "UnknownType",
    "UnknownParam",
    "MissingParam",
    "InvalidValue",
    "NonFiniteParam",
    "UnitMismatch",
    "DuplicateName",
]

