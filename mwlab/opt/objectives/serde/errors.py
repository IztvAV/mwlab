# mwlab/opt/objectives/serde/errors.py
"""
mwlab.opt.objectives.serde.errors
================================

Этот модуль содержит **канонические исключения** для подсистемы сериализации /
десериализации (serde) спецификаций mwlab.

Цели модуля (MVP-friendly)
--------------------------
1) Дать единый базовый тип ошибки SerdeError, который:
   - хранит `path` (путь до проблемного поля в документе),
   - хранит `kind` (категория ошибки: UnknownType / MissingParam / ...),
   - хранит человекочитаемое `message`,
   - опционально хранит `hints` (подсказки),
   - корректно "печатается" и может быть преобразован в dict для логов/GUI.

2) Дать простой и безопасный инструмент формирования пути `SerdePath`,
   чтобы не собирать строки вида "criteria[2].selector.params.m" вручную.

Контракт (см. документ пользователя)
------------------------------------
Сообщения об ошибках **СЛЕДУЕТ** включать:
- `path` (например: criteria[0].transform.params.transforms[2].params.basis)
- `kind` (например: UnknownType, MissingParam, InvalidValue, UnitMismatch, NonFiniteParam)
- человекочитаемое сообщение
- (опционально) `hints` (например, близкие `type` по difflib)

Примечание
----------
Модуль intentionally *не зависит* от registry/base и не импортирует другие части mwlab,
чтобы избежать циклических импортов. Это "тонкий" слой, используемый core-лоадером.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# =============================================================================
# SerdePath: удобное построение пути (path) до поля документа
# =============================================================================

# Сегмент пути:
# - str : ключ dict (например, "criteria", "selector", "params", "m")
# - int : индекс в списке (например, 0, 1, 2)
PathSegment = Union[str, int]


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_identifier(s: str) -> bool:
    """True, если строку можно безопасно печатать как .key без экранирования."""
    return bool(_IDENTIFIER_RE.match(s))


def _escape_key_for_brackets(key: str) -> str:
    """
    Экранировать ключ для формы ["..."].

    Мы используем простое экранирование двойных кавычек и обратного слеша,
    чтобы строка была читаемой и однозначной.
    """
    return key.replace("\\", "\\\\").replace('"', '\\"')


@dataclass(frozen=True)
class SerdePath:
    """
    Неизменяемый путь до поля документа serde.

    Использование:
        p = SerdePath.root().key("criteria").index(0).key("selector").key("type")
        str(p) -> "criteria[0].selector.type"

    Почему не просто строка?
    - меньше ошибок при формировании пути,
    - проще добавлять сегменты в рекурсивных функциях,
    - можно переиспользовать один базовый путь.
    """

    segments: Tuple[PathSegment, ...] = ()

    @staticmethod
    def root() -> "SerdePath":
        """Корневой путь (пустой)."""
        return SerdePath(())

    def key(self, name: str) -> "SerdePath":
        """Добавить сегмент-ключ (dict key)."""
        return SerdePath(self.segments + (str(name),))

    def index(self, i: int) -> "SerdePath":
        """Добавить сегмент-индекс (list index)."""
        return SerdePath(self.segments + (int(i),))

    def extend(self, *segments: PathSegment) -> "SerdePath":
        """Добавить произвольные сегменты (редко нужно, но полезно)."""
        return SerdePath(self.segments + tuple(segments))

    def to_string(self) -> str:
        """
        Преобразовать путь к строке вида:
            criteria[2].transform.params.transforms[1].params.basis

        Правила печати:
        - str-ключи, похожие на идентификатор: печатаем через точку `.key`
          (первый ключ без точки).
        - "сложные" ключи (с пробелами, тире, спецсимволами): печатаем как ["..."].
        - индексы: печатаем как [i].
        """
        if not self.segments:
            return ""  # корень

        out: List[str] = []
        first = True
        for seg in self.segments:
            if isinstance(seg, int):
                out.append(f"[{seg}]")
                first = False
                continue

            # seg is str
            key = seg
            if first:
                # первый сегмент: без ведущей точки
                if _is_identifier(key):
                    out.append(key)
                else:
                    out.append(f'["{_escape_key_for_brackets(key)}"]')
                first = False
            else:
                if _is_identifier(key):
                    out.append("." + key)
                else:
                    out.append(f'["{_escape_key_for_brackets(key)}"]')
        return "".join(out)

    def __str__(self) -> str:  # pragma: no cover
        return self.to_string()


def path_to_str(path: Union[str, SerdePath, None]) -> str:
    """
    Привести path к строке.

    Принимаем:
    - SerdePath
    - готовую строку
    - None (трактуем как корень)
    """
    if path is None:
        return ""
    if isinstance(path, SerdePath):
        return path.to_string()
    return str(path)


# =============================================================================
# SerdeError: базовый тип ошибок serde + подклассы по категориям
# =============================================================================

class SerdeError(ValueError):
    """
    Базовая ошибка serde.

    Поля:
    - path  : путь до проблемного поля (строка)
    - kind  : категория ошибки (строка, например "UnknownType")
    - message : человекочитаемое пояснение
    - hints : список подсказок (опционально)
    - cause : исходное исключение (опционально), полезно для отладки

    Важно:
    - `kind` задуман как машинно-обрабатываемая метка (для GUI/логов/CI).
    - `path` должен указывать на максимально точное место в документе.
    """

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        kind: str,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.path: str = path_to_str(path)
        self.kind: str = str(kind)
        self.message: str = str(message)

        # Нормализуем hints: делаем list[str] или [].
        if hints is None:
            self.hints: List[str] = []
        else:
            self.hints = [str(h) for h in hints if str(h)]

        self.cause: Optional[BaseException] = cause

        # Сообщение ValueError делаем сразу "человеческим".
        super().__init__(self._format())

    def _format(self) -> str:
        """
        Форматирование ошибки в строку.

        Пример:
            [UnknownType] criteria[0].selector.type: selector type 'SMagSelektor' not found. Hints: [...]
        """
        loc = f"{self.path}: " if self.path else ""
        base = f"[{self.kind}] {loc}{self.message}"
        if self.hints:
            base += f" Hints: {self.hints}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать ошибку к JSON/YAML-friendly dict (для логов/репортов).

        Пример:
            {
              "kind": "UnknownType",
              "path": "criteria[0].selector.type",
              "message": "...",
              "hints": ["SMagSelector", "SDBSelector"]
            }
        """
        out: Dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "message": self.message,
        }
        if self.hints:
            out["hints"] = list(self.hints)

        # cause намеренно не сериализуем как объект;
        # но можно отдать его repr для отладки.
        if self.cause is not None:
            out["cause"] = repr(self.cause)
        return out


# -----------------------------------------------------------------------------
# Подклассы по категориям (имена kind соответствуют serde-контракту)
# -----------------------------------------------------------------------------

class SchemaError(SerdeError):
    """Ошибка структуры документа (не тот формат/version, лишние поля, неверные типы узлов и т.п.)."""

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="SchemaError", message=message, hints=hints, cause=cause)


class UnknownType(SerdeError):
    """
    Неизвестный компонент `type` в registry.

    Рекомендуется включать в message:
    - kind компонента (selector/transform/aggregator/comparator)
    - значение type
    """

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="UnknownType", message=message, hints=hints, cause=cause)


class UnknownParam(SerdeError):
    """Неизвестный параметр в params (строгая проверка сигнатуры конструктора)."""

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="UnknownParam", message=message, hints=hints, cause=cause)


class MissingParam(SerdeError):
    """Отсутствует обязательный параметр конструктора компонента."""

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="MissingParam", message=message, hints=hints, cause=cause)


class InvalidValue(SerdeError):
    """
    Неверное значение (тип/диапазон/структура), в т.ч. ошибки конструктора.

    Используется как "обёртка" над ValueError/TypeError от __init__,
    а также для ручной проверки семантики параметров.
    """

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="InvalidValue", message=message, hints=hints, cause=cause)


class NonFiniteParam(SerdeError):
    """
    В параметрах обнаружены NaN/Inf.

    По контракту:
    - loader MUST отклонять документы, содержащие NaN/Inf, и показывать path.
    """

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="NonFiniteParam", message=message, hints=hints, cause=cause)


class UnitMismatch(SerdeError):
    """
    Несовместимость единиц/ожиданий компонентов (dry-валидация цепочки).

    Обычно это обёртка над ValueError, выброшенным при построении Criterion,
    когда `validate_expected_units(...)` выявил несовместимость.
    """

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="UnitMismatch", message=message, hints=hints, cause=cause)


# -----------------------------------------------------------------------------
# Дополнительные "частые" ошибки (не обязательны для контракта, но полезны в MVP)
# -----------------------------------------------------------------------------

class DuplicateName(SerdeError):
    """
    Дублирование имени критерия в `criteria[*].name`.

    Контракт рекомендует уникальность, иначе проблемы в отчётах/ключах.
    """

    def __init__(
        self,
        *,
        path: Union[str, SerdePath, None] = None,
        message: str,
        hints: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(path=path, kind="DuplicateName", message=message, hints=hints, cause=cause)


# =============================================================================
# Публичный экспорт
# =============================================================================

__all__ = [
    # path tools
    "PathSegment",
    "SerdePath",
    "path_to_str",
    # errors
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
