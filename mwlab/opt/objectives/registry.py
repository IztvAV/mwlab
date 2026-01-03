# mwlab/opt/objectives/registry.py
"""
mwlab.opt.objectives.registry
=============================

Единый реестр (registry) компонент подсистемы целей/ограничений:

- Selector
- Transform
- Aggregator
- Comparator

Задачи модуля
-------------

1) Авторегистрация классов по alias (декораторы @register_selector/...).
2) Отображение:
   - alias -> класс компонента (для сборки пайплайнов из YAML/JSON);
   - класс -> канонический alias (для стабильной сериализации);
   - класс -> все алиасы (для диагностики и обратной совместимости).
3) Утилиты для интроспекции и диагностики:
   - список известных типов по kind;
   - получение канонического имени типа по объекту/классу.

Важно
-----

- Модуль НЕ знает ничего про конкретные реализации BaseSelector/BaseTransform/...
  и не импортирует `base.py`, чтобы избежать циклических импортов.
- Все типы задаются через строковые алиасы (например, "SMagSelector", "le", "target").
- Канонический alias используется как `type` в файловом формате (YAML/JSON).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    Optional,
)

# =============================================================================
# Общая часть: нормализация alias и универсальный класс реестра
# =============================================================================

T = TypeVar("T")


def normalize_alias(alias: str) -> str:
    """
    Нормализовать строковый alias для использования в registry.

    Правила нормализации:
    - обрезать пробелы по краям (strip);
    - внутренние пробелы заменить на '_' (чтобы alias был идентификатороподобным).

    Важно:
    - регистр (case) НЕ меняется: 'LEComparator' и 'lecomparator' считаются разными
      алиасами. Канонический вид alias-а задаётся при регистрации.
    """
    s = str(alias).strip()
    if not s:
        raise ValueError("alias не должен быть пустой строкой")
    return s.replace(" ", "_")


@dataclass
class _Registry:
    """
    Внутренний универсальный реестр для одного вида компонентов.

    Примеры видов (kind):
    - "selector"
    - "transform"
    - "aggregator"
    - "comparator"

    Храним три отображения:

    1) alias_to_cls : Dict[str, Type]
       - ключ: строковый alias типа (например, "SMagSelector", "le", "target")
       - значение: класс компонента

    2) cls_to_aliases : Dict[Type, Tuple[str, ...]]
       - для каждого класса хранится кортеж всех его алиасов
       - полезно для отчётов, отладки и обратной совместимости

    3) cls_to_canonical : Dict[Type, str]
       - канонический alias (первый в списке при регистрации)
       - именно этот alias используется при сериализации в YAML/JSON
    """

    # Логическое имя вида компонента (для диагностики)
    kind: str

    # alias -> класс
    alias_to_cls: Dict[str, Type[Any]] = field(default_factory=dict)

    # класс -> все алиасы (в порядке добавления, без повторов)
    cls_to_aliases: Dict[Type[Any], Tuple[str, ...]] = field(default_factory=dict)

    # класс -> канонический alias (первый, с которым зарегистрировали класс)
    cls_to_canonical: Dict[Type[Any], str] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Регистрация
    # -------------------------------------------------------------------------
    def register(self, aliases: Union[str, Iterable[str]]) -> Callable[[Type[T]], Type[T]]:
        """
        Декоратор регистрации класса компонента под одним или несколькими alias-ами.

        Использование:
        -------------
        @register_selector("SMagSelector")
        class SMagSelector(BaseSelector):
            ...

        или

        @register_selector(("SMagSelector", "s_mag", "sdb"))
        class SMagSelector(BaseSelector):
            ...

        Правила:
        - все aliases нормализуются через normalize_alias();
        - первый alias становится КАНОНИЧЕСКИМ (используется при dump/serde);
        - если alias уже занят другим классом — выбрасывается KeyError
          (ошибка конфигурации при импорте).
        """
        # Приводим aliases к списку и нормализуем
        raw = [aliases] if isinstance(aliases, str) else list(aliases)
        if not raw:
            raise ValueError(f"{self.kind}: список aliases пуст")

        norm = tuple(normalize_alias(a) for a in raw)
        canonical = norm[0]

        def _wrap(cls: Type[T]) -> Type[T]:
            # 1) Проверяем, что ни один из алиасов не занят другим классом
            for a in norm:
                if a in self.alias_to_cls and self.alias_to_cls[a] is not cls:
                    other = self.alias_to_cls[a]
                    raise KeyError(
                        f"{self.kind.capitalize()} alias '{a}' уже зарегистрирован "
                        f"для класса {other.__name__}, нельзя повторно использовать его "
                        f"для {cls.__name__}"
                    )

            # 2) Записываем alias -> класс
            for a in norm:
                self.alias_to_cls[a] = cls

            # 3) Обновляем список алиасов для класса (поддерживаем расширение алиасов)
            prev = self.cls_to_aliases.get(cls, tuple())
            # Dict.fromkeys сохраняет порядок и удаляет дубли
            merged = tuple(dict.fromkeys(prev + norm))
            self.cls_to_aliases[cls] = merged

            # 4) Канонический alias фиксируем один раз (при первой регистрации)
            self.cls_to_canonical.setdefault(cls, canonical)

            # 5) Для удобства интроспекции помещаем алиасы в сам класс:
            #    - _aliases          : список всех алиасов
            #    - _canonical_alias  : канонический alias
            # Это не является частью публичного API, но удобно при отладке.
            try:
                setattr(cls, "_aliases", list(merged))
                setattr(cls, "_canonical_alias", self.cls_to_canonical[cls])
            except Exception:
                # Если класс запрещает добавление атрибутов (например, __slots__),
                # то пропускаем этот шаг — registry всё равно будет работать.
                pass

            return cls

        return _wrap

    # -------------------------------------------------------------------------
    # Получение класса и создание экземпляров
    # -------------------------------------------------------------------------
    def resolve_cls(self, alias: str) -> Type[Any]:
        """
        Найти класс компонента по alias (без создания экземпляра).

        Если alias не найден — выбрасывается KeyError с перечнем доступных алиасов.
        """
        key = normalize_alias(alias)
        if key not in self.alias_to_cls:
            available = sorted(self.alias_to_cls.keys())
            hints = difflib.get_close_matches(key, available, n=5, cutoff=0.6)
            hint_msg = f" Возможные варианты: {hints}" if hints else ""

            raise KeyError(
                f"{self.kind.capitalize()} '{alias}' не найден. "
                f"Известные {self.kind}-типы: {available}.{hint_msg}"
            )
        return self.alias_to_cls[key]

    def has(self, alias: str) -> bool:
        """True, если alias известен в данном реестре (после normalize_alias)."""
        key = normalize_alias(alias)
        return key in self.alias_to_cls

    def suggest(
        self,
        alias: str,
        *,
        n: int = 5,
        cutoff: float = 0.6,
        canonical_only: bool = False,
    ) -> List[str]:
        """
        Подсказки (ближайшие совпадения) для неизвестного alias.

        Используется serde-лоадером для формирования ошибок вида UnknownType
        без необходимости перехватывать KeyError и парсить его сообщение.

        canonical_only=True -> искать только среди канонических type (1 на класс).
        """
        key = normalize_alias(alias)
        pool = self.list_canonical_aliases() if canonical_only else self.list_aliases()
        return difflib.get_close_matches(key, pool, n=n, cutoff=cutoff)

    def get(self, alias: str, **kw: Any) -> Any:
        """
        Создать экземпляр компонента по alias.

        **kw передаются непосредственно в конструктор класса:
            cls = resolve_cls(alias)
            return cls(**kw)
        """
        cls = self.resolve_cls(alias)
        return cls(**kw)

    # -------------------------------------------------------------------------
    # Алиасы и канонический тип
    # -------------------------------------------------------------------------
    def canonical(self, obj_or_cls: Union[Any, Type[Any]]) -> str:
        """
        Получить канонический alias по классу или экземпляру.

        Используется при сериализации (dump) для записи `type` в YAML/JSON.
        """
        cls: Type[Any] = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
        if cls not in self.cls_to_canonical:
            raise KeyError(
                f"{self.kind}: класс {cls.__name__} не зарегистрирован в реестре"
            )
        return self.cls_to_canonical[cls]

    def canonicalize_alias(self, alias: str) -> str:
        """
        Привести произвольный alias (включая неканонические/устаревшие алиасы)
        к каноническому type (первому алиасу при регистрации класса).

        Полезно для serde:
        - loader может принять старый alias,
        - но внутренне нормализовать его в канонический вид,
          чтобы dump/диагностика были стабильными.
        """
        cls = self.resolve_cls(alias)
        return self.cls_to_canonical[cls]

    def aliases_of(self, obj_or_cls: Union[Any, Type[Any]]) -> Tuple[str, ...]:
        """
        Получить все алиасы, зарегистрированные для данного класса/экземпляра.

        Удобно для:
        - диагностических сообщений,
        - отображения поддерживаемых имён в документации/GUI,
        - обратной совместимости (старые имена).
        """
        cls: Type[Any] = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
        if cls not in self.cls_to_aliases:
            raise KeyError(f"{self.kind}: класс {cls.__name__} не зарегистрирован в реестре")
        return self.cls_to_aliases[cls]

    def list_aliases(self) -> List[str]:
        """
        Вернуть отсортированный список всех известных alias-ов в данном реестре.
        """
        return sorted(self.alias_to_cls.keys())

    def list_canonical_aliases(self) -> List[str]:
        """
        Вернуть отсортированный список канонических alias-ов (по одному на класс).

        В отличие от list_aliases(), который возвращает ВСЕ алиасы,
        этот метод возвращает ровно один официальный type на компонент.
        """
        return sorted(self.cls_to_canonical.values())

# =============================================================================
# Конкретные реестры: Selector / Transform / Aggregator / Comparator
# =============================================================================

# ВАЖНО: здесь мы специально не импортируем BaseSelector/BaseTransform/...,
# чтобы избежать циклической зависимости с base.py. Реестр работает с классами
# произвольного типа; типы здесь нужны только для статической типизации и
# документации, а не для выполнения.

_SELECTOR_REG = _Registry(kind="selector")
_TRANSFORM_REG = _Registry(kind="transform")
_AGGREGATOR_REG = _Registry(kind="aggregator")
_COMPARATOR_REG = _Registry(kind="comparator")


# Декораторы регистрации — публичная часть API
# -------------------------------------------
# Их удобно использовать в реализации компонентов:
#
#   from .registry import register_selector
#
#   @register_selector(("SMagSelector", "s_mag", "sdb"))
#   class SMagSelector(BaseSelector):
#       ...
#
register_selector = _SELECTOR_REG.register
register_transform = _TRANSFORM_REG.register
register_aggregator = _AGGREGATOR_REG.register
register_comparator = _COMPARATOR_REG.register


# =============================================================================
# Удобные фабрики создания экземпляров по alias
# =============================================================================

def get_selector(alias: str, **kw: Any) -> Any:
    """
    Создать экземпляр selector-а по alias.

    Пример:
        sel = get_selector("SMagSelector", m=1, n=1, db=True)
    """
    return _SELECTOR_REG.get(alias, **kw)


def get_transform(alias: str, **kw: Any) -> Any:
    """
    Создать экземпляр transform-а по alias.
    """
    return _TRANSFORM_REG.get(alias, **kw)


def get_aggregator(alias: str, **kw: Any) -> Any:
    """
    Создать экземпляр aggregator-а по alias.
    """
    return _AGGREGATOR_REG.get(alias, **kw)


def get_comparator(alias: str, **kw: Any) -> Any:
    """
    Создать экземпляр comparator-а по alias.
    """
    return _COMPARATOR_REG.get(alias, **kw)


def list_selectors() -> List[str]:
    """
    Список всех зарегистрированных selector-типов (alias-ов).
    """
    return _SELECTOR_REG.list_aliases()


def list_transforms() -> List[str]:
    """
    Список всех зарегистрированных transform-типов (alias-ов).
    """
    return _TRANSFORM_REG.list_aliases()


def list_aggregators() -> List[str]:
    """
    Список всех зарегистрированных aggregator-типов (alias-ов).
    """
    return _AGGREGATOR_REG.list_aliases()


def list_comparators() -> List[str]:
    """
    Список всех зарегистрированных comparator-типов (alias-ов).
    """
    return _COMPARATOR_REG.list_aliases()


# =============================================================================
# Дополнительный API для serde/диагностики: работа по kind
# =============================================================================

def _get_registry_by_kind(kind: str) -> _Registry:
    """
    Внутренняя функция: вернуть соответствующий реестр по строке kind.

    kind (без учёта регистра):
      - "selector"
      - "transform"
      - "aggregator"
      - "comparator"
    """
    k = str(kind).strip().lower()
    if k == "selector":
        return _SELECTOR_REG
    if k == "transform":
        return _TRANSFORM_REG
    if k == "aggregator":
        return _AGGREGATOR_REG
    if k == "comparator":
        return _COMPARATOR_REG
    raise ValueError("kind должен быть одним из: selector|transform|aggregator|comparator")


def resolve_type(kind: str, alias: str) -> Type[Any]:
    """
    Вернуть класс компонента по паре (kind, alias), не создавая экземпляр.

    Используется при десериализации:
    - для проверки корректности типа в YAML/JSON;
    - для последующей строгой сборки объекта (через отдельную фабрику serde).

    Пример:
        cls = resolve_type("selector", "SMagSelector")
    """
    reg = _get_registry_by_kind(kind)
    return reg.resolve_cls(alias)


def canonical_type(kind: str, obj_or_cls: Any) -> str:
    """
    Вернуть канонический alias для компонента данного вида.

    Используется при сериализации (dump) для записи поля `type`.

    Пример:
        t = canonical_type("comparator", my_comp)
        # t, например, "LEComparator"
    """
    reg = _get_registry_by_kind(kind)
    return reg.canonical(obj_or_cls)


def aliases_of(kind: str, obj_or_cls: Any) -> Tuple[str, ...]:
    """
    Вернуть все алиасы, зарегистрированные для данного компонента (класса/экземпляра).

    Полезно для:
    - диагностических сообщений (подсказки пользователю),
    - отображения в документации,
    - реализации миграций типов (старые/новые имена).
    """
    reg = _get_registry_by_kind(kind)
    return reg.aliases_of(obj_or_cls)


def known_types(kind: str) -> Tuple[str, ...]:
    """
    Вернуть кортеж всех известных `type` (alias) для данного kind.

    Может использоваться:
    - в сообщениях об ошибках (UnknownType),
    - для автодополнения/GUI,
    - в unit-тестах (проверка, что нужные компоненты зарегистрированы).
    """
    reg = _get_registry_by_kind(kind)
    return tuple(reg.list_aliases())


def known_canonical_types(kind: str) -> Tuple[str, ...]:
    """
    Вернуть кортеж канонических `type` (официальных имён) для данного kind.
    """
    reg = _get_registry_by_kind(kind)
    return tuple(reg.list_canonical_aliases())


def has_type(kind: str, alias: str) -> bool:
    """True, если type/alias существует в registry данного kind."""
    reg = _get_registry_by_kind(kind)
    return reg.has(alias)


def suggest_types(
    kind: str,
    alias: str,
    *,
    canonical_only: bool = False,
    n: int = 5,
    cutoff: float = 0.6,
) -> List[str]:
    """
    Подсказки для неизвестного type/alias.

    В serde удобно звать так:
      hints = suggest_types("aggregator", bad_type, canonical_only=True)
    """
    reg = _get_registry_by_kind(kind)
    return reg.suggest(alias, n=n, cutoff=cutoff, canonical_only=canonical_only)


def canonicalize_alias(kind: str, alias: str) -> str:
    """
    Привести alias (возможно, неканонический) к каноническому type.

    Пример:
        canonicalize_alias("comparator", "<=") -> "LEComparator"
        canonicalize_alias("aggregator", "max") -> "MaxAgg"
    """
    reg = _get_registry_by_kind(kind)
    return reg.canonicalize_alias(alias)


def list_types(kind: str) -> List[str]:
    """
    Синоним known_types(kind), но возвращает List[str].

    Удобен в сценариях, где ожидается именно список.
    """
    return list(known_types(kind))


def list_canonical_types(kind: str) -> List[str]:
    """
    Список канонических `type` (официальных имён) для данного kind.
    """
    return list(known_canonical_types(kind))


# =============================================================================
# Публичный экспорт модуля
# =============================================================================

__all__ = [
    # нормализация alias
    "normalize_alias",
    # декораторы регистрации
    "register_selector",
    "register_transform",
    "register_aggregator",
    "register_comparator",
    # фабрики по alias
    "get_selector",
    "get_transform",
    "get_aggregator",
    "get_comparator",
    # списки типов по kind/виду
    "list_selectors",
    "list_transforms",
    "list_aggregators",
    "list_comparators",
    "resolve_type",
    "canonical_type",
    "canonicalize_alias",
    "aliases_of",
    "known_types",
    "known_canonical_types",
    "has_type",
    "suggest_types",
    "list_types",
    "list_canonical_types",
]
