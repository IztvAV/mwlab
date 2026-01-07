# mwlab/opt/surrogates/__init__.py
"""
mwlab.opt.surrogates
====================

Публичный фасад подсистемы суррогат-моделей + поддержка *lazy-loading*.

Цели
----
1) Дать пользователю удобный API:
   - get_surrogate("nn", **kwargs)
   - декоратор register("alias1", "alias2", ...)

2) НЕ тащить тяжёлые зависимости (torch/botorch/sklearn и т.п.) при простом:
       import mwlab.opt.surrogates

   Для этого:
   - registry вынесен в отдельный модуль .registry
   - реализации (nn/gp/rbf/...) НЕ импортируются здесь напрямую
   - при запросе get_surrogate(...) нужный модуль подгружается динамически

3) Сохранить “приятный” импорт классов, но тоже лениво:
       from mwlab.opt.surrogates import NNSurrogate

   Это поддерживается через PEP 562: __getattr__ + __dir__.

Важно для авторов новых суррогатов
---------------------------------
Рекомендуемый паттерн в файле реализации, например mwlab/opt/surrogates/nn.py:

    from .registry import register
    from .base import BaseSurrogate

    @register("nn", "NNSurrogate")
    class NNSurrogate(BaseSurrogate):
        ...

Так мы:
- избегаем циклических импортов (модуль реализации не импортирует пакет целиком),
- гарантируем, что импорт реализации зарегистрирует класс в registry.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Базовый класс доступен всегда и лёгкий по зависимостям.
from .base import BaseSurrogate

# Фабрика и декоратор регистрации живут в registry.
from .registry import get_surrogate, register, available_surrogates

# ---------------------------------------------------------------------------
# Lazy-доступ к конкретным классам (по имени атрибута)
# ---------------------------------------------------------------------------
# Это НЕ обязано покрывать все алиасы get_surrogate(...).
# Это именно поддержка "from mwlab.opt.surrogates import NNSurrogate" без
# тяжёлых импортов при импорте пакета.
_LAZY_CLASS_IMPORTS: dict[str, str] = {
    # Имя атрибута -> относительный модуль, где он определён
    "NNSurrogate": ".nn",
    "GPSurrogate": ".gp",
    "RBFSurrogate": ".rbf",
}


def __getattr__(name: str) -> Any:
    """
    PEP 562: ленивое получение атрибутов модуля.

    Если пользователь делает:
        from mwlab.opt.surrogates import NNSurrogate
    то интерпретатор попытается взять атрибут NNSurrogate из этого модуля.
    Здесь мы:
      - импортируем соответствующий модуль реализации (.nn / .gp / .rbf),
      - возвращаем нужный атрибут.

    Если модуль не существует или не установлены опциональные зависимости —
    будет поднята исходная ImportError (это нормально и ожидаемо).
    """
    if name in _LAZY_CLASS_IMPORTS:
        mod = import_module(_LAZY_CLASS_IMPORTS[name], package=__name__)
        try:
            return getattr(mod, name)
        except AttributeError as e:
            # Защита от ситуации “модуль импортировался, но класс переименовали/удалили”.
            raise AttributeError(
                f"Модуль {_LAZY_CLASS_IMPORTS[name]!r} импортирован, но атрибут {name!r} в нём не найден."
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    PEP 562: чтобы IDE/dir() видели ленивые атрибуты.
    """
    return sorted(list(globals().keys()) + list(_LAZY_CLASS_IMPORTS.keys()))


# Публичный экспорт.
# Примечание: мы МОЖЕМ включать GPSurrogate/RBFSurrogate в __all__ —
# при `from ... import *` будут вызваны getattr(...) и выполнен lazy-import.
# Это поведение приемлемо: star-import и так “тяжёлая” операция.
__all__ = [
    "BaseSurrogate",
    "register",
    "get_surrogate",
    "available_surrogates",
    "NNSurrogate",
    "GPSurrogate",
    "RBFSurrogate",
]
