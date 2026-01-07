# mwlab/opt/surrogates/registry.py
"""
mwlab.opt.surrogates.registry
=============================

Единый реестр (registry) суррогат-моделей + фабрика get_surrogate(...).

Зачем отдельный модуль
----------------------
1) Избавляем __init__.py от “магии” с обязательными импортами реализаций.
2) Резко уменьшаем риск циклических импортов.
3) Получаем естественную точку расширения для lazy-loading.

Ключевая идея lazy-loading
--------------------------
- Реестр заполняется декоратором @register(...) при импорте модулей реализации.
- get_surrogate(name, ...) при необходимости пытается импортировать нужный модуль
  динамически (по таблице _LAZY_ALIAS_TO_MODULE), чтобы регистрация произошла.

Это позволяет:
- держать `import mwlab.opt.surrogates` лёгким,
- подключать опциональные суррогаты (GP/RBF) только если зависимости установлены.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Iterable, Optional, Tuple, Type

from .base import BaseSurrogate

# ---------------------------------------------------------------------------
# Сам реестр: alias -> класс суррогата
# ---------------------------------------------------------------------------
_SURR_REG: Dict[str, Type[BaseSurrogate]] = {}

# ---------------------------------------------------------------------------
# Таблица ленивых импортов
# ---------------------------------------------------------------------------
# Эта таблица отвечает на вопрос:
# “Если пользователь запросил alias X, какой модуль нужно импортировать,
#  чтобы соответствующий класс зарегистрировался в _SURR_REG?”
#
# Важно:
# - ключи здесь — это *возможные* имена, которые пользователь передаст в get_surrogate
# - значения — это относительные модули (внутри пакета mwlab.opt.surrogates)
#
# Рекомендация:
# - всегда регистрируйте в реализации и “короткий” alias, и имя класса:
#     @register("nn", "NNSurrogate")
# Тогда пользователю удобно, а таблица может быть простой.
_LAZY_ALIAS_TO_MODULE: Dict[str, str] = {
    # NN
    "nn": ".nn",
    "NNSurrogate": ".nn",
    "nnsurrogate": ".nn",
    "pytorch": ".nn",
    "inv_nn": ".nn",
    "inverse_nn": ".nn",

    # GP (опционально)
    "gp": ".gp",
    "GPSurrogate": ".gp",
    "gpsurrogate": ".gp",
    "botorch_gp": ".gp",

    # RBF (опционально)
    "rbf": ".rbf",
    "RBFSurrogate": ".rbf",
    "rbfsurrogate": ".rbf",
    "sklearn_gp": ".rbf",
}


def register(*aliases: str) -> Callable[[Type[BaseSurrogate]], Type[BaseSurrogate]]:
    """
    Декоратор регистрации суррогата.

    Пример:
        from .registry import register

        @register("nn", "NNSurrogate")
        class NNSurrogate(BaseSurrogate):
            ...

    Замечания по качеству API
    -------------------------
    - aliases не нормализуются агрессивно: сохраняем то, что дал автор.
      Однако для удобства пользователей обычно регистрируют и lower-case вариант.
    - При повторной регистрации alias возбуждаем KeyError: это защищает от “тихих”
      конфликтов при импорте нескольких модулей.
    """
    if not aliases:
        raise ValueError("register(...): нужно указать хотя бы один alias")

    def _wrap(cls: Type[BaseSurrogate]) -> Type[BaseSurrogate]:
        # Защита от случайной регистрации не-суррогата.
        if not issubclass(cls, BaseSurrogate):
            raise TypeError(
                f"register(...): класс {cls.__name__} должен наследоваться от BaseSurrogate"
            )

        for a in aliases:
            name = str(a).strip()
            if not name:
                raise ValueError("register(...): пустой alias недопустим")
            if name in _SURR_REG:
                raise KeyError(f"Surrogate alias '{name}' already registered")
            _SURR_REG[name] = cls

        # Для serde/интроспекции удобно хранить список алиасов на самом классе.
        cls._aliases = tuple(str(a).strip() for a in aliases)  # type: ignore[attr-defined]
        return cls

    return _wrap


def _try_lazy_import(requested_name: str) -> None:
    """
    Попытаться лениво импортировать модуль реализации по имени requested_name.

    Если requested_name неизвестен таблице _LAZY_ALIAS_TO_MODULE — ничего не делаем.
    Если известен — импортируем модуль. Он должен:
      - импортировать register,
      - вызвать @register(...) на классе,
      - тем самым заполнить _SURR_REG.
    """
    key = str(requested_name).strip()
    if not key:
        return

    # Пробуем как есть, затем lower-case (часто пользователь вводит “NN”/“Nn” и т.п.).
    candidates = [key]
    kl = key.lower()
    if kl != key:
        candidates.append(kl)

    module: Optional[str] = None
    for c in candidates:
        module = _LAZY_ALIAS_TO_MODULE.get(c)
        if module is not None:
            break

    if module is None:
        return

    # Импорт может упасть:
    # - файл модуля отсутствует,
    # - не установлены опциональные зависимости,
    # - ошибка внутри модуля.
    #
    # Мы НЕ “глотаем” такие ошибки без следа: поднимаем ImportError
    # с понятным сообщением и пробрасываем original exception как __cause__.
    try:
        import_module(module, package=__package__)
    except Exception as e:
        raise ImportError(
            f"Не удалось импортировать суррогат для alias '{requested_name}'. "
            f"Попытка импорта модуля {module!r} завершилась ошибкой. "
            f"Возможно, не установлены опциональные зависимости."
        ) from e


def get_surrogate(name: str, **kwargs) -> BaseSurrogate:
    """
    Фабрика: создать экземпляр суррогата по alias.

    Lazy-loading:
    -------------
    Если alias ещё не зарегистрирован, get_surrogate попытается импортировать
    соответствующий модуль реализации, чтобы регистрация произошла автоматически.

    Ошибки:
    -------
    - KeyError: если alias неизвестен даже после попытки lazy-import
    - ImportError: если alias известен, но модуль реализации не смог импортироваться
    """
    key = str(name).strip()
    if not key:
        raise ValueError("get_surrogate(name): пустое имя недопустимо")

    # Пытаемся найти без импортов
    if key not in _SURR_REG and key.lower() not in _SURR_REG:
        _try_lazy_import(key)

    # Повторяем поиск после lazy-import
    cls = _SURR_REG.get(key) or _SURR_REG.get(key.lower())
    if cls is None:
        avail = ", ".join(available_surrogates(try_import=False))
        raise KeyError(
            f"Surrogate '{name}' not found. Available (registered/known): [{avail}]"
        )

    # Создаём экземпляр
    return cls(**kwargs)  # type: ignore[call-arg]


def available_surrogates(*, try_import: bool = False) -> Tuple[str, ...]:
    """
    Вернуть список доступных имён суррогатов.

    Параметры
    ---------
    try_import : bool
        - False (по умолчанию): вернуть “известные” имена
          = зарегистрированные сейчас + ключи из таблицы lazy-импортов.
          Это быстрый и безопасный режим (не тянет опциональные зависимости).
        - True: попытаться импортировать все “известные” модули из таблицы lazy-импортов.
          Если зависимости не установлены — соответствующие импорты будут пропущены.
          После этого вернётся список реально зарегистрированных алиасов.

    Зачем это нужно
    ---------------
    - Для UI/CLI (подсказки пользователю).
    - Для тестов: можно проверить, что при наличии зависимостей gp/rbf действительно
      регистрируются.
    """
    if try_import:
        # Аккуратно попробуем импортировать все известные реализации.
        # Ошибки “нет зависимости” НЕ должны ломать introspection.
        for alias in list(_LAZY_ALIAS_TO_MODULE.keys()):
            if alias in _SURR_REG:
                continue
            try:
                _try_lazy_import(alias)
            except ImportError:
                pass
        return tuple(sorted(_SURR_REG.keys()))

    # Без импорта: объединяем реально зарегистрированные + “известные” lazy-алиасы.
    known = set(_SURR_REG.keys()) | set(_LAZY_ALIAS_TO_MODULE.keys())
    return tuple(sorted(known))


__all__ = [
    "register",
    "get_surrogate",
    "available_surrogates",
]
