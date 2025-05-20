#mwlab/opt/surrogates/__init__.py
"""
mwlab.opt.surrogates
====================
Публичный фасад + *registry* для всех суррогат-моделей.

* `get_surrogate("alias", **kw)` — фабрика по строковому имени.
* `register("alias")`          — декоратор для добавления новых моделей.

По-умолчанию в реестре один рабочий класс – **NNSurrogate** (обертка над
обученным `BaseLModule`).  Заглушки `GPSurrogate` и `RBFSurrogate`
появятся при установке опциональных зависимостей (`botorch`, `scikit-learn`).
"""

from __future__ import annotations
from typing import Dict, Callable, Type

from .base import BaseSurrogate

# ────────────────────────── registry ──────────────────────────
_SURR_REG: Dict[str, Type[BaseSurrogate]] = {}


def register(*aliases: str) -> Callable[[Type[BaseSurrogate]], Type[BaseSurrogate]]:
    """Декоратор: регистрирует класс-суррогат под несколькими alias-ами."""
    def _wrap(cls: Type[BaseSurrogate]):
        for name in aliases:
            if name in _SURR_REG:
                raise KeyError(f"Surrogate alias '{name}' already registered")
            _SURR_REG[name] = cls
        cls._aliases = aliases           # type: ignore[attr-defined]
        return cls
    return _wrap


def get_surrogate(name: str, **kwargs) -> BaseSurrogate:
    """Фабрика: возвращает surrogate-экземпляр по alias-у."""
    if name not in _SURR_REG:
        raise KeyError(f"Surrogate '{name}' not found. Available: {list(_SURR_REG)}")
    return _SURR_REG[name](**kwargs)     # type: ignore[call-arg]


# ────────────────────────── импорты, которые заполняют реестр ─────────
from .nn import NNSurrogate        # noqa: E402,F401

# опциональные — не падаем, если зависимостей нет
try:
    from .gp import GPSurrogate    # noqa: E402,F401
except ImportError:                # pragma: no cover
    pass
try:
    from .rbf import RBFSurrogate  # noqa: E402,F401
except ImportError:                # pragma: no cover
    pass

__all__ = ["BaseSurrogate", "NNSurrogate", "GPSurrogate",
           "RBFSurrogate", "get_surrogate", "register"]
