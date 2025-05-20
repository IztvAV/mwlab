#mwlab/opt/surrogates/gp.py
"""
GPSurrogate
===========

Если установлен **BoTorch** + **GPyTorch**, этот файл автоматически
регистрируется под alias-ами ``"gp"``, ``"botorch_gp"``.

Сейчас здесь только базовый каркас, чтобы не путаться в общей концепции.
"""

from __future__ import annotations

try:
    import botorch  # noqa: F401
    from .base import BaseSurrogate
    from . import register
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError("GPSurrogate требует 'botorch'. Установите: pip install mwlab[bo]")


# ────────────────────────────────────────────────────────────────────────────
#                            GPSurrogate
# ────────────────────────────────────────────────────────────────────────────

@register("gp", "botorch_gp")
class GPSurrogate(BaseSurrogate):
    """Kriging / Gaussian-Process surrogate (BoTorch SingleTaskGP).

    *Пока заглушка*: методы raise NotImplementedError.
    """

    supports_uncertainty: bool = True

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GPSurrogate не реализован в MVP")

    # все остальное — см. BaseSurrogate
