#mwlab/opt/surrogates/rbf.py
"""
RBFSurrogate – тонкая обертка над `sklearn.gaussian_process.RBFKernel`
или `sklearn.gaussian_process.GaussianProcessRegressor` с ядром RBF.

Файл присутствует, чтобы сразу было ясно «куда дописывать» реализацию.
"""

from __future__ import annotations

from .base import BaseSurrogate
from .registry import register

try:
    from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError("RBFSurrogate требует scikit-learn. pip install mwlab[rbf]")

@register("rbf", "sklearn_gp")
class RBFSurrogate(BaseSurrogate):
    supports_uncertainty = True

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RBFSurrogate не реализован в MVP")