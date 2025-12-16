#mwlab/opt/sensitivity/active_subspace.py
"""
active_subspace.py
==================
Градиентный анализ «активного подпространства».

* Пока используем **центральные конечные разности** (FD) – 2·d вызовов
  surrogate на каждую тестовую точку.
* Если surrogate умеет .grad() → сделаем TODO-ветку, что брать autograd.

Возвращаем собственные значения λ (убывающие) и матрицу W (столбцы – векторы).
"""

from __future__ import annotations
from typing import Mapping, Sequence

import numpy as np

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.space import DesignSpace
from mwlab.opt.objectives.specification import Specification
from .utils import batch_eval


# ────────────────────────────────────────────────────────────────────────────
def _fd_gradient(
    surrogate: BaseSurrogate,
    space: DesignSpace,
    spec: Specification,
    x0: np.ndarray,
    eps_rel: float = 1e-4,
) -> np.ndarray:
    """
    Центральная FD-схема: ∂f/∂x_i ≈ (f(x+δ) − f(x−δ)) / (2δ).

    • eps_rel задаёт относительный шаг δ = eps_rel * (hi−lo).
    • f(x) = 1, если passed, и 0 иначе (bool → float).
    """
    d = len(space)
    g = np.zeros(d)
    lows, highs = space.bounds()

    for i in range(d):
        delta = eps_rel * (highs[i] - lows[i])
        x_plus, x_minus = x0.copy(), x0.copy()
        x_plus[i] += delta
        x_minus[i] -= delta

        pts = [space.dict(x_plus), space.dict(x_minus)]
        f_plus, f_minus = batch_eval(surrogate, pts, spec).astype(float)

        g[i] = (f_plus - f_minus) / (2.0 * delta)

    return g


# ────────────────────────────────────────────────────────────────────────────
def run_active_subspace(
    surrogate: BaseSurrogate,
    space: DesignSpace,
    spec: Specification,
    *,
    k: int = 5,
    n_samples: int = 5000,
    method: str = "fd",               # 'fd' | 'grad'
    rng: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    k : int
        Число eigen-векторов, которые имеет смысл оставить (для пользователя).
        На расчёт λ, W не влияет.
    method : 'fd' | 'grad'
        Как брать градиент: finite-diff или surrogate.grad().

    Returns
    -------
    lam : (d,) eigenvalues  (отсортированы по убыванию)
    W   : (d,d) eigenvectors (W[:, i] – i-й вектор)
    """
    rng = np.random.default_rng(rng)
    d = len(space)
    C = np.zeros((d, d))              # матрица ковариации ∇f

    lows, highs = space.bounds()

    for _ in range(n_samples):
        # 1) Случайная точка внутри диапазона
        x = rng.uniform(lows, highs)

        # 2) ∇f
        if method == "fd":
            g = _fd_gradient(surrogate, space, spec, x)
        elif method == "grad" and hasattr(surrogate, "grad"):
            g = surrogate.grad(space.dict(x))          # type: ignore[attr-defined]
            g = np.asarray(g, dtype=float)
        else:
            raise ValueError("method must be 'fd' или 'grad' (если реализовано)")

        # 3) Накапливаем матрицу C
        C += np.outer(g, g)

    C /= n_samples

    # 4) Собственные значения / векторы
    lam, W = np.linalg.eigh(C)        # отсортированы по возрастанию
    idx = np.argsort(lam)[::-1]       # по убыванию
    lam = lam[idx]
    W = W[:, idx]

    return lam[:k], W[:, :k]          # отдаём только k «ведущих» направлений
