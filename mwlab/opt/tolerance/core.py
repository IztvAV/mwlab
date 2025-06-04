#mwlab/opt/tolerance/core.py
"""
core.py
=======
Упрощённый фасад «всё-в-одном» для пользователя.
"""

from __future__ import annotations
from typing import Sequence, Dict, Tuple

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.space import DesignSpace
from mwlab.opt.objectives.specification import Specification

from .oat import IndividualBoundsFinder
from .joint_box import JointBoxOptimizer


class ToleranceAnalyzer:
    """
    Высокоуровневый интерфейс:

    ta = ToleranceAnalyzer(sur, space, spec)
    δ_hi = ta.upper_bounds(top10)
    δ_ok = ta.optimize_box(δ_hi)
    """

    def __init__(
        self,
        surrogate: BaseSurrogate,
        design_space: DesignSpace,
        specification: Specification,
        *,
        seed: int = 0,
    ):
        self.oat = IndividualBoundsFinder(
            surrogate, design_space, specification, seed=seed
        )
        self.joint = JointBoxOptimizer(
            surrogate, design_space, specification, seed=seed
        )

    # ------------------- публичные методы -----------------------------------
    def upper_bounds(
        self,
        params: Sequence[str],
        *,
        target_yield: float = 0.99,
        init_frac: float = 0.02,
        max_iter: int = 12,
        n_mc: int = 4096,
        asym: bool = False,
    ):
        return self.oat.find(
            params,
            target_yield=target_yield,
            init_frac=init_frac,
            max_iter=max_iter,
            n_mc=n_mc,
            asym=asym,
        )

    def optimize_box(
        self,
        delta_upper: Dict[str, float | Tuple[float, float]],
        *,
        target_yield: float = 0.99,
        n_mc: int = 8192,
        method: str = "slsqp",
        max_iter: int = 200,
        verbose: bool = False,
    ):
        return self.joint.optimize(
            delta_upper=delta_upper,
            target_yield=target_yield,
            n_mc=n_mc,
            method=method,
            max_iter=max_iter,
            verbose=verbose,
        )
