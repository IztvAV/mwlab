#mwlab/opt/sensitivity/core.py
"""
core.py
=======
Единый фасад `SensitivityAnalyzer`, объединяющий Morris, Sobol, Active-Subspace.

* Все методы используют SALib/NumPy под капотом.
* По-умолчанию рисуют bar-plots (можно `plot=False`).
"""

from __future__ import annotations
from typing import Sequence, Mapping, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.space import DesignSpace
from mwlab.opt.objectives.specification import Specification

from . import morris as _morris
from . import sobol  as _sobol
from . import active_subspace as _asub
from .utils import barplot_dataframe


# ────────────────────────────────────────────────────────────────────────────
class SensitivityAnalyzer:
    """
    Parameters
    ----------
    surrogate : BaseSurrogate
        Уже обученная модель прямой задачи X→S.
    design_space : DesignSpace
        Границы переменных.
    specification : Specification
        Полное ТЗ, содержащее *как минимум один* критерий.
    """

    def __init__(
        self,
        surrogate: BaseSurrogate,
        design_space: DesignSpace,
        specification: Specification,
    ):
        if specification is None:
            raise ValueError("specification обязательна для анализа чувствительности.")
        self.sur   = surrogate
        self.space = design_space
        self.spec  = specification

    # ================================================================= Morris
    def morris(
        self,
        N: int = 20,
        *,
        sampler: str = "sobol",
        sampler_kw: Mapping | None = None,
        rng: int = 0,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Метод Морриса (screening).

        Возвращает DataFrame (index = param names, cols = mu_star, sigma).
        """
        df = _morris.run_morris(
            self.sur, self.space, self.spec,
            N=N, sampler=sampler, sampler_kw=sampler_kw or {}, rng=rng,
        )
        if plot:
            barplot_dataframe(df["mu_star"], title="Morris μ* (важность)", ylabel="μ*")
        return df

    # ================================================================= Sobol
    def sobol(
        self,
        params: Sequence[str] | str = "auto",
        *,
        n_base: int = 1024,
        second_order: bool = False,
        rng: int = 0,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Индексы Соболя.

        • `second_order=True` → в df.attrs['S2_pairs'] лежит Series с (i,j).
        """
        df = _sobol.run_sobol(
            self.sur, self.space, self.spec,
            params=params,
            n_base=n_base,
            second_order=second_order,
            rng=rng,
        )
        if plot:
            barplot_dataframe(df["S1"], title="Sobol S1 (первый порядок)", ylabel="S1")
        return df

    # =========================================================== Active-Subspace
    def active_subspace(
        self,
        *,
        k: int = 5,
        n_samples: int = 5000,
        method: str = "fd",
        rng: int = 0,
        plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает (λ, W) – первые k собственных значений и векторов.
        """
        lam, W = _asub.run_active_subspace(
            self.sur, self.space, self.spec,
            k=k, n_samples=n_samples, method=method, rng=rng,
        )
        if plot:
            plt.figure()
            plt.semilogy(range(1, len(lam) + 1), lam, marker="o")
            plt.xlabel("Индекс")
            plt.ylabel("Eigenvalue")
            plt.title("Спектр матрицы C (Active Subspace)")
            plt.grid(True, which="both", ls=":")
            plt.tight_layout()
            plt.show()
        return lam, W
