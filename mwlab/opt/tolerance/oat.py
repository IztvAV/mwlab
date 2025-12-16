#mwlab/opt/tolerance/oat.py
"""
oat.py
======
**OAT (One-at-a-Time) Binary Search**:

Для каждого параметра i
  1) Держим остальные на номинале;
  2) Увеличиваем радиус δᵢ, пока `Yield ≥ target`;
  3) Двоичным поиском уточняем максимальный δᵢ.

Возвращает *симметричные* ±δ по умолчанию, но может искать
`(δ_minus, δ_plus)`, если `asym=True`.
"""

from __future__ import annotations
from typing import Dict, Tuple, Sequence
from tqdm import tqdm

import numpy as np

from mwlab.opt.design.space import DesignSpace
from mwlab.opt.design.samplers import get_sampler
from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.objectives.specification import Specification


# ────────────────────────────────────────────────────────────────────────────
class IndividualBoundsFinder:
    """
    Parameters
    ----------
    surrogate, design_space, specification
        Те же объекты, что и у SensitivityAnalyzer.
    seed : int
        Сид для всех внутренних генераторов → воспроизводимость.
    """

    # ---------------- init
    def __init__(
        self,
        surrogate: BaseSurrogate,
        design_space: DesignSpace,
        specification: Specification,
        *,
        seed: int = 0,
    ):
        self.sur = surrogate
        self.space = design_space
        self.spec = specification
        self.seed = int(seed)

        # кэш “центров” и списка имён – часто пользуемся
        self._lows, self._highs = self.space.bounds()
        self._centers = 0.5 * (self._lows + self._highs)
        self._names = self.space.names()

    # ---------------- публичный метод
    def find(
        self,
        params: Sequence[str],
        *,
        target_yield: float = 0.99,
        init_frac: float = 0.02,
        max_iter: int = 12,
        n_mc: int = 4096,
        asym: bool = False,
    ) -> Dict[str, float | Tuple[float, float]]:
        """
        Возвращает словарь {name: δ}  (или (δ_minus, δ_plus) при asym=True).

        • `init_frac` – стартовый радиус = init_frac·(hi−lo).
        • `max_iter`  – шагов бинарного поиска.
        • `n_mc`      – размер Монте-Карло при оценке yield.
        """
        deltas: Dict[str, float | Tuple[float, float]] = {}
        for p in tqdm(params, desc="Поиск допусков для каждого параметра"):            # TODO: parallel
            if p not in self._names:
                raise KeyError(f"параметр '{p}' не найден в DesignSpace")
            idx = self._names.index(p)
            width = self._highs[idx] - self._lows[idx]

            if asym:
                d_minus = self._search_one(
                    p, side="minus", init=init_frac * width,
                    target=target_yield, max_iter=max_iter, n_mc=n_mc
                )
                d_plus = self._search_one(
                    p, side="plus", init=init_frac * width,
                    target=target_yield, max_iter=max_iter, n_mc=n_mc
                )
                deltas[p] = (-d_minus, d_plus)
            else:
                d_sym = self._search_one(
                    p, side="both", init=init_frac * width,
                    target=target_yield, max_iter=max_iter, n_mc=n_mc
                )
                deltas[p] = d_sym
        return deltas

    # ---------------- private ------------------------------------------------
    def _search_one(
        self,
        name: str,
        *,
        side: str,
        init: float,
        target: float,
        max_iter: int,
        n_mc: int,
    ) -> float:
        """
        Бинарный поиск верхней δ для одного параметра.
        side: 'minus' | 'plus' | 'both'
        """
        idx = self._names.index(name)

        # ---- экспоненциальный разгон ------------------------------------
        lo, hi = 0.0, init
        while True:
            if self._yield_for(name, hi, side, n_mc) >= target:
                lo, hi = hi, hi * 2
            else:
                break

        # ---- бинарный поиск --------------------------------------------
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if self._yield_for(name, mid, side, n_mc) >= target:
                lo = mid
            else:
                hi = mid
        return lo

    # --------------------------------------------------------------------
    def _yield_for(
        self, name: str, delta: float, side: str, n_mc: int
    ) -> float:
        """
        Собирает временный DesignSpace, где только parameter `name`
        варьируется на ±delta (или отдельно для minus/plus side),
        остальные фиксированы на центрах.
        """
        # --- формируем радиусы -----------------------------------------
        if side == "both":
            delta_dict = {name: delta}
        elif side == "minus":
            delta_dict = {name: (-delta, 0.0)}
        elif side == "plus":
            delta_dict = {name: (0.0, delta)}
        else:
            raise ValueError(side)

        # --- формируем «полный» словарь центров и радиусов --------------
        centers_all = {n: self._centers[self._names.index(n)] for n in self._names}
        delta_full = {n: 0.0 for n in self._names}  # δ = 0 для неактивных
        delta_full.update(delta_dict)  # заменяем активный

        sub_space = DesignSpace.from_center_delta(
            centers_all,
            delta=delta_full,
            mode="abs",
        )

        pts = sub_space.sample(n_mc, sampler=get_sampler("latin", rng=self.seed),
                               reject_invalid=False)
        preds = self.sur.batch_predict(pts)
        if isinstance(preds[0], (bool, np.bool_)):
            return float(np.mean(preds))

        return float(np.mean([self.spec.is_ok(net) for net in preds]))
