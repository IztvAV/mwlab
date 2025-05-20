#mwlab/opt/analysis/box_finder.py
"""
mwlab/opt/analysis/box_finder.py
================================
Поиск **технологического гипер-куба** (tolerance-box) с целевой
вероятностью выхода годных *yield ≥ α*.

Алгоритм реализует «LHS-global + zoom» схему из
H. Zhang et al., *IEEE TAP*, 2018 — но с возможностью:

* **sym**  – симметричный куб: δ⁺ = δ⁻, центр фиксирован в X₀;
* **asym** – асимметрия: независимые δ⁺ / δ⁻ и сдвиг центра (ищется).

Стратегии поиска
----------------
* ``strategy="lhs_global"`` – глобальный перебор «угловых» точек LHS
  (полностью реализовано).
* ``strategy="axis_zoom"``  – прежний локальный бинарный zoom
  (оставлен TODO; можно включить при нехватке времени).

Идея «lhs_global»
-----------------
#. Вокруг X₀ строится расширенный куб **C₀ = [X₀ ± δ₀]**.
#. В **C₀** генерируется *N* точек *Sobol/LHS* (предпочтительна степень 2).
#. Surrogate → pass/fail; *каждая pass-точка* рассматривается как
   «угол» кандидата:

   *sym*  → δ = |x – X₀|, бокс `[X₀ ± δ]`
   *asym* → low = min(x, X₀), high = max(x, X₀)
#. Для кандидата подсчитываем **эмпирический yield** ― долю *предварительно
   сгенерированных* выборок, попавших в бокс и прошедших ТЗ.
   Учитываем бокс **только если** yield ≥ α.
#. Из всех валидных кандидатов берем бокс с **максимальным объемом**.
   Исследуемая зона «zoom-in» до 120 % найденного бокса и повторяем цикл.

**Гарантия**: если α = 1 → *все* сэмплы в боксе прошли спецификацию;
для α < 1 – гарантируется только вероятностная оценка.

Пример
------
>>> finder = BoxFinder(strategy="lhs_global", mode="asym",
...                    target_yield=0.95, n_lhs=4096, rng=42)
>>> cube = finder.find(center_dict, design_space,
...                    surrogate, specification,
...                    delta_init=1e-4)
>>> print(cube["w"])         # (-7.9e-5, 8.3e-5)
"""
from __future__ import annotations

from typing import Dict, Tuple, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

from ..design.space import DesignSpace
from ..surrogates import BaseSurrogate
from ..objectives.specification import Specification


# ────────────────────────────────────────────────────────────────────────────
class BoxFinder:
    """
    Параметры конструктора
    ----------------------
    target_yield : float ∈ (0, 1]           – требуемая доля PASS-точек.
    n_lhs        : int                      – число Sobol/LHS-точек на итерацию.
    zoom_ratio   : float > 1                – во сколько раз расширяем
                                              поисковую зону (рекомендуется 1.2).
    eps_rel      : float                    – относительная сходимость по объёму.
    max_zoom_iter: int                      – максимум итераций «zoom-in».
    strategy     : 'lhs_global' | 'axis_zoom'
    mode         : 'sym' | 'asym'
    rng          : int | np.random.Generator | None
    """

    def __init__(
        self,
        *,
        target_yield: float = 0.95,
        n_lhs: int = 4096,
        zoom_ratio: float = 1.20,
        eps_rel: float = 1e-3,
        max_zoom_iter: int = 6,
        strategy: str = "lhs_global",
        mode: str = "sym",
        rng: int | np.random.Generator | None = None,
    ):
        if not (0 < target_yield <= 1):
            raise ValueError("target_yield must be in (0,1]")
        self.alpha = float(target_yield)
        self.N = int(n_lhs)
        self.zoom = float(zoom_ratio)
        self.eps = float(eps_rel)
        self.max_iter = int(max_zoom_iter)
        if strategy not in ("lhs_global", "axis_zoom"):
            raise ValueError("strategy must be 'lhs_global'|'axis_zoom'")
        if mode not in ("sym", "asym"):
            raise ValueError("mode must be 'sym'|'asym'")
        self.strategy = strategy
        self.mode = mode
        self._rng = np.random.default_rng(rng)

    # ======================================================================
    #                         PUBLIC ENTRY POINT
    # ======================================================================
    def find(
        self,
        center: Mapping[str, float],
        space: DesignSpace,
        surrogate: BaseSurrogate,
        spec: Specification,
        *,
        delta_init: float | Mapping[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Возвращает словарь ``{param: (-δ_left, +δ_right)}``.

        *В режиме 'sym' —  δ_left = δ_right;
        в 'asym' — возможно различие.*
        """
        if self.strategy == "lhs_global":
            return self._find_lhs_global(center, space, surrogate, spec, delta_init)
        raise NotImplementedError("axis_zoom strategy: TODO")

    # ======================================================================
    #                       INTERNAL — LHS-GLOBAL
    # ======================================================================
    # ---------- helpers ----------------------------------------------------
    def _delta_to_array(self, delta_init, names):
        if isinstance(delta_init, Mapping):
            return np.asarray([float(delta_init[n]) for n in names])
        return np.full(len(names), float(delta_init))

    def _sobol_points(self, lows: np.ndarray, highs: np.ndarray, n: int):
        """Равномерные точки внутри прямоугольника [lows, highs]."""
        d = lows.size
        # если n степень 2 → Sobol random_base2 дает ортогональную сетку
        if (n & (n - 1)) == 0:               # степень 2
            eng = qmc.Sobol(d, scramble=True, seed=self._rng)
            U = eng.random_base2(int(np.log2(n)))
        else:
            eng = qmc.Sobol(d, scramble=True, seed=self._rng)
            U = eng.random(n)
        return lows + U * (highs - lows)

    # ---------- main loop --------------------------------------------------
    def _find_lhs_global(
        self,
        center_dict: Mapping[str, float],
        space: DesignSpace,
        surrogate: BaseSurrogate,
        spec: Specification,
        delta_init: float | Mapping[str, float],
    ) -> Dict[str, Tuple[float, float]]:

        names: Sequence[str] = list(space.names())   # сохраняем порядок
        center = np.asarray([center_dict[n] for n in names])
        delta = self._delta_to_array(delta_init, names)

        best_vol = -np.inf
        best_low: np.ndarray | None = None
        best_high: np.ndarray | None = None

        for zoom_it in range(self.max_iter):
            lows, highs = center - delta, center + delta

            # 1) облако Sobol в текущей зоне
            cloud = self._sobol_points(lows, highs, self.N)

            # 2) surrogate PASS / FAIL  (batch)
            dict_cloud = [{n: float(v) for n, v in zip(names, row)} for row in cloud]
            ok_mask = np.array([spec.is_ok(net)
                                for net in surrogate.batch_predict(dict_cloud)])

            if not ok_mask.any():
                # все FAIL – уменьшаем δ и повторяем
                delta *= 0.5
                continue

            # 3) перебор всех PASS-точек как потенциальных «углов»
            pass_pts = cloud[ok_mask]
            # матрица (N_pass, d, 2) для быстрого отбора точек внутри кандидата

            for p in pass_pts:
                if self.mode == "sym":
                    rad = np.abs(p - center)
                    cand_low, cand_high = center - rad, center + rad
                else:  # asym
                    cand_low = np.minimum(p, center)
                    cand_high = np.maximum(p, center)

                # выбираем cloud-индексы, попавшие внутрь кандидата
                inside = np.all((cloud >= cand_low) & (cloud <= cand_high), axis=1)
                yield_val = np.mean(ok_mask[inside])

                if yield_val < self.alpha:
                    continue   # не проходит по целевой вероятности

                volume = np.prod(cand_high - cand_low)
                if volume > best_vol:
                    best_vol, best_low, best_high = volume, cand_low.copy(), cand_high.copy()

            # 4) проверка сходимости по объему
            if zoom_it > 0:
                best_vol_prev = getattr(self, "_prev_vol", None)
                if best_vol_prev and abs(best_vol - best_vol_prev) / best_vol_prev < self.eps:
                    break
            self._prev_vol = best_vol  # сохранить для следующего шага

            # 5) zoom-in исследуемой зоны
            if best_low is None:      # ни один кандидат не прошел – shrink
                delta *= 0.5
            else:
                span = (best_high - best_low) * 0.5 * self.zoom
                center = 0.5 * (best_low + best_high)
                delta = span

        if best_low is None or best_high is None:
            raise RuntimeError("BoxFinder: не удалось найти допустимый бокс.")

        if self.mode == "sym":
            rad = 0.5 * (best_high - best_low)
            return {n: (-r, r) for n, r in zip(names, rad)}

        # asym → δ⁻ / δ⁺ и центр = середина найденного бокса
        c_star = 0.5 * (best_low + best_high)
        d_minus = c_star - best_low
        d_plus = best_high - c_star
        return {n: (-dm, dp) for n, dm, dp in zip(names, d_minus, d_plus)}

    # ---------- repr -------------------------------------------------------
    def __repr__(self):  # pragma: no cover
        return (f"BoxFinder(strategy={self.strategy}, mode={self.mode}, "
                f"α={self.alpha}, N={self.N}, zoom={self.zoom})")
