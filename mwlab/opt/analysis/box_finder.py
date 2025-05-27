#mwlab/opt/analysis/box_finder.py
"""
Поиск технологического tolerance-box (гипер-куба допусков)
----------------------------------------------------------

Модуль реализует два алгоритма:

* **hypervolume_tol** – поиск бокса максимального гиперобъёма
  через облако Sobol-точек и «угловые» pass-кандидаты
  (аналог LHS-global-zoom из Zhang et al., IEEE TAP 2018);

* **axis_tol** – поосевое («+» и «–» грани отдельно) расширение
  tolerance-box-а до тех пор, пока сохраняется заданный yield.


Пример
------
>>> from mwlab.opt.analysis.box_finder import BoxFinder
>>> finder = BoxFinder(strategy="hypervolume_tol",
...                    mode="asym",
...                    target_yield=0.95,
...                    n_lhs=4096,
...                    conf_level=0.95,
...                    rng=42)
>>>
>>> cube = finder.find(center_dict,         # X₀
...                    design_space,        # DesignSpace
...                    surrogate,           # BaseSurrogate
...                    specification,       # Specification
...                    delta_init=5e-5)     # стартовое δ
>>> cube["w"]            # (-7.9e-5, 8.3e-5)
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple
import copy

import numpy as np
from numpy.random import Generator
from scipy.stats import norm
from scipy.spatial import cKDTree

from ..design.space import DesignSpace
from ..design.samplers import get_sampler
from ..surrogates import BaseSurrogate
from ..objectives.specification import Specification

# ────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ────────────────────────────────────────────────────────────────────────

def wilson_lower(k: int, n: int, conf_level: float = 0.95) -> float:
    """Возвращает **нижнюю** границу Wilson‑интервала для истинной доли
    успехов при наблюдаемых ``k`` успешных событиях из ``n``.

    Если ``n == 0`` — статистика отсутствует, возвращаем ``np.nan``.
    """
    if n == 0:
        return np.nan

    # Точное z‑значение
    z = norm.ppf(1 - (1 - conf_level) / 2.0)

    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    radius = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (centre - radius) / denom

# ────────────────────────────────────────────────────────────────────────
# ОСНОВНОЙ КЛАСС
# ────────────────────────────────────────────────────────────────────────

class BoxFinder:
    """Поиск максимального tolerance‑box по surrogate‑модели.

    Поддерживаются две стратегии поиска и два режима симметрии.
    Подробности см. в докстринге исходной версии.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        target_yield: float = 0.95,
        n_lhs: int = 4096,
        n_max: int = 65_536,
        conf_level: float = 0.95,
        zoom_ratio: float = 1.20,
        eps_rel: float = 1e-3,
        max_zoom_iter: int = 6,
        strategy: str = "hypervolume_tol",
        mode: str = "sym",
        rng: int | Generator | None = None,
    ):
        if strategy not in ("hypervolume_tol", "axis_tol"):
            raise ValueError("strategy must be 'hypervolume_tol'|'axis_tol'")
        if mode not in ("sym", "asym"):
            raise ValueError("mode must be 'sym'|'asym'")

        self.alpha = float(target_yield)
        self.N0 = int(n_lhs)
        self.n_max = int(n_max)
        self.conf = float(conf_level)
        if zoom_ratio <= 1:
            raise ValueError("zoom_ratio must be > 1")
        self.zoom = float(zoom_ratio)
        self.eps = float(eps_rel)
        self.max_iter = int(max_zoom_iter)
        self.strategy = strategy
        self.mode = mode
        self._rng = np.random.default_rng(rng)

        # Единый Sobol‑сэмплер на весь объект: обеспечивает согласованную
        # псевдослучайность при повторных вызовах
        self._sampler = get_sampler("sobol", scramble=True, rng=self._rng)

    # ---------------------------------------------------------------- PUBLIC
    def find(
        self,
        center: Mapping[str, float],
        space: DesignSpace,
        surrogate: BaseSurrogate,
        spec: Specification,
        *,
        delta_init: float | Mapping[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        """Главная точка входа. См. оригинальный докстринг."""
        if self.strategy == "hypervolume_tol":
            return self._find_hypervolume(center, space, surrogate, spec, delta_init)
        return self._find_axiswise(center, space, surrogate, spec, delta_init)

    # ======================================================================
    #                    ↓↓↓   ВНУТРЕННЯЯ РЕАЛИЗАЦИЯ   ↓↓↓
    # ======================================================================

    # ---------- service ---------------------------------------------------
    @staticmethod
    def _delta_to_array(delta_init, names):
        """Преобразует начальный δ (скаляр или словарь) в ndarray."""
        if isinstance(delta_init, Mapping):
            return np.asarray([float(delta_init[n]) for n in names])
        return np.full(len(names), float(delta_init))

    def _sample_cloud(
        self,
        lows: np.ndarray,
        highs: np.ndarray,
        names: Sequence[str],
        n: int,
        space: DesignSpace,
    ):
        """Сэмплирует ``n`` Sobol‑точек внутри прямоугольника [lows, highs]."""
        # Создаём локальное DesignSpace c сохранением типов переменных
        new_vars = {}
        for name, lo, hi in zip(names, lows, highs):
            v_orig = space[name]
            v_new = copy.deepcopy(v_orig)
            # Для непрерывных/целых переменных поджимаем границы
            if hasattr(v_new, "lower") and hasattr(v_new, "upper"):
                v_new.lower, v_new.upper = lo, hi  # type: ignore[attr-defined]
            new_vars[name] = v_new

        sub_space = DesignSpace(new_vars)
        dict_cloud = self._sampler.sample(sub_space, n)

        # Переводим в ndarray для работы с KD‑Tree
        cloud = np.asarray([[row[n] for n in names] for row in dict_cloud])
        return cloud, dict_cloud

    # ---------- evaluation with Wilson CI ---------------------------------
    def _is_yield_ok(
        self,
        ok_mask: np.ndarray,
        *,
        n_target: int,
    ) -> tuple[bool, float]:
        """Проверяем условие *Wilson_low ≥ α*.

        Если статистики ещё недостаточно (``len(ok_mask) < n_target``),
        возвращаем ``ok=False`` и просим вызывающий код досэмплировать.
        """
        n = len(ok_mask)
        if n < n_target:
            return False, 0.0
        k = int(ok_mask.sum())
        p_low = wilson_lower(k, n, self.conf)
        return bool(p_low >= self.alpha), float(p_low)

    # ======================================================================
    #                 HYPERVOLUME‑TOLERANCE  (LHS‑global‑zoom)
    # ======================================================================
    def _find_hypervolume(
        self,
        center_dict: Mapping[str, float],
        space: DesignSpace,
        surrogate: BaseSurrogate,
        spec: Specification,
        delta_init: float | Mapping[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        names: Sequence[str] = list(space.names())
        x0 = np.asarray([center_dict[n] for n in names])
        delta = self._delta_to_array(delta_init, names)

        best_low = best_high = None
        best_vol = 0.0
        prev_vol = None

        for _ in range(self.max_iter):
            lows, highs = x0 - delta, x0 + delta

            # ── 1) Собираем облако Sobol‑точек ────────────────────────────
            N = self.N0
            while True:
                cloud, cloud_view = self._sample_cloud(lows, highs, names, N, space)
                ok_mask = surrogate.passes_spec(cloud_view, spec)

                # Проверяем Wilson‑критерий **на текущем N**
                ok, _ = self._is_yield_ok(ok_mask, n_target=N)
                if ok or N >= self.n_max:
                    break
                N = min(2 * N, self.n_max)

            if not ok_mask.any():
                # Ни одна точка не прошла — уменьшаем радиус и пробуем снова
                delta *= 0.5
                continue

            # ── 2) KD‑Tree + угловые кандидаты ────────────────────────────
            tree = cKDTree(cloud)
            pass_pts = cloud[ok_mask]

            for p in pass_pts:
                if self.mode == "sym":
                    rad = np.abs(p - x0)
                    cand_low, cand_high = x0 - rad, x0 + rad
                else:  # asym
                    cand_low = np.minimum(p, x0)
                    cand_high = np.maximum(p, x0)

                # Быстрый pre‑filter по ∞‑норме
                box_center = 0.5 * (cand_low + cand_high)
                rad_inf = np.max(cand_high - box_center)
                idx_inside = tree.query_ball_point(box_center, r=rad_inf, p=np.inf)

                # Точная прямоугольная фильтрация
                mask_precise = np.all(
                    (cloud[idx_inside] >= cand_low) & (cloud[idx_inside] <= cand_high),
                    axis=1,
                )
                inside_mask = ok_mask[idx_inside][mask_precise]

                ok_inside, _ = self._is_yield_ok(inside_mask, n_target=max(64, int(0.05 * N)))
                if not ok_inside:
                    continue

                vol = np.prod(cand_high - cand_low)
                if vol > best_vol:
                    best_vol = vol
                    best_low, best_high = cand_low.copy(), cand_high.copy()

            # ── 3) Проверка сходимости по объёму ─────────────────────────
            if prev_vol is not None and best_vol > 0:
                if abs(best_vol - prev_vol) / prev_vol < self.eps:
                    break
            prev_vol = best_vol

            # ── 4) Адаптивный zoom исследуемой зоны ─────────────────────
            if best_low is None:  # ни один кандидат не прошёл
                delta *= 0.5
            else:
                if self.mode == "sym":
                    delta = 0.5 * (best_high - best_low) * self.zoom
                else:  # asym — допускаем сдвиг центра
                    x0 = 0.5 * (best_low + best_high)
                    delta = 0.5 * (best_high - best_low) * self.zoom

        if best_low is None:
            raise RuntimeError("hypervolume_tol: допустимый tolerance‑box не найден.")

        return self._prepare_result(best_low, best_high, center_dict, names)

    # ======================================================================
    #                        AXIS‑TOL  (axiswise zoom)
    # ======================================================================
    def _find_axiswise(
        self,
        center_dict: Mapping[str, float],
        space: DesignSpace,
        surrogate: BaseSurrogate,
        spec: Specification,
        delta_init: float | Mapping[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        names = list(space.names())
        x0 = np.asarray([center_dict[n] for n in names])

        # Симметричный радиус — один массив; асимметрия — два
        if self.mode == "sym":
            d_sym = self._delta_to_array(delta_init, names)
        else:
            d_minus = self._delta_to_array(delta_init, names)
            d_plus = self._delta_to_array(delta_init, names)

        # Внутренняя функция: возвращает Wilson‑низ и признак «условие выполнено»
        def _yield(dm: np.ndarray, dp: np.ndarray) -> tuple[float, bool]:
            N = self.N0
            while True:
                lows, highs = x0 - dm, x0 + dp
                cloud, cloud_view = self._sample_cloud(lows, highs, names, N, space)
                ok_mask = surrogate.passes_spec(cloud_view, spec)

                ok, p_low = self._is_yield_ok(ok_mask, n_target=N)
                if ok or N >= self.n_max:
                    return p_low, ok
                N = min(2 * N, self.n_max)

        # ---------------- симметричный режим ---------------------------
        if self.mode == "sym":
            delta_prev = (2 * d_sym).copy()
            for _ in range(self.max_iter):
                _, ok = _yield(d_sym, d_sym)
                d_sym *= self.zoom if ok else 0.5

                rel_change = np.max(np.abs(2 * d_sym - delta_prev) / (delta_prev + 1e-12))
                if rel_change < self.eps:
                    break
                delta_prev = (2 * d_sym).copy()

            return {n: (-d, d) for n, d in zip(names, d_sym)}

        # ---------------- асимметричный режим ---------------------------
        delta_prev = (d_plus + d_minus).copy()
        for _ in range(self.max_iter):
            # Обрабатываем каждую грань независимо
            for i in range(len(names)):
                # + side
                test_p = d_plus.copy(); test_p[i] *= self.zoom
                _, ok_p = _yield(d_minus, test_p)
                d_plus[i] = test_p[i] if ok_p else d_plus[i] * 0.5

                # − side
                test_m = d_minus.copy(); test_m[i] *= self.zoom
                _, ok_m = _yield(test_m, d_plus)
                d_minus[i] = test_m[i] if ok_m else d_minus[i] * 0.5

            rel_change = np.max(
                np.abs((d_plus + d_minus) - delta_prev) / (delta_prev + 1e-12)
            )
            if rel_change < self.eps:
                break
            delta_prev = (d_plus + d_minus).copy()

        return {n: (-dm, dp) for n, dm, dp in zip(names, d_minus, d_plus)}

    # ======================================================================
    #                      FORMAT RESULT  (sym / asym)
    # ======================================================================
    def _prepare_result(
        self,
        best_low: np.ndarray,
        best_high: np.ndarray,
        center_dict: Mapping[str, float],
        names: Sequence[str],
    ) -> Dict[str, Tuple[float, float]]:
        if self.mode == "sym":
            rad = 0.5 * (best_high - best_low)
            return {n: (-r, r) for n, r in zip(names, rad)}

        x0 = np.asarray([center_dict[n] for n in names])
        d_minus = np.maximum(0.0, x0 - best_low)
        d_plus = np.maximum(0.0, best_high - x0)
        return {n: (-dm, dp) for n, dm, dp in zip(names, d_minus, d_plus)}

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return (
            f"BoxFinder(strategy={self.strategy}, mode={self.mode}, "
            f"α={self.alpha}, conf={self.conf}, eps={self.eps}, "
            f"N0={self.N0}, zoom={self.zoom})"
        )
