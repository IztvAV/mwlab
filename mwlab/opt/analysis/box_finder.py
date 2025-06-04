"""
MWLab · opt · analysis · box_finder
===================================
Поиск **tolerance-box** (допускового гипер-куба) двумя способами.

* **hypervolume_tol** – LHS/Sobol-global + zoom (Zhang et al., IEEE TAP 2018).
  Жёсткий критерий: *все* точки внутри кандидата должны удовлетворять ТЗ.
* **axis_tol**        – поосевой zoom-search (каждая переменная отдельно).

Поддерживаются оба режима симметрии:
    • mode="sym"  → δ⁺ = δ⁻ (радиусы равны);
    • mode="asym" → δ⁺, δ⁻ независимы, центр может смещаться.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.random import Generator

from ..design.samplers import get_sampler
from ..design.space import DesignSpace
from ..objectives.specification import Specification
from ..surrogates import BaseSurrogate

# ─────────────────────────────────────────────────────────────────────────────
#                                  BoxFinder
# ─────────────────────────────────────────────────────────────────────────────
class BoxFinder:
    """
    Parameters
    ----------
    target_yield : float, default 1.0
        *Не используется* в упрощённом варианте; алгоритм всегда
        проверяет «все точки PASS». Сохраняем параметр для
        совместимости вызовов.
    n_lhs : int, default 4096
        Число Sobol-точек в облаке на каждой zoom-итерации.
    zoom_ratio : float > 1, default 1.2
        Во сколько раз расширяем/сужаем радиус при удачной/неудачной
        попытке.
    eps_rel : float, default 1e-3
        Критерий сходимости по относительному приросту объёма.
    max_zoom_iter : int, default 6
        Максимум zoom-циклов.
    strategy : {"hypervolume_tol","axis_tol"}
    mode : {"sym","asym"}
    rng : int | np.random.Generator | None
        Источник псевдослучайности для Sobol-сэмплера.
    use_axis_seed : bool, default True
        Использовать быстрый поосевой поиск как стартовую оценку δ.
    active_axes : Sequence[str] | None
        Список «значимых» переменных; остальные замораживаются (δ=0).
    """

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        *,
        target_yield: float = 1.0,
        n_lhs: int = 4096,
        zoom_ratio: float = 1.20,
        eps_rel: float = 1e-3,
        max_zoom_iter: int = 6,
        strategy: str = "hypervolume_tol",
        mode: str = "sym",
        rng: int | Generator | None = None,
        use_axis_seed: bool = True,
        active_axes: Sequence[str] | None = None,
    ) -> None:
        if strategy not in ("hypervolume_tol", "axis_tol"):
            raise ValueError("strategy must be 'hypervolume_tol'|'axis_tol'")
        if mode not in ("sym", "asym"):
            raise ValueError("mode must be 'sym'|'asym'")
        if zoom_ratio <= 1:
            raise ValueError("zoom_ratio must be > 1")

        self.strategy = strategy
        self.mode = mode
        self.N = int(n_lhs)
        self.zoom = float(zoom_ratio)
        self.eps = float(eps_rel)
        self.max_iter = int(max_zoom_iter)
        self.use_axis_seed = bool(use_axis_seed)
        self.active_axes: set[str] | None = set(active_axes) if active_axes else None

        self._rng = np.random.default_rng(rng)
        self._sampler = get_sampler("lhs", rng=self._rng) #get_sampler("sobol", scramble=True, rng=self._rng)

    # ---------------------------------------------------------------- public
    def find(
        self,
        center: Mapping[str, float],
        space: DesignSpace,
        surrogate: BaseSurrogate,
        spec: Specification,
        *,
        delta_init: float | Mapping[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        if self.strategy == "axis_tol":
            return self._find_axiswise(center, space, surrogate, spec, delta_init)
        return self._find_hypervolume(center, space, surrogate, spec, delta_init)

    # ─────────────────────────────────────────────────────────────────────
    #                           helpers
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _delta_to_array(delta_init, names) -> np.ndarray:
        if isinstance(delta_init, np.ndarray):
            if delta_init.size != len(names):
                raise ValueError("delta ndarray dimension mismatch")
            return delta_init.astype(float, copy=True)
        if isinstance(delta_init, Mapping):
            return np.asarray([float(delta_init[n]) for n in names])
        return np.full(len(names), float(delta_init))

    def _point_passes(
        self,
        point: Mapping[str, float],
        surrogate: BaseSurrogate,
        spec: Specification,
    ) -> bool:
        """Проверка одной точки через универсальный passes_spec."""
        return bool(surrogate.passes_spec(point, spec))

    # ---- Sobol-выборка в произвольном прямоугольнике -----------------
    def _sample_cloud(
        self,
        lows: np.ndarray,
        highs: np.ndarray,
        names: Sequence[str],
        n: int,
        space: DesignSpace,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Генерируем n Sobol-точек в гиперкубе [lows, highs]"""
        # создаём «обрезанный» DesignSpace с сохранением типов переменных
        sub_vars = {}
        for name, lo, hi in zip(names, lows, highs):
            v = copy.deepcopy(space[name])
            if hasattr(v, "lower") and hasattr(v, "upper"):
                v.lower, v.upper = lo, hi  # type: ignore[attr-defined]
            sub_vars[name] = v
        sub_space = DesignSpace(sub_vars)
        dict_cloud = self._sampler.sample(sub_space, n)
        cloud = np.asarray([[row[n] for n in names] for row in dict_cloud])
        return cloud, dict_cloud

    # ─────────────────────────────────────────────────────────────────────
    #                    1.  HYPERVOLUME-TOL
    # ─────────────────────────────────────────────────────────────────────
    #  Алгоритм одновременно ведёт два «потока» кандидатов:
    #
    #  • Трек-A (асим.) – прямоугольник [min(p,x0), max(p,x0)].
    #    ─ Используется только для оценки «хорошего» положения центра.
    #
    #  • Трек-B (симм.) – прямоугольник [x0 – |p–x0|, x0 + |p–x0|].
    #    ─ Из него выбираем бокс максимального объёма и,
    #      сохраняя δ⁻/δ⁺ независимыми, обновляем радиусы.
    #
    #  Центр плавно перетягиваем к barycenter всех «успешных» A-рамок
    #  (damping-фактор alpha ∈ (0,1) защищает от резких скачков).
    # ─────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────
    #      ОБНОВЛЁННЫЙ  _find_hypervolume
    # ─────────────────────────────────────────────────────────────────────
    #  • Работает в **едином** цикле `for it in range(self.max_iter)`.
    #  • В обоих режимах сначала ищет     СИММЕТРИЧНЫЙ tolerance-box
    #    вокруг текущего центра `x0`  →   best_low / best_high / best_vol.
    #  • Для   mode="asym"  *дополнительно* оценивает «куда сместить центр»
    #    по **асимметричным** рамкам [min(p,x0), max(p,x0)] и плавно
    #    сдвигает `x0` на следующую итерацию.
    #  • Радиус δ рассчитывается одинаково для обоих режимов:
    #          δ = 0.5 · (best_high − best_low) · zoom
    #  • После завершения цикла:
    #        –  в  "sym"  возвращаем ±δ,
    #        –  в  "asym" переводим их в  (δ⁻, δ⁺)  относительно исходного
    #           центра  x0_orig  (API сохранён).
    # ─────────────────────────────────────────────────────────────────────
    def _find_hypervolume(
        self,
        center_dict: Mapping[str, float],
        space: DesignSpace,
        surrogate: BaseSurrogate,
        spec: Specification,
        delta_init: float | Mapping[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Поиск максимального tolerance-box.

        В режиме «sym» центр фиксирован, δ⁺ = δ⁻.
        В режиме «asym» центр может «плавать», но на выходе возвращаем
        асимметричные (δ⁻, δ⁺) относительно исходного X₀.
        """
        # --- 0. базовая подготовка -------------------------------------------
        names = list(space.names())
        x0_orig = np.asarray([center_dict[n] for n in names], dtype=float)
        x0 = x0_orig.copy()                       # рабочий (плавающий) центр

        if not self._point_passes(center_dict, surrogate, spec):
            raise RuntimeError("Номинал не проходит Specification – поиск невозможен.")

        # стартовый радиус δ
        delta = self._delta_to_array(delta_init, names)

        # быстрый осевой seed (если включён)
        if self.use_axis_seed:
            axis_box = self._find_axiswise(center_dict, space, surrogate, spec, delta)
            axis_vec = np.asarray([abs(axis_box[n][0]) for n in names])
            delta = np.minimum(delta, axis_vec)

        # обнуляем «замороженные» оси
        if self.active_axes:
            for i, n in enumerate(names):
                if n not in self.active_axes:
                    delta[i] = 0.0

        # глобальный лучший бокс среди всех итераций
        best_low = best_high = None
        best_vol = 0.0
        prev_vol = None

        # настройки сдвига центра (для asym)
        alpha_shift = 0.5                 # damping-коэффициент
        eps_center = 1e-6                 # порог «движение центра сошлось»

        # --- 1. основной zoom-цикл -------------------------------------------
        for i in range(self.max_iter):
            print(f'iter = {i+1}/{self.max_iter} shift=\n{x0_orig - x0}')

            # 1.1. Sobol-облако вокруг текущего x0
            lows, highs = x0 - delta, x0 + delta
            cloud, dict_cloud = self._sample_cloud(lows, highs, names, self.N, space)

            ok_mask = np.asarray(surrogate.passes_spec(dict_cloud, spec), dtype=bool)

            # если нет ни одной PASS-точки → shrink и skip
            if not ok_mask.any():
                delta *= 0.5
                continue

            pass_pts = cloud[ok_mask]

            # --- накопители текущей итерации ---------------------------------
            # best_* для симметричных кандидатов
            iter_best_low = iter_best_high = None
            iter_best_vol = 0.0

            # углы асим. рамок (только когда mode=="asym")
            if self.mode == "asym":
                sum_corner = np.zeros_like(x0)
                cnt_corner = 0

            # 1.2. перебираем все PASS-точки p
            for p in pass_pts:
                # --- A) СИММЕТРИЧНЫЙ кандидат вокруг x0 ----------------------
                rad = np.abs(p - x0)
                cand_low = x0 - rad
                cand_high = x0 + rad

                inside = np.all((cloud >= cand_low) & (cloud <= cand_high), axis=1)
                if inside.any() and ok_mask[inside].all():    # all-PASS внутри рамки?
                    vol = np.prod(cand_high - cand_low)*1e20
                    if vol > iter_best_vol:
                        iter_best_vol = vol
                        iter_best_low = cand_low.copy()
                        iter_best_high = cand_high.copy()

                # --- B) АСИМ. рамка (только для asym-режима) -----------------
                if self.mode == "asym":
                    low_a = np.minimum(p, x0)
                    high_a = np.maximum(p, x0)
                    inside_a = np.all((cloud >= low_a) & (cloud <= high_a), axis=1)
                    if inside_a.any() and ok_mask[inside_a].all():
                        # обе вершины рамки валидны → учитываем их
                        sum_corner += low_a + high_a
                        cnt_corner += 2

            # --- 1.3. если в этой итерации нашли хотя бы один сим-бокс -------
            if iter_best_low is not None:
                print(f'iter_best_vol = {iter_best_vol}')
                # обновляем глобальный лучший
                if iter_best_vol > best_vol:
                    best_vol = iter_best_vol
                    best_low = iter_best_low
                    best_high = iter_best_high
                    print('New best volume')

                # новый радиус δ (одинаков для обоих режимов)
                delta = 0.5 * (iter_best_high - iter_best_low) * self.zoom

                # --- 1.4. сдвиг центра (только asym) ------------------------
                if self.mode == "asym" and cnt_corner:
                    center_cand = sum_corner / cnt_corner

                    # защитная проверка: новый центр сам должен PASS
                    cand_dict = {n: float(v) for n, v in zip(names, center_cand)}
                    if surrogate.passes_spec(cand_dict, spec):
                        shift = center_cand - x0

                        # не двигаем «замороженные» оси
                        if self.active_axes:
                            for i, n in enumerate(names):
                                if n not in self.active_axes:
                                    shift[i] = 0.0

                        # damping-смещение
                        x0 += alpha_shift * shift

                        # критерий «центр сошёлся»
                        if np.linalg.norm(shift) < eps_center:
                            # если и объём тоже стабилизировался — можно выйти
                            if prev_vol and abs(best_vol - prev_vol) / prev_vol < self.eps:
                                break
            else:
                # сим-бокса нет → ещё сильнее сжимаем δ
                delta *= 0.5
                # переход к следующей итерации
                continue

            # --- 1.5. проверка сходимости по объёму --------------------------
            if prev_vol is not None and prev_vol > 0:
                if abs(best_vol - prev_vol) / prev_vol < self.eps:
                    break
            prev_vol = best_vol

        # --- 2. формируем результат ------------------------------------------
        if best_low is None:
            raise RuntimeError("hypervolume_tol: допустимый tolerance-box не найден.")

        if self.mode == "sym":
            r = 0.5 * (best_high - best_low)
            return {n: (-ri, ri) for n, ri in zip(names, r)}

        # asym: переводим сим-радиусы в (δ⁻, δ⁺) относительно исходного X₀
        d_minus = np.maximum(0.0, x0_orig - best_low)
        d_plus = np.maximum(0.0, best_high - x0_orig)
        return {n: (-dm, dp) for n, dm, dp in zip(names, d_minus, d_plus)}



    # ─────────────────────────────────────────────────────────────────────
    #                    2.  AXIS-TOL  (поосевой zoom)
    # ─────────────────────────────────────────────────────────────────────
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

        if not self._point_passes(center_dict, surrogate, spec):
            raise RuntimeError("Номинал не проходит Specification – axis_tol невозможен.")

        d_init = self._delta_to_array(delta_init, names)
        d_minus = np.zeros_like(d_init)
        d_plus = np.zeros_like(d_init)

        for i, d0 in enumerate(d_init):
            # симметричный поиск
            if self.mode == "sym":
                delta = d0
                prev = delta
                for _ in range(self.max_iter):
                    test = center_dict.copy()
                    test[names[i]] = float(x0[i] + delta)
                    ok = self._point_passes(test, surrogate, spec)
                    delta = delta * self.zoom if ok else delta * 0.5
                    if abs(delta - prev) / (prev + 1e-12) < self.eps:
                        break
                    prev = delta
                d_minus[i] = d_plus[i] = delta
            # асимметрия
            else:
                # + сторона
                delta = d0
                prev = delta
                for _ in range(self.max_iter):
                    test = center_dict.copy()
                    test[names[i]] = float(x0[i] + delta)
                    ok = self._point_passes(test, surrogate, spec)
                    delta = delta * self.zoom if ok else delta * 0.5
                    if abs(delta - prev) / (prev + 1e-12) < self.eps:
                        break
                    prev = delta
                d_plus[i] = delta

                # – сторона
                delta = d0
                prev = delta
                for _ in range(self.max_iter):
                    test = center_dict.copy()
                    test[names[i]] = float(x0[i] - delta)
                    ok = self._point_passes(test, surrogate, spec)
                    delta = delta * self.zoom if ok else delta * 0.5
                    if abs(delta - prev) / (prev + 1e-12) < self.eps:
                        break
                    prev = delta
                d_minus[i] = delta

        return {n: (-dm, dp) for n, dm, dp in zip(names, d_minus, d_plus)}

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BoxFinder(strategy={self.strategy}, mode={self.mode}, "
            f"N={self.N}, zoom={self.zoom}, eps={self.eps})"
        )

