#mwlab/opt/tolerance/projection.py
"""
mwlab.opt.tolerance.projection
==============================
Максимальный гипер-блок, гарантированно вложенный в область pass
(“Projection of pass-region”).

Алгоритм
--------
1.  Генерируем *большое* облако Monte-Carlo точек и оставляем только pass.
2.  Стартовый блок [l_raw, u_raw] берем как min/max по каждой координате.
3.  Итеративно проверяем 2^d углов блока:
       если найден FAIL-угол → «усушаем» блок вдоль
       координаты, в которой отклонение от центра наибольшее.
   Повторяем, пока все углы pass **или** достигнут max_iter.
4.  (опц.) Дополнительная MC-валидация yield ≥ target,
    при необходимости симметрично ужимаем блок на 1 %.

Ограничения
-----------
* Требуется **быстрый** surrogate-классификатор (bool), напр. `SpecClassifier`.
* Практично при d ≤ 40 (углов 2^d), оптимально до 20-25 параметров.
"""

from __future__ import annotations
from typing import Dict, Tuple, Sequence

import numpy as np

from mwlab.opt.design.space import DesignSpace
from mwlab.opt.design.samplers import  PointDict, get_sampler
from mwlab.opt.surrogates.base import BaseSurrogate


# ────────────────────────────────────────────────────────────────────────────
class ProjectionBoxOptimizer:
    """
    Parameters
    ----------
    classifier : BaseSurrogate
        Должен быстро возвращать bool (pass/fail).
    design_space : DesignSpace
        Полный набор параметров.
    seed : int
        Сид для reproducibility.
    """

    # ---------------- init --------------------------------------------------
    def __init__(
        self,
        classifier: BaseSurrogate,
        design_space: DesignSpace,
        *,
        seed: int = 0,
    ):
        self.clf = classifier
        self.space = design_space
        self.seed = int(seed)

        self._names = self.space.names()
        self._lows, self._highs = self.space.bounds()
        self._centers = 0.5 * (self._lows + self._highs)

    # ---------------------------------------------------------------- helper
    def _mc_pass_points(
        self,
        n_mc: int,
        delta_limits: Dict[str, float | Tuple[float, float]],
    ) -> np.ndarray:
        """
        Генерирует Monte-Carlo внутри ±delta_limits, возвращает массив pass-точек.
        """
        centers_all = {n: c for n, c in zip(self._names, self._centers)}
        sub_space = DesignSpace.from_center_delta(
            centers_all,
            delta=delta_limits,
            mode="abs",
        )
        pts = sub_space.sample(
            n_mc,
            sampler=get_sampler("sobol", rng=self.seed),
            reject_invalid=False,
        )
        mask = self.clf.batch_predict(pts)   # bool list
        X = np.vstack([sub_space.vector(p) for p in pts])
        return X[np.asarray(mask)]

    @staticmethod
    def _angles(lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
        """
        Возвращает (2^d, d) массив углов гипер-блока.
        """
        d = len(lows)
        # индексы 0..2^d-1 → биты определяют выбор low/high
        grid = np.indices((2,) * d).reshape(d, -1).T
        return lows + grid * (highs - lows)

    # ---------------------------------------------------------------- public
    def find_box(  # ← метод класса ProjectionBoxOptimizer
            self,
            *,
            delta_limits: Dict[str, float | Tuple[float, float]],
            n_mc_init: int = 262_144,
            epsilon: float = 0.02,
            max_iter: int = 60,
            target_yield_check: float | None = 0.99,
            n_mc_check: int = 262_144,
            verbose: bool = True,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Подбирает *наибольший* гипер-блок, полностью лежащий в pass-области.

        Возвращает словарь ``{name: (δ_minus, δ_plus)}``
        — абсолютные радиусы слева/справа от центра.

        Параметры
        ----------
        delta_limits
            Ограничения ±δ, внутри которых строим начальную MC-выборку.
        n_mc_init
            Размер стартового облака Monte-Carlo.
        epsilon
            Относительное «усыхание» блока при обнаружении FAIL-угла.
        max_iter
            Максимум итераций shrink-цикла.
        target_yield_check
            Если задано (0..1), выполняется финальная валидация yield ≥ value.
        n_mc_check
            Размер MC-выборки на валидацию.
        verbose
            True → печать служебных сообщений и показ tqdm-progressbar.
        """
        # ----- 1. Стартовый набор pass-точек -------------------------------
        X_pass = self._mc_pass_points(n_mc_init, delta_limits)
        if verbose:
            print(f"[Projection] стартовых pass-точек : {len(X_pass):,}")

        # min / max по каждой координате → грубый «сырой» блок
        lows = X_pass.min(axis=0)
        highs = X_pass.max(axis=0)

        # ----- 2. Итеративное «усыхание» блока -----------------------------
        try:
            from tqdm import tqdm  # красивый progressbar
            _steps = tqdm(range(max_iter), desc="Projection-shrink", unit="step")
        except ModuleNotFoundError:
            _steps = range(max_iter)  # fallback: обычный range

        for step in _steps:
            # 2a. Все углы текущего блока (2^d штук)
            corners = self._angles(lows, highs)  # (2^d, d)

            # 2b. Проверяем углы по одному → ранний выход при первом FAIL
            for vec in corners:
                if not self.clf.predict(self.space.dict(vec)):
                    fail_vec = vec  # нашли FAIL-угол
                    break
            else:  # цикл не прервался ⇒ все pass
                if verbose:
                    print(f"[Projection] все углы pass на шаге {step}")
                break  # hyper-блок валиден — стоп

            # 2c. Выбираем координату, по которой |dx| максимально
            diff = np.abs(fail_vec - self._centers)
            j = diff.argmax()  # индекс «самого далекого» измерения

            # 2d. Сдвигаем соответствующую границу внутрь на epsilon
            if fail_vec[j] < self._centers[j]:  # FAIL слева от центра
                lows[j] += epsilon * (self._centers[j] - lows[j])
            else:  # FAIL справа
                highs[j] -= epsilon * (highs[j] - self._centers[j])

            # 2e. Обновляем строку прогресса краткой статистикой
            if verbose and hasattr(_steps, "set_postfix_str"):
                volume = np.prod(highs - lows)
                _steps.set_postfix_str(
                    f"shrink axis={self._names[j]}, volume={volume:.2e}"
                )

        # ----- 3. Доп. MC-валидация блока на требуемый yield ---------------
        if target_yield_check is not None:
            def _blk_delta() -> Dict[str, Tuple[float, float]]:
                """Текущие радиусы блока → словарь δ⁻/δ⁺."""
                return {
                    n: (-(c - lo), hi - c)
                    for n, c, lo, hi in zip(self._names, self._centers, lows, highs)
                }

            X_val = self._mc_pass_points(n_mc_check, _blk_delta())
            yield_blk = len(X_val) / n_mc_check
            if verbose:
                print(f"[Projection] первичная проверка: yield={yield_blk:.3f}")

            # пока yield < target — симметрично ужимаем блок на 1 %
            while yield_blk < target_yield_check:
                lows += 0.01 * (self._centers - lows)
                highs -= 0.01 * (highs - self._centers)

                X_val = self._mc_pass_points(n_mc_check, _blk_delta())
                yield_blk = len(X_val) / n_mc_check
                if verbose:
                    print(f"  shrink 1 %  →  yield={yield_blk:.3f}")

        # ----- 4. Формируем словарь (−δ⁻, +δ⁺) -----------------------------
        delta_out: Dict[str, Tuple[float, float]] = {}
        for n, c, lo, hi in zip(self._names, self._centers, lows, highs):
            delta_out[n] = (lo - c, hi - c)  # левая δ < 0, правая δ > 0

        return delta_out

