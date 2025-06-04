#mwlab/opt/tolerance/joint_box.py
"""
joint_box.py
============
Совместная оптимизация гипер-блока ±δ_i с условием `Yield ≥ target`.

Используем SciPy SLSQP по log-объёму (эквивалентно максимизации продукта δ_i).
Если установлен пакет `cma`, можно выбрать `method="cmaes"` – глобальное,
но здесь даём только последовательную реализацию.
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np

from scipy.optimize import minimize, OptimizeResult

from mwlab.opt.design import DesignSpace, get_sampler
from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.objectives.specification import Specification
from mwlab.opt.objectives import YieldObjective


# ────────────────────────────────────────────────────────────────────────────
class JointBoxOptimizer:
    """
    Parameters
    ----------
    surrogate, design_space, specification  – как всегда.
    """

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

        self._names = self.space.names()
        lows, highs = self.space.bounds()
        self._centers = 0.5 * (lows + highs)

    # ------------------------------------------------------------------------
    def optimize(
        self,
        delta_upper: Dict[str, float | Tuple[float, float]],
        *,
        target_yield: float = 0.99,
        n_mc: int = 8192,
        method: str = "slsqp",
        max_iter: int = 200,
        verbose: bool = False,
    ) -> Tuple[Dict[str, float | Tuple[float, float]], OptimizeResult]:
        """
        Возвращает (δ_dict, scipy_res).

        δ_dict содержит **симметричные** или асимметричные радиусы,
        точно в той же форме, что и `delta_upper`.
        """
        names = list(delta_upper)
        k = len(names)

        # ---------------------- трансформация переменных -----------------
        # работаем в лог-пространстве для стабильности градиента
        def to_vec(d: Dict[str, float | Tuple[float, float]]):
            vec = []
            for n in names:
                v = d[n]
                if isinstance(v, tuple):
                    vec.extend([np.log(max(v[0], 1e-12)), np.log(max(v[1], 1e-12))])
                else:
                    vec.append(np.log(max(v, 1e-12)))
            return np.array(vec)

        # начальная точка = δ_upper / 2
        x0_vec = to_vec({
            n: (v[0]*0.5, v[1]*0.5) if isinstance(v, tuple) else v*0.5
            for n, v in delta_upper.items()
        })

        bounds = []
        for n in names:
            v = delta_upper[n]
            if isinstance(v, tuple):
                bounds.extend([(np.log(1e-12), np.log(v[0])),   # минус
                               (np.log(1e-12), np.log(v[1]))])  # плюс
            else:
                bounds.append((np.log(1e-12), np.log(v)))

        # ---------------------- вспомогательные функции ------------------
        def _dict_from_vec(vec):
            out = {}
            it = iter(vec)
            for n, v in delta_upper.items():
                if isinstance(v, tuple):
                    d_minus = np.exp(next(it))
                    d_plus  = np.exp(next(it))
                    out[n] = (-d_minus, d_plus)
                else:
                    out[n] = np.exp(next(it))
            return out

        def _subspace(vec):
            δ = _dict_from_vec(vec)
            # переводим в формат DesignSpace.from_center_delta
            return self.space.freeze_axes(
                [q for q in self._names if q not in names]
            ).from_center_delta(
                centers={n: self._centers[self._names.index(n)] for n in names},
                delta=δ,
                mode="abs",
            )

        def objective(vec):
            # minimise −Σ log δ  →  maximise объём
            return -np.sum(vec)

        def yield_constraint(vec):
            ds = _subspace(vec)
            y = YieldObjective(
                surrogate=self.sur,
                spec=self.spec,
                design_space=ds,
                sampler=get_sampler("sobol", rng=self.seed),
                n_mc=n_mc
            )()
            return y - target_yield

        # ---------------------- запуск оптимизации -----------------------
        if method.lower() == "slsqp":
            res = minimize(
                objective,
                x0_vec,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": yield_constraint},
                options={"maxiter": max_iter, "ftol": 1e-3, "disp": verbose},
            )
            δ_opt = _dict_from_vec(res.x)
            return δ_opt, res

        elif method.lower() == "cmaes":
            try:
                import cma
            except ModuleNotFoundError as err:
                raise ImportError("method='cmaes' требует пакета 'cma'") from err

            es = cma.CMAEvolutionStrategy(
                x0_vec,
                0.3,
                {"bounds": list(zip(*bounds)), "verbose": verbose},
            )
            while not es.stop():
                xs = es.ask()
                cs = [yield_constraint(x) for x in xs]
                fs = [objective(x) + 1e6 * max(0, -c) for x, c in zip(xs, cs)]
                es.tell(xs, fs)
            best = es.result.xbest
            res = OptimizeResult(x=best, fun=objective(best), nit=es.result.iterations)
            return _dict_from_vec(best), res

        else:
            raise ValueError("method должен быть 'slsqp' или 'cmaes'")
