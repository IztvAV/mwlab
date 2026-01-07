#mwlab/opt/objectives/yield_max.py
"""
mwlab.opt.objectives.yield_max
==============================
YieldObjective
--------------
Оценивает вероятность выполнения ТЗ (*yield*, выход годных изделий)
по **суррогатной модели** методом Монте-Карло.

Алгоритм
--------
1) В пределах заданного DesignSpace генерируем n_mc точек.
2) Прогоняем их через surrogate.
3) Для каждой точки проверяем Specification.is_ok(net).
4) Yield = mean(pass).

Оптимизация производительности
------------------------------
Если surrogate реализует метод passes_spec(points, spec),
он используется как быстрый путь (векторизация / GPU-friendly).
Если нет — выполняется fallback: batch_predict + spec.is_ok поштучно.

Пример
------
>>> # 1) Design-space ±100 µm
>>> space = DesignSpace({"w": ContinuousVar(center=0, delta=100e-6)})
>>>
>>> # 2) Dummy surrogate (уже обученный BaseLModule)
>>> sur = NNSurrogate(pl_module=trained_lm)
>>>
>>> # 3) Specification: |S11| ≤ –22 dB в полосе
>>> crit = BaseCriterion(
...     selector   = SMagSelector(1,1, band=(2.0,2.4), db=True),
...     aggregator = MaxAgg(),
...     comparator = LEComparator(-22),
...     name="S11"
... )
>>> spec = Specification([crit])
>>>
>>> # 4) YieldObjective
>>> sampler = get_sampler("normal", k=3, rng=2025)
>>> yobj = YieldObjective(
...     surrogate   = sur,
...     spec        = spec,
...     design_space= space,
...     sampler     = sampler,
...     n_mc        = 4096,
... )
>>> yield_val = yobj()          # 0.87, например
"""

from __future__ import annotations

from typing import Mapping, Any, Optional

import numpy as np

from mwlab.opt.design.space import DesignSpace
from mwlab.opt.design.samplers import BaseSampler, get_sampler
from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.objectives.specification import Specification


class YieldObjective:
    """
    Parameters
    ----------
    surrogate : BaseSurrogate
        Эмулятор прямой задачи X→S (NNSurrogate, GPSurrogate …).
    spec : Specification
        Техзадание, содержащее набор Criterion-ов.
    design_space : DesignSpace
        Границы параметров, внутри которых выполняем Монте-Карло.
    sampler : BaseSampler | str, default="normal"
        Способ выборки DOE-точек. Можно передать строковый alias.
    n_mc : int, default=8192
        Размер выборки Монте-Карло.
    sampler_kwargs : dict, optional
        Аргументы, пробрасываемые в фабрику `get_sampler`, если sampler задан строкой.
    """

    def __init__(
        self,
        *,
        surrogate: BaseSurrogate,
        spec: Specification,
        design_space: DesignSpace,
        sampler: BaseSampler | str = "normal",
        n_mc: int = 8192,
        sampler_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.sur: BaseSurrogate = surrogate
        self.spec: Specification = spec
        self.space: DesignSpace = design_space

        self.n_mc: int = int(n_mc)
        if self.n_mc <= 0:
            raise ValueError("n_mc must be a positive integer")

        if isinstance(sampler, str):
            # Если нужен детерминизм, передавайте rng через sampler_kwargs, напр. {"rng": 2025}
            self.sampler: BaseSampler = get_sampler(sampler, **(dict(sampler_kwargs or {})))
        else:
            self.sampler = sampler

    def __call__(self) -> float:
        """
        Запускает Монте-Карло и возвращает yield ∈ [0,1].
        """
        # 1) Сэмплируем точки в пределах DesignSpace
        pts = self.space.sample(
            self.n_mc,
            sampler=self.sampler,
            reject_invalid=False,  # дизайн-спейс уже валиден
        )

        # 2) Быстрый путь: surrogate.passes_spec, если есть
        fn = getattr(self.sur, "passes_spec", None)
        if callable(fn):
            ok_mask = fn(pts, self.spec)
            ok = np.asarray(ok_mask, dtype=bool)
            return float(np.mean(ok))

        # 3) Fallback: batch_predict + spec.is_ok
        nets = self.sur.batch_predict(pts)
        ok = np.fromiter((self.spec.is_ok(net) for net in nets), dtype=bool, count=len(nets))
        return float(np.mean(ok))

    def __repr__(self) -> str:  # pragma: no cover
        return f"YieldObjective(n_mc={self.n_mc}, sampler={self.sampler}, spec={self.spec.name})"


__all__ = ["YieldObjective"]
