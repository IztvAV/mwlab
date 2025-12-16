#mwlab/opt/objectives/yield_max.py
"""
mwlab.opt.objectives.yield_max
==============================
**YieldObjective**
Оценивает вероятность выполнения ТЗ (*yield*, выход годных изделий)
по **суррогатной модели** методом Монте-Карло.

Идея
----
1. Вокруг «центра» (обычно — оптимальное решение x₀) генерируем `n_mc`
   случайных точек проектных параметров в пределах `DesignSpace`.
2. Пропускаем их сквозь surrogate → получаем набор S-параметров.
3. Для каждого экземпляра проверяем `Specification.is_ok(net)`.
4. Yield = mean(pass).

Подойдет для:
* анализа технологических допусков (tolerance);
* быстрой прикидки yield перед тяжелым HFSS-пересчетом;
* цели оптимизации (*maximize yield*) — нужно всего лишь `-yield`
  преобразовать в «minimization cost».

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
from typing import Mapping

import numpy as np

from ..design.space import DesignSpace
from ..design.samplers import BaseSampler, get_sampler
from ..surrogates import BaseSurrogate
from .specification import Specification


# ────────────────────────────────────────────────────────────────────────────
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
        Способ выборки DOE-точек.  Можно передать строковый alias.
    n_mc : int, default=8192
        Размер выборки Монте-Карло.
    sampler_kwargs : dict, optional
        Аргументы, пробрасываемые в фабрику `get_sampler`.
    """

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        *,
        surrogate: BaseSurrogate,
        spec: Specification,
        design_space: DesignSpace,
        sampler: BaseSampler | str = "normal",
        n_mc: int = 8192,
        sampler_kwargs: Mapping | None = None,
    ):
        self.sur: BaseSurrogate = surrogate
        self.spec: Specification = spec
        self.space: DesignSpace = design_space
        self.n_mc: int = int(n_mc)
        if self.n_mc <= 0:
            raise ValueError("n_mc must be a positive integer")

        if isinstance(sampler, str):
            self.sampler: BaseSampler = get_sampler(sampler, **(sampler_kwargs or {}))
        else:
            self.sampler = sampler

    # ---------------------------------------------------------------- call
    def __call__(self) -> float:  # noqa: D401
        """
        Запускает Монте-Карло и возвращает yield ∈ [0,1].

        Note
        ----
        *При необходимости* можно переопределить этот метод так, чтобы он
        возвращал и доверительный интервал (например,
        `yield, (low, high)` по методу Вильсона).  Пока MVP — только mean.
        """
        # 1) Сэмплируем точки в пределах tolerance-объёма
        pts = self.space.sample(
            self.n_mc,
            sampler=self.sampler,
            reject_invalid=False,   # дизайн-спейс уже валиден
        )

        # 2) Проверяем спецификацию через суррогат.
        #    Всегда используем BaseSurrogate.passes_spec:
        #    - для NNSurrogate сработает быстрый векторный путь (GPU-friendly);
        #    - для остальных будет корректный fallback через batch_predict.
        ok_mask = self.sur.passes_spec(pts, self.spec)  # np.ndarray[bool]
        return float(np.mean(ok_mask))

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return (f"YieldObjective(n_mc={self.n_mc}, sampler={self.sampler}, "
                f"spec={self.spec.name})")

