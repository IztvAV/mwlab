#mwlab/opt/sensitivity/sobol.py
"""
sobol.py
========
Обёртка над алгоритмом Соболя (Saltelli sampling) из **SALib**.

* Поддерживает first-order (`S1`) и total-order (`ST`) индексы.
* При `second_order=True` добавляет DataFrame-колонку `S2` (MultiIndex).

Ограничение: из-за квадратичного роста числа комбинаций второй порядок
имеет смысл после предварительного скрининга (оставить 10–20 параметров).
"""

from __future__ import annotations
from typing import Mapping, Sequence, List

import numpy as np
import pandas as pd
import inspect
import warnings

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.space import DesignSpace
from mwlab.opt.objectives.specification import Specification
from .utils import ensure_salib, batch_eval

# ────────────────────────────────────────────────────────────────────────────
def _space_to_salib_problem(space: DesignSpace, subset: Sequence[str] | None = None):
    """Тот же helper, но с опцией подмножества параметров."""
    names = list(map(str, subset)) if subset is not None else space.names()
    lows, highs = space.bounds()
    idx = [space.names().index(n) for n in names]
    bounds = np.stack([lows[idx], highs[idx]], axis=1).tolist()
    return {"num_vars": len(names), "names": names, "bounds": bounds}


# ────────────────────────────────────────────────────────────────────────────
def run_sobol(
    surrogate: BaseSurrogate,
    space: DesignSpace,
    spec: Specification,
    *,
    params: Sequence[str] | str = "auto",
    n_base: int = 1024,
    second_order: bool = False,
    sampler_kw: Mapping | None = None,
    rng: int = 0,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    params : list[str] | "auto"
        Какие параметры анализировать.  "auto" → все.
    n_base : int
        Базовый размер Saltelli (реальных симуляций ≈ (2d+2)*n_base).
    second_order : bool
        Считать ли индексы 2-го порядка `S2_ij`.

    Returns
    -------
    df : DataFrame с колонками 'S1', 'ST' (и 'S2' при second_order).
    """
    ensure_salib()
    # 0) Импортируем анализатор (он общий)
    from SALib.analyze import sobol as sobol_analyze

    # Пытаемся взять «новый» генератор (SALib ≥ 1.5)
    try:
        from SALib.sample import sobol as sobol_sampler
    except ImportError:                       # старые версии
        from SALib.sample import saltelli as sobol_sampler  # type: ignore

    # 1) Определяем подмножество параметров
    if isinstance(params, str):
        if params != "auto":
            raise ValueError("params может быть 'auto' или списком имён.")
        subset = None
    else:
        subset = params

    problem = _space_to_salib_problem(space, subset)

    # 2) Saltelli-выборка
    sample_sig = inspect.signature(sobol_sampler.sample)
    sample_kwargs = dict(
        problem       = problem,
        N             = n_base,
        calc_second_order = second_order,
    )
    if "seed" in sample_sig.parameters:
        sample_kwargs["seed"] = rng

    # Сам вызов оборачиваем контекстом подавления лишних предупреждений
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"SALib\.sample",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"SALib\.util",
        )
        X = sobol_sampler.sample(**sample_kwargs)

    # 3) Формируем ПОЛНЫЙ вектор параметров:               <-- фиксация
    lows, highs = space.bounds()
    centers = 0.5 * (lows + highs)           # можно тонко: x0 от пользователя

    full_pts = []
    for row in X:
        full = centers.copy()                # стартуем с центров
        for j, name in enumerate(problem["names"]):
            idx = space.names().index(name)  # позиция параметра в полном списке
            full[idx] = row[j]               # заменяем только varied-координату
        full_pts.append(space.dict(full))

    pts = full_pts
    Y_bool = batch_eval(surrogate, pts, spec).astype(float)

    # 4) Анализ
    res = sobol_analyze.analyze(
        problem,
        Y_bool,
        calc_second_order=second_order,
        print_to_console=False,
    )

    # 5) DataFrame
    df = pd.DataFrame({
        "S1": res["S1"],
        "ST": res["ST"],
    }, index=problem["names"])

    if second_order:
        # превращаем квадратную матрицу → Series MultiIndex (i,j)
        S2_mat = res["S2"]
        names = problem["names"]
        idx_i, idx_j, vals = [], [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                idx_i.append(names[i])
                idx_j.append(names[j])
                vals.append(S2_mat[i, j])
        s2 = pd.Series(vals, index=pd.MultiIndex.from_arrays([idx_i, idx_j]))
        df["S2"] = np.nan                 # заглушка на уровне df; сам S2 отдаём отдельно
        df.attrs["S2_pairs"] = s2         # прячем Series в .attrs
    return df

