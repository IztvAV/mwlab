#mwlab/opt/sensitivity/morris.py
"""
morris.py
=========
Обёртка над реализацией метода Морриса из библиотеки **SALib**.

Функция `run_morris` возвращает `pandas.DataFrame` с колонками
`mu_star` (средний по модулю элементарный эффект) и `sigma`
(стандартное отклонение эффектов) для каждого параметра.

Если SALib не установлен → вызов `ensure_salib()` выбросит ImportError.
"""

from __future__ import annotations
from typing import Mapping, Sequence, List

import numpy as np
import pandas as pd

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.space import DesignSpace
from mwlab.opt.objectives.specification import Specification
from .utils import ensure_salib, batch_eval

# ────────────────────────────────────────────────────────────────────────────
def _space_to_salib_problem(space: DesignSpace) -> Mapping:
    """
    Конвертирует `DesignSpace` → dict, который ждёт SALib:
    {
        'num_vars': d,
        'names'   : ['x1', 'x2', ...],
        'bounds'  : [[lo1, hi1], ...]
    }
    """
    lows, highs = space.bounds()
    return {
        "num_vars": len(space),
        "names": space.names(),
        "bounds": np.stack([lows, highs], axis=1).tolist(),
    }


# ────────────────────────────────────────────────────────────────────────────
def run_morris(
    surrogate: BaseSurrogate,
    space: DesignSpace,
    spec: Specification,
    *,
    N: int = 20,
    sampler: str = "sobol",
    sampler_kw: Mapping | None = None,
    rng: int = 0,
) -> pd.DataFrame:
    """
    Главная «пользовательская» функция для метода Морриса.

    Parameters
    ----------
    surrogate, space, spec :   как и в SensitivityAnalyzer
    N : int
        Число «траекторий» Морриса (чем больше — тем точнее, но симуляций ≈ N*(d+1)).
    sampler : str
        Alias сэмплера DesignSpace (по-умолчанию Sobol).
    rng : int
        Seed воспроизводимости.

    Returns
    -------
    df : pandas.DataFrame
        Индекс = имена параметров, колонки `mu_star`, `sigma`.
    """
    # 1) Проверяем зависимость SALib
    ensure_salib()
    from SALib.sample import morris as morris_sample   # импорт только если точно есть
    from SALib.analyze import morris as morris_analyze

    # 2) Описываем задачу для SALib
    problem = _space_to_salib_problem(space)

    # 3) Генерируем DOE-матрицу (M, d) в физических единицах
    X = morris_sample.sample(
        problem,
        N=N,
        optimal_trajectories=None,   # можно добавить параметр пользователю
        seed=rng,
    )

    # 4) Полный вектор: центральные значения + варьируемые координаты
    lows, highs = space.bounds()
    centers = 0.5 * (lows + highs)

    full_pts = []
    for row in X:
        vec = centers.copy()
        for j, name in enumerate(problem["names"]):
            idx = space.names().index(name)
            vec[idx] = row[j]
        full_pts.append(space.dict(vec))

    pts = full_pts
    Y_bool = batch_eval(surrogate, pts, spec)          # (M,) bool

    # 5) Для аналитики Морриса нужен float, используем 0/1
    Y = Y_bool.astype(float)

    # 6) Анализ
    res = morris_analyze.analyze(
        problem,
        X,
        Y,
        conf_level=0.95,
        print_to_console=False,
        num_levels=4,
        num_resamples=512,
        seed=rng,
    )

    # 7) Укладываем в DataFrame
    df = pd.DataFrame({
        "mu_star": res["mu_star"],
        "sigma":   res["sigma"],
    }, index=problem["names"])

    return df
