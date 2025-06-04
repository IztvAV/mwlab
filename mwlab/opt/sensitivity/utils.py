#mwlab/opt/sensitivity/utils.py
"""
utils.py
========
Вспомогательные функции, используемые всеми анализаторами.

* `batch_eval`          – быстрый batched прогон surrogate → scalar y.
* `ensure_salib`        – ленивое подтверждение, что установлен SALib.
* `barplot_dataframe`   – единообразный bar-plot (matplotlib + seaborn-стиль).
"""

from __future__ import annotations
from typing import List, Mapping, Sequence

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.space import DesignSpace
from mwlab.opt.objectives.specification import Specification


# ────────────────────────────────────────────────────────────────────────────
def batch_eval(
    surrogate: BaseSurrogate,
    points: Sequence[Mapping[str, float]],
    specification: Specification,
) -> np.ndarray:
    """
    Прогоняет **батч** параметров через surrogate и возвращает bool-маску
    `passed`, где `True` означает «устройство прошло ТЗ».

    Пояснения
    ---------
    * Surrogate может отдавать сложные объекты (`rf.Network`).
      Мы проверяем каждый через `specification.is_ok(net)`.
    * Вернётся NumPy-вектор shape (B,), dtype=bool.
    """
    preds = surrogate.batch_predict(points)      # быстрый forward (GPU)
    return np.fromiter(
        (specification.is_ok(net) for net in preds),
        dtype=bool,
    )


# ────────────────────────────────────────────────────────────────────────────
def ensure_salib() -> None:
    """
    Гарантирует, что пакет **SALib** доступен.

    * Если модуль импортируется успешно – ничего не делает.
    * Если SALib не найден, выбрасывает детализированный ImportError
      с инструкцией `pip install mwlab[analysis]`.
    """
    if importlib.util.find_spec("SALib") is None:
        raise ImportError(
            "Для этого метода требуется пакет 'SALib'. "
            "Установите опциональную зависимость:  pip install mwlab[analysis]"
        )


# ────────────────────────────────────────────────────────────────────────────
def barplot_dataframe(
    series: pd.Series,
    *,
    title: str,
    ylabel: str | None = None,
    figsize_scale: float = 0.25,
):
    """
    Рисует горизонтальный bar-plot важности параметров.

    • `series`          – именованный pandas.Series;
    • `figsize_scale`   – коэффициент для автоматической ширины фигуры.
    """
    series = series.sort_values(ascending=True)    # горизонтально снизу-вверх
    fig_w = max(4.0, figsize_scale * len(series))  # auto-width

    ax = series.plot.barh(figsize=(fig_w, 4))
    ax.set_title(title)
    ax.set_xlabel(ylabel or series.name)
    ax.grid(True, axis="x", ls=":", alpha=0.6)
    plt.tight_layout()
    plt.show(block=False)
    return ax.figure
