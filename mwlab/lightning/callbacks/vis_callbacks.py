# mwlab/lightning/callbacks/vis_callbacks.py
"""
vis_callbacks.py
================

Колбэки визуализации для PyTorch Lightning + MWLab.
"""

from __future__ import annotations
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from mwlab.lightning.tools.vis_tools import plot_sparams_compare


class PlotSparamsCompareCallback(L.Callback):
    """
    Каждые `every_n_epochs` эпох строит figure с примером работы модели
    (используется plot_sparams_compare) и логирует в TensorBoard.

    Parameters
    ----------
    every_n_epochs : int
        Частота логирования.
    split : 'train'|'val'|'test'|'predict'
        Набор данных для примеров.
    n_samples : int
        Сколько приборов рисовать.
    pairs : list[(m,n)]
        Какие S‑параметры показывать.
    unit : 'dB'|'mag'|'deg'
    """
    def __init__(self,
                 every_n_epochs: int = 25,
                 *,
                 split: str = "val",
                 n_samples: int = 4,
                 pairs=[(1, 1), (2, 1)],
                 unit: str = "dB"):
        super().__init__()
        self.every = every_n_epochs
        self.kw = dict(split=split,
                       n_samples=n_samples,
                       pairs=pairs,
                       unit=unit)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Логируем figure в конце каждой валидационной эпохи."""
        if (trainer.current_epoch + 1) % self.every:
            return
        fig = plot_sparams_compare(trainer, pl_module, trainer.datamodule,
                                   **self.kw)

        logger = trainer.logger
        if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_figure"):
            logger.experiment.add_figure(
                f"{self.kw['split']}/sparams_compare",
                fig,
                global_step=trainer.global_step,
            )
        plt.close(fig)

