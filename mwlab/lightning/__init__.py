# mwlab/mwfilter_lightning/__init__.py
"""
Под‑пакет MWLab для PyTorch Lightning.
"""

from .base_lm import BaseLModule            # noqa: F401
from .base_lm_with_metrics import BaseLMWithMetrics  # noqa: F401
from .touchstone_ldm import TouchstoneLDataModule    # noqa: F401

# новые инструменты и колбэки
from .tools.vis_tools import plot_sparams_compare    # noqa: F401
from .callbacks.vis_callbacks import PlotSparamsCompareCallback # noqa: F401


__all__ = [
    "BaseLModule",
    "BaseLMWithMetrics",
    "TouchstoneLDataModule",
    "plot_sparams_compare",
    "PlotSparamsCompareCallback",
]