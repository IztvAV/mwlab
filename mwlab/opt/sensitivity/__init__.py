# mwlab/opt/sensitivity/__init__.py
"""
mwlab.opt.sensitivity
=====================
Подпакет для методов анализа чувствительности (Morris, Sobol, Active-Subspace).

* При импорте один раз активируем «красивый» стиль seaborn.
* Пользователь видит только единый фасад `SensitivityAnalyzer`.
"""

from __future__ import annotations
import warnings, re

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=re.escape("unique with argument"),
)

# ─────────────────────────── оформление графиков ──────────────────────────
import seaborn as sns

# Whitegrid-тема смотрится аккуратно и не конфликтует с фирменным стилем MWLab.
sns.set_theme(style="whitegrid")

# ─────────────────────────── публичный фасад ──────────────────────────────
from .core import SensitivityAnalyzer  # noqa: E402  (импорт после seaborn)

__all__ = ["SensitivityAnalyzer"]


