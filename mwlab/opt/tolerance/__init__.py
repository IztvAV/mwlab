"""
mwlab.opt.tolerance
===================
Подсистема поиска **технологических допусков** (tolerance allocation).

Содержит два уровня абстракции:
--------------------------------
* `IndividualBoundsFinder` – по-одиночке подбирает δ для каждого параметра
  (One-At-a-Time бинарный поиск, OAT-BS).  Даёт *верхние* оценки.
* `JointBoxOptimizer`      – совместно оптимизирует гиперблок δ так,
  чтобы `Yield ≥ target`.  Принимает ограничения δ ≤ δ_upper.

Фасад `ToleranceAnalyzer` предоставляет короткое API:
>>> ta = ToleranceAnalyzer(surrogate, space, spec)
>>> δ_hi = ta.upper_bounds(params, target_yield=0.99)
>>> δ_ok = ta.optimize_box(delta_upper=δ_hi, target_yield=0.99)
"""

from .core import ToleranceAnalyzer
from .projection import ProjectionBoxOptimizer

__all__ = ["ToleranceAnalyzer", "ProjectionBoxOptimizer"]
