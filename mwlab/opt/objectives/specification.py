#mwlab/opt/objectives/specification.py
"""
mwlab.opt.objectives.specification
==================================
**Specification** — формализованное «техническое задание» для СВЧ-устройства.

* Состоит из *нескольких* `Criterion`-ов, объединённых логическим **AND**.
* Каждый Criterion — это тройка
  `Selector` → `Aggregator` → `Comparator`
  (см. `base.py`, `selectors.py`, `aggregators.py`, `comparators.py`).

Основные публичные методы
-------------------------
* `is_ok(net)`     → bool    — выполняется ли ТЗ?
* `penalty(net)`   → float   — суммарный штраф ≥ 0 (0 ⇔ ОК)
* `report(net)`    → dict    — подробный отчёт по каждому критерию.

Пример
-------
>>> # Критерий: |S11| ≤ –22 dB в полосе 2-2.4 ГГц
>>> crit_s11 = BaseCriterion(
...     selector   = SMagSelector(1, 1, band=(2.0, 2.4), db=True),
...     aggregator = MaxAgg(),
...     comparator = LEComparator(limit=-22, unit="dB"),
...     weight = 1,
...     name="S11_inband"
... )
>>>
>>> spec = Specification([crit_s11])
>>> ok = spec.is_ok(net)          # net — skrf.Network
>>> pen = spec.penalty(net)       # 0 → pass
>>> print(spec.report(net))       # подробности
"""
from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple

import numpy as np
import skrf as rf

from .base import BaseCriterion


# ────────────────────────────────────────────────────────────────────────────
class Specification:
    """
    Коллекция `Criterion`-ов, соединенных операцией «и».

    Parameters
    ----------
    criteria : Sequence[BaseCriterion]
        Список критериев (должен быть непустым).
    name : str, optional
        Для логов/отчётов (по-умолчанию “spec”).

    Notes
    -----
    *Penalty* используется в задачах оптимизации с мягкими ограничениями
    (cost-function = Σ penalty).  При yield-анализе берется только
    булево `is_ok`.
    """

    # ---------------------------------------------------------------- init
    def __init__(self, criteria: Sequence[BaseCriterion], *, name: str = "spec"):
        if not criteria:
            raise ValueError("Specification: список criteria пуст")
        self.criteria: Tuple[BaseCriterion, ...] = tuple(criteria)
        self.name = str(name)

    # ======================================================================
    #                       PUBLIC API
    # ======================================================================
    # ---------------------------------------------------------------- value helpers
    def values(self, net: rf.Network) -> Dict[str, float]:
        """
        Возвращает словарь «имя критерия → агрегированное значение».
        """
        return {c.name: c.value(net) for c in self.criteria}

    # ---------------------------------------------------------------- is_ok
    def is_ok(self, net: rf.Network) -> bool:
        """
        True, если **все** критерии удовлетворены (pass).
        """
        return all(c.is_ok(net) for c in self.criteria)

    # ---------------------------------------------------------------- penalty
    def penalty(self, net: rf.Network, *, reduce: str = "sum") -> float:
        """
        Суммарный штраф ≥ 0.

        Parameters
        ----------
        reduce : {'sum', 'mean', 'max'}, default='sum'
            Как агрегировать individual penalty каждого критерия.
        """
        vals = np.fromiter((c.penalty(net) for c in self.criteria), dtype=float)
        if reduce == "sum":
            return float(np.sum(vals))
        if reduce == "mean":
            return float(np.mean(vals))
        if reduce == "max":
            return float(np.max(vals))
        raise ValueError("reduce должен быть 'sum' | 'mean' | 'max'")

    # ---------------------------------------------------------------- report
    def report(self, net: rf.Network) -> Dict[str, Any]:
        """
        Подробный отчёт по всем критериям + итоговые поля
        '__all_ok__' и '__penalty__'.
        """
        rep: Dict[str, Any] = {}

        for c in self.criteria:
            val = c.value(net)
            rep[c.name] = {
                "value": val,
                "ok": c.comp.is_ok(val),  # type: ignore[attr-defined]
                "penalty": c.comp.penalty(val),  # type: ignore[attr-defined]
                "weight": c.weight,
            }

        # ←–– сводные показатели считаем до добавления в rep
        total_penalty = sum(d["penalty"] for d in rep.values())
        all_ok = all(d["ok"] for d in rep.values())

        rep["__all_ok__"] = all_ok
        rep["__penalty__"] = total_penalty
        return rep

    # ---------------------------------------------------------------- len / iter helpers
    def __len__(self):            # pragma: no cover
        return len(self.criteria)

    def __iter__(self):           # pragma: no cover
        return iter(self.criteria)

    # ---------------------------------------------------------------- repr / str
    def __repr__(self):           # pragma: no cover
        inner = ", ".join(c.name for c in self.criteria)
        return f"Specification({inner})"

    __str__ = __repr__