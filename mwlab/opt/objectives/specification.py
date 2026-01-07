# mwlab/opt/objectives/specification.py
"""
mwlab.opt.objectives.specification
=================================

Модуль предоставляет «пользовательский фасад» для формализации требований к СВЧ-устройству
в терминах mwlab: набор критериев, объединённых логическим AND.

Зачем нужен отдельный модуль, если есть BaseSpecification?
----------------------------------------------------------
В `base.py` определены базовые строительные блоки подсистемы целей/ограничений:
`BaseCriterion`, `CriterionResult`, `BaseSpecification`.

Концепция Specification
-----------------------
Specification = AND-набор Criterion-ов.

Каждый Criterion — композиция:
    Selector ∘ Transform ∘ Aggregator ∘ Comparator

- Selector извлекает кривую из `NetworkLike`
- Transform (опционально) предварительно обрабатывает кривую
- Aggregator сворачивает кривую в скаляр
- Comparator интерпретирует скаляр как pass/fail и возвращает штраф

Выходные данные (report)
------------------------
`report(net)` формируется **из CriterionResult**, который возвращает `BaseCriterion.evaluate()`.
Таким образом, отчёт:
- не дублирует вычисления,
- содержит метаданные единиц (`freq_unit`, `value_unit`),
- содержит информацию о цепочке компонентов (классы Selector/Transform/Aggregator/Comparator),
- устойчив к политике NaN/Inf на стороне Comparator.

"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Literal, Tuple

import numpy as np

from .base import BaseCriterion, BaseSpecification, CriterionResult
from .network_like import NetworkLike


Reduction = Literal["sum", "mean", "max"]
_REDUCTION_MODES: Tuple[str, ...] = ("sum", "mean", "max")


class Specification(BaseSpecification):
    """
    Пользовательская спецификация (набор критериев), объединённая логическим AND.

    Parameters
    ----------
    criteria : Sequence[BaseCriterion]
        Непустой список критериев.
    name : str, default="spec"
        Имя спецификации для логов/отчётов.

    Notes
    -----
    - В задачах yield/верификации используется `is_ok(net)`.
    - В задачах оптимизации с мягкими ограничениями используется `penalty(net)`,
      при этом конкретная форма штрафов определяется компараторами.
    """

    def __init__(self, criteria: Sequence[BaseCriterion], *, name: str = "spec"):
        if not criteria:
            raise ValueError("Specification: список criteria пуст")
        super().__init__(criteria=criteria, name=name)

    # -------------------------------------------------------------------------
    # Основные операции
    # -------------------------------------------------------------------------
    def penalty(self, net: NetworkLike, *, reduction: Reduction = "sum") -> float:
        """
        Суммарный штраф по спецификации.

        Parameters
        ----------
        reduction : {"sum","mean","max"}, default="sum"
            Правило свёртки штрафов отдельных критериев:
            - "sum"  : сумма weighted_penalty
            - "mean" : среднее weighted_penalty
            - "max"  : худший критерий (minimax)

        Returns
        -------
        float
            Итоговый штраф (>=0 для корректно настроенных компараторов).
        """
        red = str(reduction).strip().lower()
        if red not in _REDUCTION_MODES:
            raise ValueError(f"reduction должен быть одним из {_REDUCTION_MODES}")

        return super().penalty(net, reduction=red)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # Удобные “срезы” результатов
    # -------------------------------------------------------------------------
    def values(self, net: NetworkLike) -> Dict[str, float]:
        """
        Вернуть словарь «имя критерия -> агрегированное значение».

        Замечание:
        Для согласованности и эффективности значения берутся из `evaluate(net)`,
        а не вычисляются повторно через `c.value(net)`.
        """
        res = self.evaluate(net)
        return {r.name: float(r.value) for r in res}

    # -------------------------------------------------------------------------
    # Отчёт
    # -------------------------------------------------------------------------
    @staticmethod
    def _reduction_penalties(values: np.ndarray, reduction: Reduction) -> float:
        """
        Внутренняя свёртка штрафов без повторного пересчёта критериев.

        Parameters
        ----------
        values : np.ndarray
            1-D массив штрафов (weighted_penalty).
        reduction : {"sum","mean","max"}
            Правило свёртки.

        Returns
        -------
        float
            Сводный штраф.
        """
        if values.size == 0:
            return 0.0

        red = str(reduction).strip().lower()
        if red == "sum":
            return float(np.sum(values))
        if red == "mean":
            return float(np.mean(values))
        if red == "max":
            return float(np.max(values))

        raise ValueError(f"reduction должен быть одним из {_REDUCTION_MODES}")

    def report(self, net: NetworkLike, *, reduction: Reduction = "sum") -> Dict[str, Any]:
        """
        Подробный отчёт по всем критериям и сводные поля.

        Parameters
        ----------
        reduction : {"sum","mean","max"}, default="sum"
            Как формировать поле '__penalty__' в отчёте.

        Returns
        -------
        Dict[str, Any]
            Структура отчёта. Для удобства человека и сериализации в JSON
            используется словарь.

        Формат отчёта
        -------------
        report[name] = {
            "value": float,
            "ok": bool,
            "raw_penalty": float,
            "weighted_penalty": float,
            "weight": float,
            "units": {"freq": str, "value": str},
            "chain": {"selector": str, "transform": str, "aggregator": str, "comparator": str},
        }

        Служебные поля:
        - "__all_ok__"  : bool  (AND по ok)
        - "__penalty__" : float (свёртка weighted_penalty по правилу reduction)
        """
        results: List[CriterionResult] = self.evaluate(net)

        rep: Dict[str, Any] = {}
        penalties = np.asarray([r.weighted_penalty for r in results], dtype=float)

        for r in results:
            rep[r.name] = {
                "value": float(r.value),
                "ok": bool(r.ok),
                "raw_penalty": float(r.raw_penalty),
                "weighted_penalty": float(r.weighted_penalty),
                "weight": float(r.weight),
                "units": {
                    "freq": str(r.freq_unit),
                    "value": str(r.value_unit),
                },
                "chain": {
                    "selector": str(r.selector),
                    "transform": str(r.transform),
                    "aggregator": str(r.aggregator),
                    "comparator": str(r.comparator),
                },
            }

        rep["__all_ok__"] = bool(all(r.ok for r in results))
        rep["__penalty__"] = self._reduction_penalties(penalties, reduction)

        return rep

    # -------------------------------------------------------------------------
    # Представления
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"Specification({self.name}, n={len(self.criteria)})"


__all__ = ["Specification"]
