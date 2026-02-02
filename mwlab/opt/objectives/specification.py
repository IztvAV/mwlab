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
- Comparator интерпретирует скаляр как pass/fail и возвращает:
  - penalty (штраф),
  - reward (вознаграждение за запас), если компаратор его поддерживает.

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

from typing import Any, Dict, List, Sequence, Literal, Tuple, Optional

import numpy as np

from .base import BaseCriterion, BaseSpecification, CriterionResult
from .network_like import NetworkLike


Reduction = Literal["sum", "mean", "max"]
_REDUCTION_MODES: Tuple[str, ...] = ("sum", "mean", "max")

RewardReduction = Literal["min", "softmin", "mean", "sum", "max"]
_REWARD_REDUCTION_MODES: Tuple[str, ...] = ("min", "softmin", "mean", "sum", "max")


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

        # Проверка уникальности имён и защиты служебных ключей
        names = [c.name for c in self.criteria]
        reserved = {"__all_ok__", "__penalty__", "__reward__"}
        for nm in names:
            if nm in reserved:
                raise ValueError(f"Specification: имя критерия '{nm}' зарезервировано")
        if len(set(names)) != len(names):
            # Можно сделать и мягче (автосуффиксы), но лучше падать явно
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Specification: имена критериев должны быть уникальны, дубликаты: {dup}")

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

    def reward(
        self,
        net: NetworkLike,
        *,
        reduction: RewardReduction = "min",
        tau: float = 0.1,
    ) -> float:
        """
        Сводный reward по спецификации (запас/margin).

        reduction:
          - "min"     : минимальный reward по критериям (предпочтительно)
          - "softmin" : гладкая аппроксимация min, параметр tau>0
          - "mean"    : среднее
          - "sum"     : сумма
        """
        red = str(reduction).strip().lower()
        if red not in _REWARD_REDUCTION_MODES:
            raise ValueError(f"reward reduction должен быть одним из {_REWARD_REDUCTION_MODES}")
        return super().reward(net, reduction=red, tau=float(tau))  # type: ignore[arg-type]

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

    @staticmethod
    def _reduction_rewards(values: np.ndarray, reduction: RewardReduction, *, tau: float) -> float:
        """
        Свёртка reward без повторного пересчёта критериев.
        Поддержка: min/softmin/mean/sum/max.
        """
        x = np.asarray(values, dtype=float)
        if x.size == 0:
            return 0.0

        red = str(reduction).strip().lower()
        if red == "min":
            return float(np.min(x))
        if red == "max":
            return float(np.max(x))
        if red == "mean":
            return float(np.mean(x))
        if red == "sum":
            return float(np.sum(x))
        if red == "softmin":
            t = float(tau)
            if t <= 0.0:
                raise ValueError("reward_tau (tau) must be > 0 for softmin")
            m = float(np.min(x))
            z = np.exp(-(x - m) / t)
            return float(m - t * np.log(np.mean(z)))

        raise ValueError(f"reward reduction должен быть одним из {_REWARD_REDUCTION_MODES}")

    def report(
        self,
        net: NetworkLike,
        *,
        reduction: Reduction = "sum",
        reward_reduction: RewardReduction = "min",
        reward_tau: float = 0.1,
        results: Optional[List[CriterionResult]] = None,
    ) -> Dict[str, Any]:
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
            "raw_reward": float,
            "weighted_reward": float,
            "weight": float,
            "units": {"freq": str, "value": str},
            "chain": {"selector": str, "transform": str, "aggregator": str, "comparator": str},
        }

        Служебные поля:
        - "__all_ok__"  : bool  (AND по ok)
        - "__penalty__" : float (свёртка weighted_penalty по правилу reduction)
        - "__reward__"  : float (свёртка weighted_reward; правило задаётся BaseSpecification.reward)
        """
        # NOTE: один прогон evaluate() даёт и penalty, и reward
        if results is None:
            results = self.evaluate(net)

        rep: Dict[str, Any] = {}
        penalties = np.asarray([r.weighted_penalty for r in results], dtype=float)
        rewards = np.asarray(
            [float(getattr(r, "weighted_reward", 0.0)) for r in results if
             float(getattr(r, "reward_weight", 1.0)) > 0.0],
            dtype = float,
        )
        rewards = np.clip(rewards, 0.0, None)

        for r in results:
            rep[r.name] = {
                "value": float(r.value),
                "ok": bool(r.ok),
                "raw_penalty": float(r.raw_penalty),
                "weighted_penalty": float(r.weighted_penalty),
                "raw_reward": float(getattr(r, "raw_reward", 0.0)),
                "weighted_reward": float(getattr(r, "weighted_reward", 0.0)),
                "weight": float(r.weight),
                "reward_weight": float(getattr(r, "reward_weight", 1.0)),
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
        rep["__reward__"] = self._reduction_rewards(rewards, reward_reduction, tau=float(reward_tau))
        rep["__reward_reduction__"] = str(reward_reduction)
        rep["__reward_tau__"] = float(reward_tau)

        return rep

    # -------------------------------------------------------------------------
    # Представления
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"Specification({self.name}, n={len(self.criteria)})"


__all__ = ["Specification"]