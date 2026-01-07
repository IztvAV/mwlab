# mwlab/opt/surrogates/base.py
"""
mwlab.opt.surrogates.base
=========================

Абстрактный базовый класс BaseSurrogate.

Назначение
----------
Определяет минимальный контракт для суррогат-моделей (NN, GP, RBF и т.п.),
чтобы оптимизаторы и анализаторы могли работать с ними единообразно.

Ключевые принципы
-----------------
1) В MVP обязателен только predict(x) — точечный прогноз.
2) batch_predict(xs) по умолчанию фолбэчится на predict в цикле.
3) passes_spec(xs, spec) даёт универсальную проверку выполнимости ТЗ
   через Specification.is_ok(...), но может быть переопределён потомками
   для ускорения (например, NN может считать критерии “векторно” без создания
   тяжёлых объектов).
4) penalty_spec(xs, spec, reduction="sum") — универсальный расчёт штрафа
   по Specification.penalty(...) (также может быть переопределён потомками).

Важно про типы x/xs
-------------------
В MWLab дизайн-параметры могут быть не только float:
- int (IntegerVar),
- str / уровни (CategoricalVar, OrdinalVar),
- иногда bool и т.п.

Поэтому в базовом API используется Mapping[str, Any], а не Mapping[str, float].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence, List, Tuple, Union, TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    # Импорт только для типизации, чтобы избежать тяжёлых/циклических импортов в runtime.
    from mwlab.opt.objectives.specification import Specification


class BaseSurrogate(ABC):
    """
    Интерфейс surrogate-модели.

    Атрибуты
    --------
    supports_uncertainty : bool
        Если True, surrogate поддерживает return_std=True и возвращает σ.
    """

    supports_uncertainty: bool = False

    # ---------------------------------------------------------------------
    # Обязательный API
    # ---------------------------------------------------------------------
    @abstractmethod
    def predict(
        self,
        x: Mapping[str, Any],
        *,
        return_std: bool = False,
    ) -> Any | Tuple[Any, float]:
        """
        Точечный прогноз.

        Параметры
        ---------
        x : Mapping[str, Any]
            Словарь параметров в “физическом виде” (float/int/str/...).
        return_std : bool
            Если True и модель поддерживает неопределённость (supports_uncertainty=True),
            вернуть (y, sigma). Иначе → NotImplementedError.

        Возвращает
        ----------
        y : Any
            Обычно это объект, похожий на rf.Network (duck-typing),
            либо иной формат, понятный objective/spec.
        (y, sigma) : Tuple[Any, float]
            Если return_std=True и модель поддерживает неопределённость.
        """

    # ---------------------------------------------------------------------
    # Батчевый API (опционально ускорять в потомках)
    # ---------------------------------------------------------------------
    def batch_predict(
        self,
        xs: Sequence[Mapping[str, Any]],
        *,
        return_std: bool = False,
    ) -> List[Any] | Tuple[List[Any], List[float]]:
        """
        Батчевый прогноз.

        Базовая реализация — безопасный fallback: вызывает predict(...) в цикле.
        Потомки (особенно NN) почти всегда переопределяют этот метод, чтобы
        делать векторизованный/батчевый инференс.

        Замечание по контракту:
        - если return_std=True, но supports_uncertainty=False → NotImplementedError
        """
        if return_std and not self.supports_uncertainty:
            raise NotImplementedError("This surrogate does not support uncertainty (σ).")

        ys: List[Any] = []
        sigmas: List[float] = []

        for x in xs:
            out = self.predict(x, return_std=return_std)
            if return_std:
                # В этом режиме ожидаем (y, sigma)
                if not isinstance(out, tuple) or len(out) != 2:
                    raise TypeError(
                        "batch_predict(..., return_std=True): predict должен возвращать (y, sigma)"
                    )
                y, s = out
                ys.append(y)
                sigmas.append(float(s))
            else:
                ys.append(out)

        return (ys, sigmas) if return_std else ys

    # ---------------------------------------------------------------------
    # Проверка спецификации (универсальный fallback)
    # ---------------------------------------------------------------------
    def passes_spec(
        self,
        xs: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        spec: "Specification",
    ) -> Union[bool, np.ndarray]:
        """
        Проверить, удовлетворяют ли предсказания surrogate спецификации spec.

        Поведение
        ---------
        xs = dict:
            возвращает bool

        xs = batch (Sequence[dict]):
            возвращает np.ndarray[bool] формы (B,)

        Реализация
        ----------
        Базовая реализация ничего не знает о внутреннем формате surrogate:
        - делает predict/batch_predict,
        - затем вызывает spec.is_ok(...) на каждом предсказании.

        Потомки могут существенно ускорить этот метод, например:
        - вычислять нужные метрики напрямую “на тензорах”,
        - не создавая тяжёлые объекты сетей на каждую точку.
        """
        # --- одиночная точка
        if isinstance(xs, Mapping):
            y = self.predict(xs)
            return bool(spec.is_ok(y))

        # --- batch
        preds = self.batch_predict(xs)  # list[Any]
        # count=len(preds) делает fromiter быстрее и предсказуемее по памяти.
        return np.fromiter((bool(spec.is_ok(p)) for p in preds), dtype=bool, count=len(preds))

    # ---------------------------------------------------------------------
    # Штраф по спецификации (универсальный fallback)
    # ---------------------------------------------------------------------
    def penalty_spec(
        self,
        xs: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        spec: "Specification",
        *,
        reduction: Literal["sum", "mean", "max"] = "sum",
    ) -> Union[float, np.ndarray]:
        """
        Рассчитать штраф(ы) по спецификации spec для предсказаний surrogate.

        Поведение
        ---------
        xs = dict:
            возвращает float

        xs = batch (Sequence[dict]):
            возвращает np.ndarray[float] формы (B,)

        Реализация
        ----------
        Базовая реализация ничего не знает о внутреннем формате surrogate:
        - делает predict/batch_predict,
        - затем вызывает spec.penalty(...) на каждом предсказании.

        Потомки могут ускорить, например:
        - считать метрики напрямую “на тензорах”,
        - не создавая тяжёлые объекты, если они не нужны.
        """
        # --- одиночная точка
        if isinstance(xs, Mapping):
            y = self.predict(xs)
            return float(spec.penalty(y, reduction=reduction))

        # --- batch
        preds = self.batch_predict(xs)
        return np.fromiter(
            (float(spec.penalty(p, reduction=reduction)) for p in preds),
            dtype=float,
            count=len(preds),
        )

    # ---------------------------------------------------------------------
    # Persistence (опционально)
    # ---------------------------------------------------------------------
    def save(self, path: str | Path):
        """Сохранить surrogate на диск. По умолчанию не реализовано."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path):
        """Загрузить surrogate с диска. По умолчанию не реализовано."""
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Optional API (для будущего: GP/RBF/градиенты)
    # ---------------------------------------------------------------------
    def fit(self, X, Y):
        """Обучение модели (актуально для GP/RBF). Для NN-обёртки может не требоваться."""
        raise NotImplementedError("fit() не реализован для этого surrogate")

    def grad(self, x: Mapping[str, Any]):
        """Градиент по входам: понадобится для gradient-based оптимизации."""
        raise NotImplementedError("grad() не реализован")
