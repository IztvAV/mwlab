#mwlab/opt/surrogates/base.py
"""
Абстрактный базовый класс `BaseSurrogate`.

Определяет **минимальный контракт**, чтобы любой оптимизатор/анализатор
мог работать с surrogate, не зная его природы (NN, GP, RBF …).

Методы, помеченные `NotImplementedError`, можно постепенно
реализовывать в потомках, не ломая уже написанный код.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence, List, Tuple, Union
import numpy as np

class BaseSurrogate(ABC):
    """Интерфейс surrogate-модели.

    *В MVP* обязателен **только** метод `predict`.  Все остальное —
    наращивается по мере появления GP / RBF / multi-fidelity.
    """

    supports_uncertainty: bool = False  # переопределяют те, кто умеет σ

    # ────────────────────────────── API пользователя ────────────────────
    @abstractmethod
    def predict(
        self,
        x: Mapping[str, float],
        *,
        return_std: bool = False,
    ) -> Any | Tuple[Any, float]:
        """Один точечный прогноз.

        * `x` – словарь параметров в «физических единицах».
        * Если `return_std=True` и модель поддерживает неопределенность,
          возвращает `(y, σ)`.  Иначе ⇒ NotImplementedError.
        """

    def batch_predict(
        self,
        xs: Sequence[Mapping[str, float]],
        *,
        return_std: bool = False,
    ) -> List[Any] | Tuple[List[Any], List[float]]:
        """Конвенция: по-умолчанию просто итерирует `predict`."""
        ys, sig = [], []
        for x in xs:
            out = self.predict(x, return_std=return_std)
            if return_std and self.supports_uncertainty:
                y, s = out
                ys.append(y)
                sig.append(s)
            else:
                ys.append(out)
        return (ys, sig) if (return_std and self.supports_uncertainty) else ys

    def passes_spec(                     # <-- универсальный интерфейс
        self,
        xs: Union[Mapping[str, float],
                  Sequence[Mapping[str, float]]],
        spec: "Specification",
    ):
        """
        Вернуть указатель(и) «проходит ли *spec*».

        • Если `xs` – один словарь → bool.
        • Если `xs` – последовательность → numpy.bool_ массив.
        """
        if isinstance(xs, Mapping):          # одиночная точка
            y = self.predict(xs)
            return bool(spec.is_ok(y))

        # fallback-batch: используем batch_predict + поэлементно
        preds = self.batch_predict(xs)
        return np.fromiter((spec.is_ok(p) for p in preds), dtype=bool)

    # ────────────────────────────── Persistence ─────────────────────────
    def save(self, path: str | Path):
        """Сохраняет surrogate на диск (потомки могут переопределить)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path):
        """Восстанавливает surrogate (класс-метод)."""
        raise NotImplementedError

    # ────────────────────────────── Optional API ────────────────────────
    def fit(self, X, Y):
        """Для GP/RBF.  NN-обертке не требуется."""
        raise NotImplementedError("fit() не реализован для этого surrogate")

    def grad(self, x: Mapping[str, float]):
        """Возврат ∇y – понадобится для gradient-based оптимизации."""
        raise NotImplementedError("grad() не реализован")

