#mwlab/opt/objectives/metrics.py
"""
mwlab.opt.objectives.metrics
============================
**Метрики «проходит / не проходит ТЗ»** на базе `Specification`.

Зачем нужны
-----------
* Оценить, насколько **правильно** суррогатная модель предсказывает
  выполнение технического задания — независимо от MSE.
* Помочь принять решение «достаточно ли точна модель, чтобы искать допуски,
  или нужно до-симулировать еще CST/HFSS-точек».

Доступные классы
----------------
* `SpecPassAccuracy`  – общая точность (TP+TN / ALL)
* `SpecRecall`        – полнота, *True-Positive Rate*
* `SpecPrecision`     – точность, *Positive Predictive Value*
* `SpecF1`            – гармоническое среднее Precision и Recall

Как работают
------------
1. Получают **сырые** тензоры *(batch, …)* из `validation_step`.
2. (Если задан) снимают масштабирующие скейлеры выхода.
3. Декодируют тензор → `rf.Network` через `TouchstoneCodec`.
4. С помощью `Specification.is_ok(net)` переводят в bool (0/1).
5. Передают bool-вектора во встроенный класс из `torchmetrics`
   (`Accuracy`, `Recall`, `Precision`, `F1Score`).

Пример подключения к Lightning-модулю
-------------------------------------
```python
from torchmetrics import MeanSquaredError
from mwlab.opt.objectives.metrics import (
        SpecPassAccuracy, SpecRecall, SpecPrecision, SpecF1)
from mwlab.opt.objectives.selectors import SMagSelector, AxialRatioSelector
from mwlab.opt.objectives.aggregators import MaxAgg, MinAgg
from mwlab.opt.objectives.comparators import LEComparator, GEComparator
from mwlab.opt.objectives.base import BaseCriterion
from mwlab.opt.objectives.specification import Specification

# --- формируем ТЗ ----------------------------------------------------------
crit_s11 = BaseCriterion(
    selector   = SMagSelector(1, 1, db=True),   # S11
    aggregator = MaxAgg(),                      # максимум по частоте
    comparator = LEComparator(-22, unit="dB"),  # < -22 dB
    name="S11"
)
crit_s21 = BaseCriterion(
    selector   = SMagSelector(2, 1, db=True),   # S21
    aggregator = MaxAgg(),
    comparator = LEComparator(-22, unit="dB"),
    name="S21"
)
crit_ar = BaseCriterion(
    selector   = AxialRatioSelector(db=False),  # AR без dB!
    aggregator = MinAgg(),
    comparator = GEComparator(0.96),
    name="AxialRatio"
)
spec = Specification([crit_s11, crit_s21, crit_ar])

# --- метрики ---------------------------------------------------------------
from torchmetrics import MetricCollection
metrics = MetricCollection({
    "mse"       : MeanSquaredError(),
    "spec_acc"  : SpecPassAccuracy(specification=spec,
                                   codec=codec,     # TouchstoneCodec
                                   scaler_out=scaler_out),
    "spec_rec"  : SpecRecall(specification=spec,
                             codec=codec, scaler_out=scaler_out),
    "spec_prec" : SpecPrecision(specification=spec,
                                codec=codec, scaler_out=scaler_out),
    "spec_f1"   : SpecF1(specification=spec,
                         codec=codec, scaler_out=scaler_out),
})

# --- Lightning-модуль ------------------------------------------------------
pl_module = BaseLMWithMetrics(model=model,
                              codec=codec,
                              scaler_out=scaler_out,
                              metrics=metrics)
При обучении вы получите в TensorBoard:
`val_spec_acc`, `val_spec_rec`, `val_spec_prec`, `val_spec_f1`.
"""

from __future__ import annotations

from typing import List
import torch
from torch import Tensor
from torchmetrics import Accuracy, Recall, Precision, F1Score

import skrf as rf  # используется в decode_s, неявно
from .specification import Specification
from mwlab.codecs.touchstone_codec import TouchstoneCodec

#───────────────────────────────────────────── helper inverse (общий)
def _inverse_if_needed(t: Tensor, scaler):
    """
    Снимает нормализацию, если передан scaler.
    Поддерживаются оба варианта API: .inverse() и .inverse_transform().
    """
    if scaler is None:
        return t
    if hasattr(scaler, "inverse"):
        return scaler.inverse(t)
    if hasattr(scaler, "inverse_transform"):
        arr = scaler.inverse_transform(t.cpu().numpy())
        return torch.as_tensor(arr, dtype=t.dtype, device=t.device)
    raise AttributeError("scaler has neither inverse nor inverse_transform")

#───────────────────────────────────────────── базовый класс
class _SpecMetricMixin:
    """
    Микс-ин: реализует update(preds, target) для метрик pass/fail.
    """

    def __init__(
        self,
        *,
        specification: Specification,
        codec: TouchstoneCodec,
        scaler_out=None,
        **kwargs,
    ):
        # 1) забираем task, если вдруг передали вручную
        task = kwargs.pop("task", "binary")

        # 2) инициализируем настоящий класс метрики
        #    positional → для Accuracy/Recall/… фабрики
        super().__init__(task, **kwargs)  # type: ignore[misc]

        # 3) собственные атрибуты
        self.spec = specification
        self.codec = codec
        self.scaler_out = scaler_out

    def update(self, preds: Tensor, target: Tensor):  # type: ignore[override]
        preds = _inverse_if_needed(preds, self.scaler_out)
        target = _inverse_if_needed(target, self.scaler_out)

        y_hat: List[bool] = []
        y_true: List[bool] = []

        for p, y in zip(preds, target):
            net_pred = self.codec.decode_s(p)
            net_true = self.codec.decode_s(y)
            y_hat.append(self.spec.is_ok(net_pred))
            y_true.append(self.spec.is_ok(net_true))

        #ph = preds.new_tensor(y_hat, dtype=torch.float32)
        #pt = preds.new_tensor(y_true, dtype=torch.float32)
        ph = torch.tensor(y_hat, dtype=torch.bool, device=preds.device)
        pt = torch.tensor(y_true, dtype=torch.bool, device=preds.device)

        super().update(ph, pt)  # type: ignore[arg-type]


#───────────────────────────────────────────── конкретные метрики
class SpecPassAccuracy(_SpecMetricMixin, Accuracy):
    """Accuracy pass/fail относительно Specification."""

class SpecRecall(_SpecMetricMixin, Recall):
    """Recall (TPR) pass/fail относительно Specification."""

class SpecPrecision(_SpecMetricMixin, Precision):
    """Precision (PPV) pass/fail относительно Specification."""

class SpecF1(_SpecMetricMixin, F1Score):
    """F1-score pass/fail относительно Specification."""


__all__ = [
    "SpecPassAccuracy",
    "SpecRecall",
    "SpecPrecision",
    "SpecF1",
]

