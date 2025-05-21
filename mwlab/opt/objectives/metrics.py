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
1. Получают батч **сырых** тензоров S-параметров из `validation_step`.
2. При необходимости снимают нормализацию (`scaler_out.inverse`).
3. Декодируют тензор → `rf.Network` через `TouchstoneCodec`.
4. С помощью `Specification.is_ok(net)` превращают в bool (0 / 1).
5. Передают bool-вектора во встроенную метрику `torchmetrics`.

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
from torchmetrics.classification import (
    BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score,
)
from .specification import Specification
from mwlab.codecs.touchstone_codec import TouchstoneCodec

#───────────────────────────────────────────── helpers
def _inverse_if_needed(t: Tensor, scaler):
    """
    Снимает нормализацию, если передан `scaler`.

    * Поддерживает API `.inverse()` и `.inverse_transform()`.
    * В остальных случаях возвращает вход без изменений.
    """
    if scaler is None:
        return t

    if hasattr(scaler, "inverse"):
        return scaler.inverse(t)

    if hasattr(scaler, "inverse_transform"):
        arr = scaler.inverse_transform(t.detach().cpu().numpy())
        return torch.as_tensor(arr, dtype=t.dtype, device=t.device)

    raise AttributeError("Scaler has neither inverse() nor inverse_transform().")


def _unflatten_with_codec(t: Tensor, codec: TouchstoneCodec) -> Tensor:
    """
    Приводит плоский вектор `(B, C*F)` к форме `(B, C, F)`,
    исходя из `codec.y_channels` (`C`) и `codec.freq_hz` (`F`).

    Если размерность не совпадает — возвращает тензор без изменений.
    """
    if t.dim() == 1:                            # (C*F,) → (1, C*F)
        t = t.unsqueeze(0)

    B, prod = t.shape
    C = len(codec.y_channels)
    F = len(codec.freq_hz)

    if prod != C * F:
        return t                                # уже «правильная» форма

    return t.view(B, C, F)

#───────────────────────────────────────────── базовый mix-in
class _SpecMetricMixin:
    """
    Микс-ин: формирует bool-метки pass/fail и вызывает `update()` родительской
    *Binary*\*-метрики.
    """

    def __init__(
        self,
        *,
        specification: Specification,
        codec: TouchstoneCodec,
        scaler_out=None,
        **kwargs,
    ):
        super().__init__(**kwargs)  # type: ignore[misc]
        self.spec = specification
        self.codec = codec
        self.scaler_out = scaler_out

    def update(self, preds: Tensor, target: Tensor):  # type: ignore[override]
        # 1) восстанавливаем форму (B, C, F), если пришёл «флэт»
        preds = _unflatten_with_codec(preds, self.codec)
        target = _unflatten_with_codec(target, self.codec)

        # 2) возвращаем данные из «скейл-пространства» в инженерные единицы
        preds = _inverse_if_needed(preds, self.scaler_out)
        target = _inverse_if_needed(target, self.scaler_out)

        # 3) получаем метки pass/fail
        y_hat: List[bool] = []
        y_true: List[bool] = []

        for p, y in zip(preds, target):
            net_pred = self.codec.decode_s(p)
            net_true = self.codec.decode_s(y)
            y_hat.append(self.spec.is_ok(net_pred))
            y_true.append(self.spec.is_ok(net_true))

        ph = torch.tensor(y_hat, dtype=torch.bool, device=preds.device)
        pt = torch.tensor(y_true, dtype=torch.bool, device=preds.device)
        super().update(ph, pt)  # type: ignore[arg-type]


#───────────────────────────────────────────── конкретные метрики
class SpecPassAccuracy(_SpecMetricMixin, BinaryAccuracy):
    """Accuracy (TP+TN)/ALL относительно `Specification`."""

class SpecRecall(_SpecMetricMixin, BinaryRecall):
    """Recall / True-Positive Rate относительно `Specification`."""

class SpecPrecision(_SpecMetricMixin, BinaryPrecision):
    """Precision / Positive Predictive Value относительно `Specification`."""

class SpecF1(_SpecMetricMixin, BinaryF1Score):
    """F1-score (гармоническое среднее Precision и Recall)."""


__all__ = [
    "SpecPassAccuracy",
    "SpecRecall",
    "SpecPrecision",
    "SpecF1",
]

