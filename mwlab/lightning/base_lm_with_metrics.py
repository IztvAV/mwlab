# mwlab/mwfilter_lightning/base_lm_with_metrics.py
"""
BaseLMWithMetrics
=================

Расширяет `BaseLModule`, добавляя накопление и логирование произвольных
метрик во время **validate / test**.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score, Metric
from typing import Optional, Callable, Any

from mwlab.lightning.base_lm import BaseLModule
from mwlab.codecs.touchstone_codec import TouchstoneCodec


class RelativeAccuracy(Metric):
    """
    Метрика accuracy для регрессии:
    Accuracy = 1 - mean(abs((pred - target) / target))
    Ограничивается 0 при относительной ошибке > 1.
    """

    def __init__(self, eps: float = 1e-12, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.eps = eps
        self.add_state("sum_relative_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError(
                f"`preds` и `target` должны иметь одинаковую форму, но получили {preds.shape} и {target.shape}")

        # избегаем деления на ноль
        rel_error = torch.abs((preds - target) / (target + self.eps))
        mean_error = rel_error.mean()

        self.sum_relative_error += mean_error
        self.total += 1

    def compute(self):
        avg_error = self.sum_relative_error / self.total
        return 1.0 - avg_error if avg_error < 1 else torch.tensor(0.0)



def MAE_error(output, target):
    error = torch.mean(torch.abs(output - target)).item()
    return 1 - error
# ─────────────────────────────────────────────────────────────────────────────
#                             BaseLMWithMetrics
# ─────────────────────────────────────────────────────────────────────────────
class BaseLMWithMetrics(BaseLModule):
    """
    Lightning‑модуль с поддержкой валидационных и тестовых метрик.

    Все параметры конструктора совпадают с `BaseLModule`,
    дополнительно можно передать `metrics` — либо `MetricCollection`,
    либо словарь `{name: metric}`.

    Параметры
    ----------
    model : nn.Module
        Модель для обучения.

    swap_xy : bool, default=False
        Если True — решается инверсная задача (Y → X).

    auto_decode : bool, default=True
        Если True — автоматически декодировать выходы в TouchstoneData на predict_step.

    codec : TouchstoneCodec, optional
        Кодек для декодирования предсказаний.

    loss_fn : Callable, optional
        Функция потерь. По умолчанию MSELoss().

    optimizer_cfg : dict, optional
        Конфигурация оптимизатора.

    scheduler_cfg : dict, optional
        Конфигурация планировщика.

    scaler_in : nn.Module, optional
        Скейлер для нормализации входа.

    scaler_out : nn.Module, optional
        Скейлер для нормализации выхода.

    metrics : MetricCollection | dict, optional
        Набор метрик для расчета на валидации и тесте.
    metrics : MetricCollection | dict, optional
        Набор метрик для расчета на валидации и тесте.
        *Внутри* модуля он клонируется дважды (`val_*/ test_*`).
    """

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        model: nn.Module,
        *,
        # ----- task mode ----------------------------------------------------
        swap_xy: bool = False,
        auto_decode: bool = True,
        codec: Optional[TouchstoneCodec] = None,
        # ----- optimisation -------------------------------------------------
        loss_fn: Optional[Callable] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
        # ----- scalers ------------------------------------------------------
        scaler_in: Optional[nn.Module] = None,
        scaler_out: Optional[nn.Module] = None,
        # ----- metrics ------------------------------------------------------
        metrics: Optional[MetricCollection | dict] = None,
    ):
        super().__init__(
            model=model,
            swap_xy=swap_xy,
            auto_decode=auto_decode,
            codec=codec,
            loss_fn=loss_fn,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            scaler_in=scaler_in,
            scaler_out=scaler_out,
        )

        # --- подготавливаем набор метрик ----------------------------------
        if metrics is None:
            metrics = {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "r2" : R2Score(),
                "acc": RelativeAccuracy()
            }
        if isinstance(metrics, dict):
            metrics = MetricCollection(metrics)

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    # ======================================================================
    #                               helpers
    # ======================================================================
    def _prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Приводит `y` к той же шкале, что использует loss и метрики.
        """
        if self.scaler_out is not None:
            return self.scaler_out(y)
        return y

    # ======================================================================
    #                        validation / test loop
    # ======================================================================
    def validation_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        y_t = self._prepare_targets(y)

        loss = self.loss_fn(preds, y_t)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # 💡 Flatten для совместимости с метриками
        preds_flat = preds.view(preds.size(0), -1)
        y_flat = y_t.view(y_t.size(0), -1)

        metric_dict = self.val_metrics(preds_flat, y_flat)
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        acc = MAE_error(preds, y_t)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        y_t = self._prepare_targets(y)

        loss = self.loss_fn(preds, y_t)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # 💡 Flatten для метрик
        preds_flat = preds.view(preds.size(0), -1)
        y_flat = y_t.view(y_t.size(0), -1)

        metric_dict = self.test_metrics(preds_flat, y_flat)
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    # ---------------------------------------------------------------- repr
    def extra_repr(self) -> str:  # pragma: no cover
        metric_names = ",".join(sorted(self.val_metrics.keys()))
        return f"{super().extra_repr()}, metrics=[{metric_names}]"
