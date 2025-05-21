# mwlab/lightning/base_lm_with_metrics.py
"""
BaseLMWithMetrics
=================

Расширяет `BaseLModule`, добавляя накопление и логирование произвольных
метрик во время **validate / test**.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score
from typing import Optional, Callable, Any

from mwlab.lightning.base_lm import BaseLModule
from mwlab.codecs.touchstone_codec import TouchstoneCodec


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
                "r2" : R2Score()
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
        #preds_flat = preds.view(preds.size(0), -1)
        #y_flat = y_t.view(y_t.size(0), -1)
        #metric_dict = self.val_metrics(preds_flat, y_flat)

        # 👉 передаем **не** сплющенные тензоры
        metric_dict = self.val_metrics(preds, y_t)

        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        y_t = self._prepare_targets(y)

        loss = self.loss_fn(preds, y_t)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # 💡 Flatten для метрик
        #preds_flat = preds.view(preds.size(0), -1)
        #y_flat = y_t.view(y_t.size(0), -1)
        #metric_dict = self.test_metrics(preds_flat, y_flat)

        # 👉 передаем **не** сплющенные тензоры
        metric_dict = self.val_metrics(preds, y_t)

        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    # ---------------------------------------------------------------- repr
    def extra_repr(self) -> str:  # pragma: no cover
        metric_names = ",".join(sorted(self.val_metrics.keys()))
        return f"{super().extra_repr()}, metrics=[{metric_names}]"
