# mwlab/lightning/base_lm_with_metrics.py

"""
Расширение BaseLModule с аккумуляцией метрик на валидации и тесте.

Функциональность
----------------
✓ Поддержка произвольных метрик (torchmetrics.MetricCollection);
✓ Корректная работа со скейлерами (scaler_in / scaler_out);
✓ Поддержка прямой и обратной задачи (swap_xy);
✓ Формат batch-ей:
    (x, y)            – без meta
    (x, y, meta)      – с meta
"""

import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError
from typing import Optional, Callable, Any, Tuple

from mwlab.lightning.base_lm import BaseLModule
from mwlab.codecs.touchstone_codec import TouchstoneCodec


# ─────────────────────────────────────────────────────────────────────────────
#                             BaseLMWithMetrics
# ─────────────────────────────────────────────────────────────────────────────

class BaseLMWithMetrics(BaseLModule):
    """
    Базовый LightningModule + сбор метрик на валидации и тесте.

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
    """

    def __init__(
            self,
            model: nn.Module,
            *,
            # ------------- режим задачи ----------------------------------------
            swap_xy: bool = False,
            auto_decode: bool = True,
            codec=None,
            # ------------- обучение -------------------------------------------
            loss_fn: Optional[Callable] = None,
            optimizer_cfg: Optional[dict] = None,
            scheduler_cfg: Optional[dict] = None,
            # ------------- скейлеры -------------------------------------------
            scaler_in: Optional[nn.Module] = None,
            scaler_out: Optional[nn.Module] = None,
            # ------------- метрики --------------------------------------------
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

        # ── Метрики ────────────────────────────────────────────────────────
        if metrics is None:
            metrics = {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
            }
        if isinstance(metrics, dict):
            metrics = MetricCollection(metrics)

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    # ======================================================================
    #                             INTERNAL HELPERS
    # ======================================================================

    def _split_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Поддержка batch-ей с meta (3 элемента) и без (2 элемента)."""
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        return x, y, meta

    def _prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Приводим targets к той же шкале, что и использовалась в loss:
            • Прямая задача  (swap_xy=False)  → scaler_out(y)
            • Обратная       (swap_xy=True)   → *без* скейлера
        """
        if not self.swap_xy and self.scaler_out is not None:
            return self.scaler_out(y)
        return y

    # ======================================================================
    #                            validation / test
    # ======================================================================

    def validation_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)

        preds = self(x)
        y_t = self._prepare_targets(y)

        loss = self.loss_fn(preds, y_t)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        metrics = self.val_metrics(preds, y_t)
        self.log_dict(metrics, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)

        preds = self(x)
        y_t = self._prepare_targets(y)

        loss = self.loss_fn(preds, y_t)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        metrics = self.test_metrics(preds, y_t)
        self.log_dict(metrics, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        return loss

