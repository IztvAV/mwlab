# mwlab/lightning/base_lm.py
"""
BaseLModule
===========

Универсальный Lightning‑модуль MWLab, предназначенный для регрессии
*как прямой* (X → Y), *так и обратной* (Y → X) задачи.
Учтены следующие изменения:

Поддерживает:
* прямую и обратную регрессию (флаг **swap_xy**);
* любые скейлеры с методами `fit / forward / inverse`;
* автоматическое декодирование S‑параметров (`TouchstoneCodec.decode`)
  во время `predict_step`.
"""


from __future__ import annotations

import torch
import lightning as L
from torch import nn
from typing import Callable, Optional, Tuple, Any, List

from mwlab.codecs.touchstone_codec import TouchstoneCodec


# ─────────────────────────────────────────────────────────────────────────────
#                                  BaseLModule
# ─────────────────────────────────────────────────────────────────────────────
class BaseLModule(L.LightningModule):
    """
    Базовый класс LightningModule для MWLab.

    Параметры
    ----------
    model : nn.Module
        Модель для обучения.

    swap_xy : bool, default=False
        Если True — инверсная задача (Y → X вместо стандартной X → Y).

    auto_decode : bool, default=True
        Если True — автоматически применять TouchstoneCodec.decode()
        к выходам модели в predict_step.

    codec : TouchstoneCodec, optional
        Кодек для преобразования TouchstoneData ↔︎ тензоров (используется в predict_step).

    scaler_in : nn.Module, optional
        Скейлер для нормализации входных данных (например, StdScaler).

    scaler_out : nn.Module, optional
        Скейлер для нормализации выходных данных.

    loss_fn : Callable, optional
        Функция потерь. По умолчанию используется MSELoss().

    optimizer_cfg : dict, optional
        Конфигурация оптимизатора. Пример: {"name": "Adam", "lr": 1e-3}.

    scheduler_cfg : dict, optional
        Конфигурация планировщика learning rate. Пример: {"name": "StepLR", "step_size": 10}.
    """

    # ----------------------------------------------------------------- init
    def __init__(
        self,
        model: nn.Module,
        *,
        # -------- режим задачи ---------------------------------------------
        swap_xy: bool = False,  # True → инверсная задача (Y ➜ X)
        auto_decode: bool = True,  # декодировать ли Output → TouchstoneData
        # -------- вспомогательные модули -----------------------------------
        codec: Optional[TouchstoneCodec] = None,
        scaler_in: Optional[nn.Module] = None,
        scaler_out: Optional[nn.Module] = None,
        # -------- обучение --------------------------------------------------
        loss_fn: Optional[Callable] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
    ):
        super().__init__()
        # — основные компоненты
        self.model = model
        self.swap_xy = bool(swap_xy)
        self.auto_decode = bool(auto_decode)
        self.codec = codec

        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

        # замораживаем скейлеры (не обучаются)
        for sc in (self.scaler_in, self.scaler_out):
            if sc is not None:
                for p in sc.parameters():  # type: ignore[attr-defined]
                    p.requires_grad_(False)

        self.loss_fn = loss_fn or nn.MSELoss()

        # — сохраняем гиперпараметры (кроме больших объектов)
        self.save_hyperparameters(
            {
                "optimizer_cfg": optimizer_cfg or {"name": "Adam", "lr": 1e-3},
                "scheduler_cfg": scheduler_cfg,
                "swap_xy": self.swap_xy,
                "auto_decode": self.auto_decode,
            }
        )

    # ──────────────────────────────────────────────── internal helpers
    @staticmethod
    def _apply_inverse(scaler: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Инвертируем скейлер независимо от его API."""
        if hasattr(scaler, "inverse"):
            return scaler.inverse(x)
        if hasattr(scaler, "inverse_transform"):
            arr = scaler.inverse_transform(x.cpu().numpy())
            return torch.as_tensor(arr, device=x.device, dtype=x.dtype)
        raise AttributeError("Scaler has neither inverse nor inverse_transform")

    @staticmethod
    def _split_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Поддерживаем два формата:
            (x, y)            – без meta
            (x, y, meta)      – с meta
        """
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        return x, y, meta

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """X → Z (с учетом scaler_in)."""
        if self.scaler_in is not None:
            x = self.scaler_in(x)
        return self.model(x)

    # ───────────────────────────────────────────── shared step (train/val/test)
    def _shared_step(self, batch):
        x, y, _ = self._split_batch(batch)
        preds = self(x)

        # целевое значение всегда приводим к той же шкале, что и модель
        if self.scaler_out is not None:
            y = self.scaler_out(y)

        loss = self.loss_fn(preds, y)
        return loss

    # ---------------------------------------------------------------- train
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ---------------------------------------------------------------- val
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    # ---------------------------------------------------------------- test
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        * swap_xy = False → возвращает TouchstoneData (если codec+meta и auto_decode=True)
        * swap_xy = True  → возвращает список словарей параметров или Tensor
        """
        x, _, meta = self._split_batch(batch)
        preds = self(x)

        # 1) inverse‑scaling выхода (если есть)
        if self.scaler_out is not None:
            preds = self._apply_inverse(self.scaler_out, preds)

        # 2) формируем вывод в зависимости от постановки
        if self.swap_xy:
            if self.codec is None:
                return preds  # fallback: просто Tensor
            # формируем по‑экземплярно
            out: List[dict] = []
            for row in preds:
                out.append(
                    {k: float(v) for k, v in zip(self.codec.x_keys, row)}
                )
            return out
        else:
            if self.auto_decode and self.codec is not None and meta is not None:
                return self.codec.decode(preds, meta)
            return preds

    # ───────────────────────────────────────────── optim & schedulers
    def _extract_cfg(self, cfg: dict, key: str = "name"):
        """Разбираем конфиг вида {"name": "Adam", "lr": 1e-3} → ("Adam", {...})."""
        name = str(cfg.get(key))
        params = {k: v for k, v in cfg.items() if k != key}
        return name, params

    def configure_optimizers(self):
        # --- optimizer -----------------------------------------------------
        opt_name, opt_params = self._extract_cfg(self.hparams.optimizer_cfg)
        optimizer_cls = getattr(torch.optim, opt_name, None)
        if optimizer_cls is None:
            raise ValueError(f"torch.optim has no optimizer '{opt_name}'")
        optimizer = optimizer_cls(self.parameters(), **opt_params)

        # --- scheduler (optional) -----------------------------------------
        if self.hparams.scheduler_cfg is None:
            return optimizer

        sch_name, sch_params = self._extract_cfg(self.hparams.scheduler_cfg)
        scheduler_cls = getattr(torch.optim.lr_scheduler, sch_name, None)
        if scheduler_cls is None:
            raise ValueError(f"torch.optim.lr_scheduler has no scheduler '{sch_name}'")

        scheduler = scheduler_cls(optimizer, **sch_params)
        is_plateau = sch_name.lower().endswith("plateau")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # можно поменять на "step", если нужно
                **({"monitor": "val_loss"} if is_plateau else {}),
            },
        }

    # ───────────────────────────────────────────── checkpoint helpers
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # сохраняем скейлеры
        state["mw_scalers"] = {}
        if self.scaler_in is not None:
            state["mw_scalers"]["in"] = self.scaler_in.state_dict()
        if self.scaler_out is not None:
            state["mw_scalers"]["out"] = self.scaler_out.state_dict()
        # сохраняем TouchstoneCodec в виде bytes
        if self.codec is not None:
            state["mw_codec"] = self.codec.dumps()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        scalers_state = state_dict.pop("mw_scalers", {})
        if "in" in scalers_state and self.scaler_in is not None:
            self.scaler_in.load_state_dict(scalers_state["in"])
        if "out" in scalers_state and self.scaler_out is not None:
            self.scaler_out.load_state_dict(scalers_state["out"])
        if "mw_codec" in state_dict:
            self.codec = TouchstoneCodec.loads(state_dict.pop("mw_codec"))
        super().load_state_dict(state_dict, strict=strict)

    # ---------------------------------------------------------------- repr
    def extra_repr(self) -> str:  # pragma: no cover
        task = "inverse (Y→X)" if self.swap_xy else "direct (X→Y)"
        codec_str = "yes" if self.codec else "—"
        return (
            f"task={task}, model={self.model.__class__.__name__}, "
            f"codec={codec_str}, auto_decode={self.auto_decode}"
        )
