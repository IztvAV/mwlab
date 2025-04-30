# mwlab/lightning/base_lm.py

"""
Базовый LightningModule MWLab.

Функциональность
----------------
✓ Модель можно обучать как на «прямую» (X ➜ Y), так и на «обратную» (Y ➜ X) задачу.
✓ Поддерживаются входные и выходные скейлеры (StdScaler, MinMaxScaler и др.).
✓ TouchstoneCodec (если передан) автоматически сохраняется в checkpoint
  и доступен на инференсе.
✓ Флаг swap_xy хранится в checkpoint-е — при load_from_checkpoint()
  модуль «помнит», была ли модель обучена в обратной постановке.
✓ predict_step по-умолчанию:
    • прямая задача  →  TouchstoneData (auto_decode=True)
    • обратная       →  словарь параметров
"""

import torch
import lightning as L
from torch import nn
from typing import Callable, Optional, Tuple, Any, Mapping

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
        Конфигурация планировщика learning rate. Пример: {"name": "StepLR", "step_size": 10, "gamma": 0.1}.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        # ── режим задачи ───────────────────────────────────────────────────
        swap_xy: bool = False,        # True → инверсная задача (Y ➜ X)
        auto_decode: bool = True,     # декодировать ли Output → TouchstoneData
        # ── вспомогательные модули ─────────────────────────────────────────
        codec: Optional[TouchstoneCodec] = None,
        scaler_in: Optional[nn.Module] = None,
        scaler_out: Optional[nn.Module] = None,
        # ── обучение ──────────────────────────────────────────────────────
        loss_fn: Optional[Callable] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
    ):
        super().__init__()
        # ―― основные компоненты ―――――――――――――――――――――――――――――――――――――――――――
        self.model = model
        self.swap_xy = bool(swap_xy)
        self.auto_decode = bool(auto_decode)
        self.codec = codec

        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

        # замораживаем скейлеры (не обучаются)
        for sc in (self.scaler_in, self.scaler_out):
            if sc is not None:
                for p in sc.parameters():        # type: ignore[arg-type]
                    p.requires_grad_(False)

        self.loss_fn = loss_fn or nn.MSELoss()

        # ―― сохраняем гиперпараметры ―――――――――――――――――――――――――――――――――――
        self.save_hyperparameters(
            {
                "optimizer_cfg": optimizer_cfg or {"name": "Adam", "lr": 1e-3},
                "scheduler_cfg": scheduler_cfg,
                "swap_xy": self.swap_xy,
                "auto_decode": self.auto_decode,
            }
        )

    # ──────────────────────────────────────────────────────────────────── utils
    @staticmethod
    def _apply_inverse(scaler: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Инвертируем скейлер (StdScaler.inverse или inverse_transform NumPy)."""
        if hasattr(scaler, "inverse"):
            return scaler.inverse(x)
        if hasattr(scaler, "inverse_transform"):
            return torch.as_tensor(
                scaler.inverse_transform(x.cpu().numpy()), device=x.device
            )
        raise AttributeError("Scaler has neither inverse nor inverse_transform")

    @staticmethod
    def _split_meta(meta: Mapping[str, Any], idx: int) -> dict:
        """Из batched-meta получить компонент №idx."""
        out = {}
        for k, v in meta.items():
            if torch.is_tensor(v) and v.dim() > 0:
                out[k] = v[idx]
            elif isinstance(v, (list, tuple)):
                out[k] = v[idx]
            else:
                out[k] = v
        return out

    @staticmethod
    def _unpack_batch(batch):
        """Поддерживает форматы (x,y) и (x,y,meta)."""
        return batch if len(batch) == 3 else (*batch, None)

    # ───────────────────────────────────────────────────────────────── forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """X → Z (скейлер входа применяется, если есть)."""
        if self.scaler_in is not None:
            x = self.scaler_in(x)
        return self.model(x)

    # ───────────────────────────────────────────────────────────── shared step
    def _shared_step(self, batch):
        """Общий шаг для train/val/test — считает loss."""
        x, y, _ = self._unpack_batch(batch)
        preds = self(x)

        # если предсказываем Y, то надо нормализовать target
        if not self.swap_xy and self.scaler_out is not None:
            y = self.scaler_out(y)

        # если предсказываем X (swap_xy=True), скейлер_out не нужен
        loss = self.loss_fn(preds, y)
        return loss

    # ---------------------------------------------------------------- training
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ---------------------------------------------------------------- validate
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0))

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0))

    # ---------------------------------------------------------------- predict
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, meta = self._unpack_batch(batch)
        preds = self(x)

        # -------- прямая модель (X→Y) --------------------------------------
        if not self.swap_xy:
            if self.scaler_out is not None:
                preds = self._apply_inverse(self.scaler_out, preds)

            if not (self.auto_decode and self.codec and meta):
                return preds

            # batched decode
            if preds.dim() == 2:   # (C,F)
                return self.codec.decode(preds, meta)

            out = [
                self.codec.decode(preds[i], self._split_meta(meta, i))
                for i in range(preds.size(0))
            ]
            return out[0] if len(out) == 1 else out

        # -------- обратная модель (Y→X) -----------------------------------
        if not self.auto_decode or self.codec is None:
            return preds          # raw tensor

        keys = self.codec.x_keys
        if preds.dim() == 1:
            return {k: float(v) for k, v in zip(keys, preds)}

        out = [{k: float(v) for k, v in zip(keys, row)} for row in preds]
        return out[0] if preds.size(0) == 1 else out

    # ─────────────────────────────────────────────────────────— optim & sched
    def _extract_cfg(self, cfg: dict, key: str = "name"):
        """Разбираем конфиг вида {"name": "Adam", "lr": 1e-3} → ("Adam", {...})."""
        name = cfg.get(key)
        params = {k: v for k, v in cfg.items() if k != key}
        return name, params

    def configure_optimizers(self):
        # ── optimizer ───────────────────────────────────────────────────────
        opt_name, opt_params = self._extract_cfg(self.hparams.optimizer_cfg)
        optimizer_cls = getattr(torch.optim, opt_name, None)
        if optimizer_cls is None:
            raise ValueError(f"torch.optim has no optimizer '{opt_name}'")
        optimizer = optimizer_cls(self.parameters(), **opt_params)

        # ── scheduler (опционально) ────────────────────────────────────────
        if self.hparams.scheduler_cfg is None:
            return optimizer

        sch_name, sch_params = self._extract_cfg(self.hparams.scheduler_cfg)
        scheduler_cls = getattr(torch.optim.lr_scheduler, sch_name, None)
        if scheduler_cls is None:
            raise ValueError(
                f"torch.optim.lr_scheduler has no scheduler '{sch_name}'"
            )
        scheduler = scheduler_cls(optimizer, **sch_params)

        sched_dict = {"optimizer": optimizer, "lr_scheduler": scheduler}
        if sch_name.lower().endswith("plateau"):
            sched_dict["monitor"] = "val_loss"
        return sched_dict

    # ───────────────────────────────────────────────────── checkpoint helpers
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # скейлеры
        state["mw_scalers"] = {}
        if self.scaler_in is not None:
            state["mw_scalers"]["in"] = self.scaler_in.state_dict()
        if self.scaler_out is not None:
            state["mw_scalers"]["out"] = self.scaler_out.state_dict()
        # TouchstoneCodec
        if self.codec is not None:
            state["mw_codec"] = self.codec.dumps()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # скейлеры
        scalers_state = state_dict.pop("mw_scalers", {})
        if "in" in scalers_state and self.scaler_in is not None:
            self.scaler_in.load_state_dict(scalers_state["in"])
        if "out" in scalers_state and self.scaler_out is not None:
            self.scaler_out.load_state_dict(scalers_state["out"])
        # TouchstoneCodec
        if "mw_codec" in state_dict:
            self.codec = TouchstoneCodec.loads(state_dict.pop("mw_codec"))
        super().load_state_dict(state_dict, strict=strict)

    # ───────────────────────────────────────────────────────────── extra repr
    def extra_repr(self) -> str:
        codec_str = "yes" if self.codec is not None else "—"
        task = "inverse (Y→X)" if self.swap_xy else "direct (X→Y)"
        return (
            f"task={task}, model={self.model.__class__.__name__}, "
            f"codec={codec_str}, auto_decode={self.auto_decode}"
        )

