# mwlab/lightning/base_lm.py
"""
BaseLModule
===========

Универсальный Lightning‑модуль MWLab c поддержкой:

* прямой (X→Y) **и** обратной (Y→X) задачи регрессии (`swap_xy`);
* любых скейлеров (`scaler_in` / `scaler_out`) – как `nn.Module`‑подмодулей;
* автоматического декодирования S‑параметров с помощью `TouchstoneCodec`;
* удобных **инференс‑методов**:
    * `predict_s(params_dict)  -> rf.Network`  — прямая задача (X→S);
    * `predict_x(rf.Network)  -> dict`         — обратная задача (S→X).

"""
from __future__ import annotations

import importlib
import pydoc
from typing import Callable, Optional, Tuple, Any, List, Mapping, Dict

import torch
import lightning as L
from torch import nn
import skrf as rf

from mwlab.codecs.touchstone_codec import TouchstoneCodec

__all__ = ["BaseLModule"]

# ─────────────────────────────────────────────────────────────────────────────
#                            helpers: class import
# ─────────────────────────────────────────────────────────────────────────────

def _locate_class(path: str):
    """Импортирует класс по строке ``pkg.sub:Cls`` или ``pkg.sub.Cls``."""
    if ":" in path:
        module_path, cls_name = path.split(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    obj = pydoc.locate(path)
    if obj is None:
        raise ImportError(f"Не удалось импортировать класс {path!r}")
    return obj

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
        swap_xy: bool = False,
        auto_decode: bool = True,
        # -------- вспомогательные модули -----------------------------------
        codec: Optional[TouchstoneCodec] = None,
        scaler_in: Optional[nn.Module] = None,
        scaler_out: Optional[nn.Module] = None,
        # -------- обучение --------------------------------------------------
        loss_fn: Optional[Callable] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()

        # --- основные компоненты ------------------------------------------
        self.model = model
        self.swap_xy = bool(swap_xy)
        self.auto_decode = bool(auto_decode)
        self.codec = codec
        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

        # замораживаем скейлеры (их веса не должны обучаться)
        for sc in (self.scaler_in, self.scaler_out):
            if sc is not None:
                for p in sc.parameters():  # type: ignore[attr-defined]
                    p.requires_grad_(False)

        self.loss_fn = loss_fn or nn.MSELoss()

        # --- сохраняем гиперпараметры -------------------------------------
        # (скейлеры и codec как объекты в hparams не кладем)
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
    def _split_batch(batch):
        """Поддерживаем форматы (x,y) и (x,y,meta)."""
        if len(batch) == 3:
            return batch  # type: ignore
        x, y = batch
        return x, y, None

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """X → Z (учитываем scaler_in, если есть)."""
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

    # ---------------------------------------------------------------- predict (batched через Trainer)
    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Логика batch‑predict, вызываемая Lightning‑ом.
            * swap_xy = False → возвращает TouchstoneData (если codec+meta и auto_decode=True)
            * swap_xy = True  → возвращает список словарей параметров или Tensor
        """
        x, _, meta = self._split_batch(batch)
        preds = self(x)

        # inverse‑scaling выхода (если есть)
        if self.scaler_out is not None:
            preds = self._apply_inverse(self.scaler_out, preds)

        # ---------------- ПРЯМАЯ задача (X→S) ----------------
        if not self.swap_xy:
            if self.codec is None or not self.auto_decode:
                return preds

            out: List[Any] = []
            for i in range(preds.size(0)):
                per_pred = preds[i]
                if meta is not None:
                    per_meta = meta[i]  # type: ignore[index]
                    out.append(self.codec.decode(per_pred, per_meta))  # TouchstoneData
                else:
                    out.append(self.codec.decode_s(per_pred))  # rf.Network
            return out

        # ---------------- ОБРАТНАЯ задача (Y→X) --------------
        if self.codec is None:
            return preds

        out: List[Dict[str, float]] = []
        for row in preds:
            out.append(self.codec.decode_x(row))

        return out

    # ───────────────────────────────────────────── optim & schedulers
    def _extract_cfg(self, cfg: dict, key: str = "name"):
        """Разбираем конфиг вида {"name": "Adam", "lr": 1e-3} → ("Adam", {...})."""
        name = str(cfg.get(key))
        params = {k: v for k, v in cfg.items() if k != key}
        return name, params

    def configure_optimizers(self):
        # optimizer
        opt_name, opt_params = self._extract_cfg(self.hparams.optimizer_cfg)
        optimizer_cls = getattr(torch.optim, opt_name, None)
        if optimizer_cls is None:
            raise ValueError(f"torch.optim has no optimizer '{opt_name}'")
        optimizer = optimizer_cls(self.parameters(), **opt_params)

        # scheduler (optional)
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
                "interval": "epoch",
                **({"monitor": "val_loss"} if is_plateau else {}),
            },
        }

    # ───────────────────────────────────────────── checkpoint helpers
    @staticmethod
    def _dump_scaler(scaler: nn.Module) -> Dict[str, Any]:
        return {
            "path": f"{scaler.__class__.__module__}:{scaler.__class__.__qualname__}",
            "kwargs": getattr(scaler, "_init_kwargs", {}),
            "state": scaler.state_dict(),
        }

    @staticmethod
    def _load_scaler(payload: Dict[str, Any]) -> nn.Module:
        cls = _locate_class(payload["path"])
        scaler = cls(**payload.get("kwargs", {}))
        scaler.load_state_dict(payload["state"])
        for p in scaler.parameters():  # type: ignore[attr-defined]
            p.requires_grad_(False)
        return scaler

    def state_dict(self, *args, **kwargs):  # noqa: D401
        state = super().state_dict(*args, **kwargs)
        # сохраняем TouchstoneCodec (он не nn.Module)
        if self.codec is not None:
            state["mw_codec"] = self.codec.dumps()
        if self.scaler_in is not None:
            state["mw_scaler_in"] = self._dump_scaler(self.scaler_in)
        if self.scaler_out is not None:
            state["mw_scaler_out"] = self._dump_scaler(self.scaler_out)

        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):  # noqa: D401
        # восстанавливаем codec (до вызова super, чтобы predict_step уже знал)
        if "mw_codec" in state_dict:
            self.codec = TouchstoneCodec.loads(state_dict.pop("mw_codec"))
        if "mw_scaler_in" in state_dict and self.scaler_in is None:
            self.scaler_in = self._load_scaler(state_dict.pop("mw_scaler_in"))
        if "mw_scaler_out" in state_dict and self.scaler_out is None:
            self.scaler_out = self._load_scaler(state_dict.pop("mw_scaler_out"))
        super().load_state_dict(state_dict, strict=strict)

    # =====================================================================
    #                           PUBLIC inference API
    # =====================================================================
    @torch.no_grad()
    def predict_s(self, params: Mapping[str, float]) -> rf.Network:
        """Прямая задача **X→S** для одного экземпляра.

        *Требует* `swap_xy=False` и наличия `self.codec`.
        """
        if self.swap_xy:
            raise RuntimeError("predict_s можно вызывать только при swap_xy=False")
        if self.codec is None:
            raise RuntimeError("predict_s требует, чтобы в модуле был codec")

        self.eval()
        x_t = self.codec.encode_x(params).to(self.device).unsqueeze(0)
        y_t = self(x_t)
        if self.scaler_out is not None:
            y_t = self._apply_inverse(self.scaler_out, y_t)
        return self.codec.decode_s(y_t[0])

    @torch.no_grad()
    def predict_x(self, net: rf.Network) -> Dict[str, float]:
        if not self.swap_xy:
            raise RuntimeError("predict_x доступен только при swap_xy=True")
        if self.codec is None:
            raise RuntimeError("predict_x требует codec")
        self.eval()
        y_t, _ = self.codec.encode_s(net)
        y_t = y_t.to(self.device).unsqueeze(0)
        x_pred = self(y_t)[0]
        if self.scaler_out is not None:
            x_pred = self._apply_inverse(self.scaler_out, x_pred)
        return self.codec.decode_x(x_pred)
    # ---------------------------------------------------------------- repr
    def extra_repr(self) -> str:  # noqa: D401
        task = "inverse (Y→X)" if self.swap_xy else "direct (X→Y)"
        codec_str = "yes" if self.codec else "—"
        return (
            f"task={task}, model={self.model.__class__.__name__}, "
            f"codec={codec_str}, auto_decode={self.auto_decode}"
        )

    __str__ = extra_repr
