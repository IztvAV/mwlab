#mwlab/opt/surrogates/nn.py
"""
NNSurrogate – обертка над уже **обученным** Lightning-модулем MWLab
(`BaseLModule` / `BaseLMWithMetrics`).

* Прямая задача (swap_xy=False) – использует `pl_module.predict_s()`
  и возвращает **rf.Network**.
* Обратная (swap_xy=True)       – вызывает `pl_module.predict_x()`
  и отдаёт `dict` параметров.

Преимущества:
* Никаких повторных fit; готова сразу после `trainer.fit(...)`.
* Поддержка GPU – управляется самим Lightning-модулем.
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import skrf as rf

from mwlab.lightning.base_lm import _locate_class
from ..design.space import DesignSpace
from .base import BaseSurrogate
from ..surrogates import register   # регистрируем alias


# ────────────────────────────────────────────────────────────────────────────
#                               NNSurrogate
# ────────────────────────────────────────────────────────────────────────────

@register("nn", "pytorch")
class NNSurrogate(BaseSurrogate):
    """Обертка над `BaseLModule`."""

    def __init__(
        self,
        pl_module: "BaseLModule",
        *,
        design_space: DesignSpace | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Parameters
        ----------
        pl_module : BaseLModule
            Уже обученный Lightning-модуль.
        design_space : DesignSpace | None
            Храним ссылку «для справки» (нормализация здесь не нужна,
            тк codec внутри модуля).
        device : torch.device | str | None
            Переместить модель на cpu/cuda:0 …  (None → оставить как есть).
        """
        from mwlab.lightning.base_lm import BaseLModule

        if not isinstance(pl_module, BaseLModule):
            raise TypeError("NNSurrogate expects a BaseLModule")

        if pl_module.swap_xy:
            raise ValueError(
                "NNSurrogate предназначен только для прямой модели "
                "(swap_xy=False).  Для обратной задачи используйте "
                "InverseNNSurrogate или оптимизируйте по прямому surrogate."
            )

        self.model = pl_module.eval()
        if device is not None:
            self.model.to(device)
        self.design_space = design_space
        self._class_path = (
            f"{pl_module.__class__.__module__}:{pl_module.__class__.__qualname__}"
        )

    # ─────────────────────────────── predict ────────────────────────────
    # прямой прогноз
    @torch.no_grad()
    def predict(self, x: Mapping[str, float], *, return_std: bool = False):
        if return_std:
            raise NotImplementedError("NN surrogate has no σ; use ensemble drop-out.")
        return self.model.predict_s(x)  # → rf.Network

    # ─────────────────────────────── save / load ────────────────────────
    def save(self, path: str | Path):
        path = Path(path)
        payload = {
            "class": self._class_path,
            "state_dict": self.model.state_dict(),
            "swap_xy": self.model.swap_xy,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "NNSurrogate":
        from importlib import import_module

        payload = torch.load(path, map_location="cpu")

        #module_path, cls_name = payload["class"].split(":")
        #ModCls = getattr(import_module(module_path), cls_name)
        #lm = ModCls.__new__(ModCls)     # type: ignore[call-arg]
        #lm.__init__(model=None)         # dummy init; потом state_dict

        ModCls = _locate_class(payload["class"])
        # создаем экземпляр без побочных эффектов
        lm = ModCls.__new__(ModCls)  # type: ignore[call-arg]
        ModCls.__init__(lm, model=nn.Identity())  # минимальный valid-init

        lm.load_state_dict(payload["state_dict"])
        return cls(pl_module=lm)

    # ---------------------------------------------------------------- batch-API
    def batch_predict(
            self,
            xs: Sequence[Mapping[str, float]],
            *,
            return_std: bool = False,
    ):
        if return_std:
            raise NotImplementedError("NN surrogate не возвращает дисперсию.")

        codec = getattr(self.model, "codec", None)
        if codec is None:  # fallback на построчный режим
            return [self.predict(x) for x in xs]

        with torch.no_grad():
            X = torch.stack([codec.encode_x(p) for p in xs]).to(self.model.device)
            preds = self.model(X)
            if self.model.scaler_out is not None:
                preds = self.model._apply_inverse(self.model.scaler_out, preds)
            return [codec.decode_s(row) for row in preds]

    # -------------------------------------------------------------------
    def __repr__(self):  # pragma: no cover
        return f"NNSurrogate(model={self.model.__class__.__name__}, swap_xy={self.model.swap_xy})"



# ────────────────────────────────────────────────────────────────────────────
#                            InverseNNSurrogate
# ────────────────────────────────────────────────────────────────────────────

@register("inv_nn", "inverse_nn")
class InverseNNSurrogate(BaseSurrogate):
    """Обертка над BaseLModule(**swap_xy=True**) – X ← S."""

    def __init__(self, pl_module: "BaseLModule", *, device=None):
        if not pl_module.swap_xy:
            raise ValueError("InverseNNSurrogate ждёт модель с swap_xy=True")
        self.model = pl_module.eval()
        if device is not None:
            self.model.to(device)

    def predict(self, net: rf.Network, *, return_std=False):
        if return_std:
            raise NotImplementedError
        return self.model.predict_x(net)     # → dict
