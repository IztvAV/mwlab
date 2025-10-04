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
import numpy as np

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
        #from importlib import import_module

        payload = torch.load(path, map_location="cpu")
        ModCls = _locate_class(payload["class"])

        # создаем экземпляр без побочных эффектов
        lm = ModCls.__new__(ModCls)  # type: ignore[call-arg]
        try:
            ModCls.__init__(lm, model=nn.Identity())
        except TypeError:
            # запасной безопасный init
            ModCls.__init__(lm, model=nn.Identity(), swap_xy=False, auto_decode=False)

        lm.load_state_dict(payload["state_dict"])
        # явная проверка соответствия «прямая»/«обратная»
        if getattr(lm, "swap_xy", False):
            raise ValueError("Checkpoint contains swap_xy=True; use InverseNNSurrogate.load(...)")

        return cls(pl_module=lm)

    # ---------------------------------------------------------------- batch-API
    @torch.no_grad()
    def batch_predict(
            self,
            xs: Sequence[Mapping[str, float]],
            *,
            return_std: bool = False,
    ):
        if return_std:
            raise NotImplementedError("NN surrogate has no σ.")
        # 1) быстрый путь, если модуль его реализует
        if hasattr(self.model, "predict_s_batch"):
            return self.model.predict_s_batch(xs)
        # 2) иначе — по одному (корректно, но медленнее)
        return [self.predict(x) for x in xs]

    # ----------------------------------------------- оптимизированный passes_spec
    def passes_spec(self, xs, spec):
        """
        Быстрая проверка спецификации для батча X-параметров.

        • Если `xs` — одиночный dict → вызываем базовый метод
          (он пройдёт через обычный `predict`).

        • Для батча:
          1. Кодируем X-вектор каждого dict через `codec.encode_x`.
          2. Склеиваем их в один тензор (B,Dx) и прогоняем сеть
             **одним** forward-проходом (GPU-friendly).
          3. Приводим результат к NumPy и пытаемся отдать его в
             `spec.fast_is_ok` — это самый быстрый путь.
          4. Если `fast_is_ok` нет, декодируем каждую строку в
             `rf.Network` через `codec.decode_s` и вызываем `is_ok`.
        """
        # -------- utils ----------------------------------------------
        def _device_of(module: torch.nn.Module) -> torch.device:
            try:
                return next(module.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        # -------- одиночная точка ------------------------------------
        if isinstance(xs, Mapping):
            return super().passes_spec(xs, spec)

        codec = getattr(self.model, "codec", None)
        if codec is None:                         # safety-fallback
            return super().passes_spec(xs, spec)

        # -------- векторный encode X ---------------------------------
        with torch.no_grad():
            dev = getattr(self.model, "device", _device_of(self.model))
            X = torch.stack([codec.encode_x(p) for p in xs]) \
                    .to(dev)                     # (B,Dx)
            Y = self.model(X)                    # (B,C,F)  или (B,*)
            if self.model.scaler_out is not None:
                Y = self.model._apply_inverse(self.model.scaler_out, Y)
            Y_np = Y.cpu().numpy()               # → NumPy

        # -------- fast path через Specification ----------------------
        if hasattr(spec, "fast_is_ok"):
            try:
                return spec.fast_is_ok(Y_np)     # (B,) bool
            except Exception:
                # если векторная реализация упала — откатываемся
                pass

        # -------- медленный fallback ---------------------------------
        return np.fromiter(
            (
                spec.is_ok(
                    codec.decode_s(torch.as_tensor(row, dtype=torch.float32))
                )
                for row in Y_np
            ),
            dtype = bool,
        )

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

