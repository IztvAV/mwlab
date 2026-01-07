# mwlab/opt/surrogates/nn.py
"""
mwlab.opt.surrogates.nn
======================

NNSurrogate – обёртка над уже **обученным** Lightning-модулем MWLab (`BaseLModule`).

Назначение
----------
Этот surrogate реализует прямую модель X → S:
- на вход принимает точку проектных параметров `x` (dict/Mapping),
- на выход возвращает предсказанные S-параметры в формате **skrf.Network**
  (или NetworkLike-совместимый объект, если ваш BaseLModule так делает).

Ключевые особенности
--------------------
1) Никакого fit: surrogate готов сразу после обучения Lightning-модуля.
2) Поддержка GPU определяется самим Lightning-модулем и `device` в конструкторе.
3) Оптимизированный `passes_spec`:
   - для батча параметров делает один forward-pass (GPU-friendly),
   - при наличии `Specification.fast_is_ok(...)` пытается использовать его,
   - иначе корректно падает в безопасный (но более медленный) путь:
     decode_s -> spec.is_ok(net).

Важно про архитектуру (variant B)
--------------------------------
В варианте B registry вынесен в `mwlab.opt.surrogates.registry`, поэтому:
- регистрируем alias через `from .registry import register`,
- НЕ импортируем `mwlab.opt.surrogates` как пакет (избегаем циклов и лишнего lazy).

Также в этом файле определён InverseNNSurrogate (X ← S).
Его удобнее держать рядом, но он **не предназначен** для целей/оценки спецификаций
в стиле objectives (Penalty/Yield), т.к. там ожидается X→Network.
Поэтому InverseNNSurrogate переопределяет `passes_spec` и выдаёт понятную ошибку.

Требования к BaseLModule
------------------------
Ожидается, что `pl_module` имеет:
- `swap_xy: bool` (False для прямой модели, True для обратной),
- `predict_s(x: Mapping[str, Any]) -> rf.Network` для прямой модели,
- опционально `predict_s_batch(xs: Sequence[Mapping[str, Any]]) -> list[rf.Network]`
  для быстрого батча,
- опционально `codec` со средствами:
    - `encode_x(x_dict) -> torch.Tensor` (Dx,)
    - `decode_s(y_tensor) -> rf.Network` (или NetworkLike)

и что `pl_module(X)` выполняет forward по закодированному X.

Примечание по типам параметров
------------------------------
Используем `Mapping[str, Any]`, потому что параметры могут быть:
- float/int,
- строки (categorical),
- np scalar types и т.п.
Конкретные ограничения должен накладывать codec внутри Lightning-модуля.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import skrf as rf

from mwlab.lightning.base_lm import _locate_class
from ..design.space import DesignSpace
from .base import BaseSurrogate
from .registry import register  # ВАЖНО: variant B — регистрируем через registry, не через пакет


# ────────────────────────────────────────────────────────────────────────────
#                               NNSurrogate
# ────────────────────────────────────────────────────────────────────────────

@register("nn", "pytorch")
class NNSurrogate(BaseSurrogate):
    """
    Обёртка над обученным `BaseLModule` для прямой модели (swap_xy=False).

    Parameters
    ----------
    pl_module : BaseLModule
        Уже обученный Lightning-модуль.
    design_space : DesignSpace | None
        Храним ссылку «для справки» (нормализация обычно внутри codec модуля).
    device : torch.device | str | None
        Если задан — перемещает модель на указанное устройство.
    """

    supports_uncertainty: bool = False

    def __init__(
        self,
        pl_module: "BaseLModule",
        *,
        design_space: DesignSpace | None = None,
        device: str | torch.device | None = None,
    ):
        # Импортируем BaseLModule внутри, чтобы не создавать тяжёлых зависимостей при импорте пакета.
        from mwlab.lightning.base_lm import BaseLModule

        if not isinstance(pl_module, BaseLModule):
            raise TypeError("NNSurrogate expects a BaseLModule instance")

        # Этот surrogate — только для прямой модели.
        if getattr(pl_module, "swap_xy", False):
            raise ValueError(
                "NNSurrogate предназначен только для прямой модели (swap_xy=False). "
                "Для обратной задачи используйте InverseNNSurrogate."
            )

        self.model = pl_module.eval()

        # Если пользователь явно просит — переносим модель на устройство.
        if device is not None:
            self.model.to(device)

        self.design_space = design_space

        # Для save/load сохраняем путь до класса Lightning-модуля.
        self._class_path = f"{pl_module.__class__.__module__}:{pl_module.__class__.__qualname__}"

    # ─────────────────────────────── predict ────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        x: Mapping[str, Any],
        *,
        return_std: bool = False,
    ):
        """
        Точечный прогноз X → Network.

        Важно:
        - return_std здесь не поддерживается (нет модели неопределённости).
        """
        if return_std:
            raise NotImplementedError(
                "NNSurrogate не поддерживает return_std. "
                "Если нужна неопределённость — используйте ансамбль/MC-dropout на уровне модели."
            )

        fn = getattr(self.model, "predict_s", None)
        if not callable(fn):
            raise AttributeError("Lightning-модуль не реализует метод predict_s(x) для прямого прогноза X→S")
        return fn(x)  # ожидаем rf.Network (или NetworkLike-совместимый объект)

    # ───────────────────────────── batch_predict ────────────────────────────
    @torch.no_grad()
    def batch_predict(
        self,
        xs: Sequence[Mapping[str, Any]],
        *,
        return_std: bool = False,
    ):
        """
        Батчевый прогноз.

        Реализовано так:
        1) Если Lightning-модуль имеет `predict_s_batch` — используем его.
        2) Иначе безопасный fallback через последовательные `predict`.
        """
        if return_std:
            raise NotImplementedError("NNSurrogate не поддерживает return_std.")

        if not xs:
            return []

        fn = getattr(self.model, "predict_s_batch", None)
        if callable(fn):
            return fn(xs)

        return [self.predict(x) for x in xs]

    # ───────────────────────────── save / load ──────────────────────────────
    def save(self, path: str | Path):
        """
        Сохранение surrogate.

        Сохраняем:
        - class path Lightning-модуля (module:qualname),
        - state_dict,
        - swap_xy (на всякий случай для диагностики).
        """
        path = Path(path)
        payload = {
            "class": self._class_path,
            "state_dict": self.model.state_dict(),
            "swap_xy": bool(getattr(self.model, "swap_xy", False)),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "NNSurrogate":
        """
        Загрузка surrogate из файла, сохранённого методом save().

        Важно:
        - создаём экземпляр Lightning-модуля максимально «безопасно»,
          чтобы не требовать полного набора init-параметров.
        - затем грузим state_dict.
        """
        payload = torch.load(path, map_location="cpu")
        ModCls = _locate_class(payload["class"])

        # Создаём объект без побочных эффектов конструктора.
        lm = ModCls.__new__(ModCls)  # type: ignore[call-arg]

        # Пытаемся вызвать __init__ в максимально совместимых вариантах.
        # Это типичный компромисс для восстановления LightningModule без полного конфига.
        try:
            ModCls.__init__(lm, model=nn.Identity())
        except TypeError:
            # Запасной путь для модулей с другими сигнатурами.
            ModCls.__init__(lm, model=nn.Identity(), swap_xy=False, auto_decode=False)

        lm.load_state_dict(payload["state_dict"])

        # Явная проверка соответствия: NNSurrogate только для swap_xy=False.
        if bool(getattr(lm, "swap_xy", False)):
            raise ValueError(
                "Checkpoint содержит swap_xy=True. "
                "Это обратная модель; используйте InverseNNSurrogate.load(...) или другой загрузчик."
            )

        return cls(pl_module=lm)

    # ───────────────────── optimized passes_spec for NN ──────────────────────
    def passes_spec(self, xs, spec):
        """
        Быстрая проверка спецификации для батча параметров X.

        Поддерживаемые случаи
        ---------------------
        1) xs — одиночный dict:
           -> безопасный путь через BaseSurrogate.passes_spec (predict -> spec.is_ok)

        2) xs — batch (Sequence[Mapping]):
           - если у модели нет codec -> fallback на BaseSurrogate.passes_spec
           - иначе:
              a) encode_x для каждого dict -> тензор X (B,Dx)
              b) forward модели -> Y (B, ...)
              c) применить inverse scaling (если есть scaler_out)
              d) если spec.fast_is_ok существует -> попытаться вернуть (B,) bool
              e) иначе decode_s для каждой строки -> spec.is_ok(net)

        Замечания по устойчивости
        -------------------------
        - Любые падения fast-path приводят к fallback без разрушения процесса.
        - Возвращаем np.ndarray[bool] для батча.
        """
        # -------- одиночная точка -------------------------------------------
        if isinstance(xs, Mapping):
            return super().passes_spec(xs, spec)

        # На пустом батче возвращаем пустую маску.
        if xs is None:
            return np.zeros((0,), dtype=bool)
        try:
            B = len(xs)
        except Exception:
            # Если это не Sequence — fallback.
            return super().passes_spec(xs, spec)

        if B == 0:
            return np.zeros((0,), dtype=bool)

        codec = getattr(self.model, "codec", None)
        if codec is None:
            # Без codec не можем сделать векторизацию X→Y.
            return super().passes_spec(xs, spec)

        # -------- utils: определить устройство модели ------------------------
        def _device_of(module: torch.nn.Module) -> torch.device:
            try:
                return next(module.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        # -------- forward на батче ------------------------------------------
        with torch.no_grad():
            dev = getattr(self.model, "device", _device_of(self.model))

            # encode_x -> (Dx,) для каждой точки; stack -> (B,Dx)
            X = torch.stack([codec.encode_x(p) for p in xs]).to(dev)

            # forward: ожидаем (B, ...) (часто (B,C,F))
            Y = self.model(X)

            # если модель хранит scaler_out и метод обратного преобразования — применяем
            scaler_out = getattr(self.model, "scaler_out", None)
            fn_inv = getattr(self.model, "_apply_inverse", None)
            if scaler_out is not None and callable(fn_inv):
                Y = fn_inv(scaler_out, Y)

            Y_np = Y.detach().cpu().numpy()

        # -------- fast path: spec.fast_is_ok --------------------------------
        fn_fast = getattr(spec, "fast_is_ok", None)
        if callable(fn_fast):
            try:
                m = fn_fast(Y_np)
                m = np.asarray(m, dtype=bool)
                if m.shape == (B,):
                    return m
                # если форма неожиданная — не рискуем, fallback ниже
            except Exception:
                # fast_is_ok мог упасть — не мешаем оптимизации
                pass

        # -------- fallback: decode_s + spec.is_ok ----------------------------
        # Здесь предполагаем, что codec.decode_s умеет принимать torch.Tensor
        # с формой, соответствующей одной строке выхода модели.
        return np.fromiter(
            (
                bool(spec.is_ok(codec.decode_s(torch.as_tensor(row, dtype=torch.float32))))
                for row in Y_np
            ),
            dtype=bool,
            count=B,
        )

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return f"NNSurrogate(model={self.model.__class__.__name__}, swap_xy={getattr(self.model, 'swap_xy', None)})"


# ────────────────────────────────────────────────────────────────────────────
#                            InverseNNSurrogate
# ────────────────────────────────────────────────────────────────────────────

@register("inv_nn", "inverse_nn")
class InverseNNSurrogate(BaseSurrogate):
    """
    Обёртка над BaseLModule(**swap_xy=True**) – обратная модель: X ← S.

    ВАЖНО
    -----
    Этот surrogate НЕ предназначен для целей Penalty/Yield, потому что там
    ожидается predict(x_dict) -> NetworkLike.

    Здесь predict(net) -> dict параметров.
    Поэтому passes_spec(...) запрещён явным образом.
    """

    supports_uncertainty: bool = False

    def __init__(self, pl_module: "BaseLModule", *, device: str | torch.device | None = None):
        from mwlab.lightning.base_lm import BaseLModule

        if not isinstance(pl_module, BaseLModule):
            raise TypeError("InverseNNSurrogate expects a BaseLModule instance")

        if not bool(getattr(pl_module, "swap_xy", False)):
            raise ValueError("InverseNNSurrogate ждёт модель с swap_xy=True")

        self.model = pl_module.eval()
        if device is not None:
            self.model.to(device)

        self._class_path = f"{pl_module.__class__.__module__}:{pl_module.__class__.__qualname__}"

    @torch.no_grad()
    def predict(self, net: rf.Network, *, return_std: bool = False):
        if return_std:
            raise NotImplementedError("InverseNNSurrogate не поддерживает return_std")

        fn = getattr(self.model, "predict_x", None)
        if not callable(fn):
            raise AttributeError("Lightning-модуль не реализует метод predict_x(net) для обратного прогноза S→X")
        return fn(net)  # ожидаем dict параметров

    def passes_spec(self, xs, spec):
        """
        Явно запрещаем: спецификации применяются к NetworkLike, а этот surrogate
        возвращает X, а не NetworkLike.
        """
        raise TypeError(
            "InverseNNSurrogate.passes_spec неприменим: модель решает обратную задачу (S→X), "
            "а Specification проверяется на NetworkLike (S-параметрах)."
        )

    def save(self, path: str | Path):
        path = Path(path)
        payload = {
            "class": self._class_path,
            "state_dict": self.model.state_dict(),
            "swap_xy": bool(getattr(self.model, "swap_xy", True)),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "InverseNNSurrogate":
        payload = torch.load(path, map_location="cpu")
        ModCls = _locate_class(payload["class"])

        lm = ModCls.__new__(ModCls)  # type: ignore[call-arg]
        try:
            ModCls.__init__(lm, model=nn.Identity())
        except TypeError:
            ModCls.__init__(lm, model=nn.Identity(), swap_xy=True, auto_decode=False)

        lm.load_state_dict(payload["state_dict"])

        if not bool(getattr(lm, "swap_xy", False)):
            raise ValueError("Checkpoint содержит swap_xy=False; используйте NNSurrogate.load(...)")

        return cls(pl_module=lm)

    def __repr__(self):  # pragma: no cover
        return f"InverseNNSurrogate(model={self.model.__class__.__name__}, swap_xy={getattr(self.model, 'swap_xy', None)})"


__all__ = [
    "NNSurrogate",
    "InverseNNSurrogate",
]
