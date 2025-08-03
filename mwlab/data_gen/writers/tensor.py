# mwlab/data_gen/writers/tensor.py
"""
TensorWriter – универсальный Writer для любых «сырых» объектов
==============================================================
Первоначально задуман для аккумуляции батчей **torch.Tensor**, но на практике
может хранить *любые* объекты, если downstream‑код знает, как их обрабатывать.

Опция ``stack``
---------------
Если ``True`` (по умолчанию) и все элементы ``outputs`` являются Tensor‑ами
одинаковой формы, Writer делает ``torch.stack`` → получается единый 1‑го уровня
тензор ``(N, *shape)``.  При несовместимых формах Writer автоматически
возвращается к списку объектов.

Метод :pyfunc:`result` возвращает собранный ``data`` + ``meta`` + ``params`` –
это удобно в тестах и Jupyter‑прототипах.
"""

from __future__ import annotations

import threading
from typing import Any, List, Mapping, Sequence

import torch

from mwlab.data_gen.base import Batch, MetaBatch, Outputs, Writer

__all__ = ["TensorWriter"]


class TensorWriter(Writer):
    """Writer‑аккумулятор для тензоров или произвольных Python‑объектов."""

    # ------------------------------------------------------------------ init
    def __init__(self, *, stack: bool = True):
        self.stack = stack
        self._data: List[Any] = []
        self._meta: List[Any] = []
        self._params: List[Any] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write
    def write(self, outputs: Outputs, meta: MetaBatch, params: Batch):  # noqa: D401
        if not (len(outputs) == len(meta) == len(params)):
            raise ValueError("TensorWriter: входные последовательности разной длины")
        with self._lock:
            self._data.extend(outputs)
            self._meta.extend(meta)
            self._params.extend(params)

    # ------------------------------------------------------------------ flush
    def flush(self):  # noqa: D401 – данные уже в RAM
        pass

    # ------------------------------------------------------------------ result helper
    def result(self) -> dict[str, Any]:  # noqa: D401
        """Собирает итоговые data/meta/params в единый словарь."""
        with self._lock:
            data = list(self._data)
            meta = list(self._meta)
            params = list(self._params)

        if self.stack and data and all(isinstance(x, torch.Tensor) for x in data):
            shapes = {tuple(t.shape) for t in data}
            data_obj = torch.stack(data) if len(shapes) == 1 else data
        else:
            data_obj = data
        return {"data": data_obj, "meta": meta, "params": params}
