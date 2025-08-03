# mwlab/data_gen/writers/list.py
"""
ListWriter – самый лёгкий Writer
================================
Держит всё, что приходит от генератора, прямо в памяти.  Полезен в тестах,
прототипах и случаях, когда объём данных невелик или когда Writer нужен как
«промежуточный приёмник» перед дальнейшей обработкой в RAM.

Контракт Writer-а (см. ``mwlab.data_gen.base.Writer``):
    write(outputs, meta_batch, params_batch)
*Все* входные последовательности одинаковой длины.
"""

from __future__ import annotations

import threading
from typing import Any, List, Sequence

from mwlab.data_gen.base import MetaBatch, Outputs, Writer, Batch

__all__ = ["ListWriter"]


class ListWriter(Writer):
    """Аккумулирует данные, метаинформацию и params в обычных списках."""

    # ------------------------------------------------------------------ init
    def __init__(self):
        self._outputs: List[Any] = []
        self._meta: List[Any] = []
        self._params: List[Any] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write
    def write(self, outputs: Outputs, meta: MetaBatch, params: Batch):  # noqa: D401
        if not (len(outputs) == len(meta) == len(params)):
            raise ValueError("ListWriter: разные длины входных последовательностей")
        with self._lock:
            self._outputs.extend(outputs)
            self._meta.extend(meta)
            self._params.extend(params)

    # ------------------------------------------------------------------ flush
    def flush(self):  # noqa: D401 – здесь нечего сбрасывать
        pass

    # ------------------------------------------------------------------ helpers
    def result(self):  # noqa: D401
        """Возвращает словарь с накопленными данными (для тестов/отладок)."""
        with self._lock:
            return {
                "data": list(self._outputs),
                "meta": list(self._meta),
                "params": list(self._params),
            }
