# mwlab/data_gen/writers/ram.py
"""
RAMWriter – Writer, который копит TouchstoneData в оперативной памяти
====================================================================
Использует :class:`mwlab.io.backends.RAMBackend` в качестве контейнера.  Полезно,
когда следующий этап пайплайна (например, обучение ML-модели) происходит в той
же сессии Python и не требует промежуточных файлов.
"""

from __future__ import annotations

import threading
from typing import Mapping, Sequence

from mwlab.data_gen.base import Batch, MetaBatch, Outputs, Writer
from mwlab.io.backends.in_memory import RAMBackend
from mwlab.io.touchstone import TouchstoneData

__all__ = ["RAMWriter"]


class RAMWriter(Writer):
    """Собирает TouchstoneData в RAMBackend."""

    # ------------------------------------------------------------------ init
    def __init__(self):
        self._backend = RAMBackend()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write
    def write(self, outputs: Outputs, meta: MetaBatch, params: Batch):  # noqa: D401
        if not (len(outputs) == len(meta) == len(params)):
            raise ValueError("RAMWriter: входные последовательности разной длины")
        for ts_obj, meta_dct in zip(outputs, meta):
            if not isinstance(ts_obj, TouchstoneData):
                raise TypeError("RAMWriter ожидает TouchstoneData в outputs")
            if meta_dct:
                if not isinstance(meta_dct, Mapping):
                    raise TypeError("meta_batch должен быть dict или пустым")
                ts_obj.params.update(meta_dct)
            with self._lock:
                self._backend.append(ts_obj)

    # ------------------------------------------------------------------ helpers
    def backend(self) -> RAMBackend:  # noqa: D401
        """Возвращает внутренний RAMBackend (read‑only)."""
        return self._backend

    def flush(self):  # noqa: D401 – ничего сбрасывать не нужно
        pass
