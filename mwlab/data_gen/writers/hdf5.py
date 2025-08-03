# mwlab/data_gen/writers/hdf5.py
"""
HDF5Writer – Writer, агрегирующий TouchstoneData в один *.h5*-файл
==================================================================
Использует тот же внутренний формат, что и :class:`mwlab.io.backends.HDF5Backend`,
поэтому результат можно напрямую открывать этим backend-ом или использовать
в :class:`mwlab.datasets.TouchstoneDataset`.

Опции:
* ``compression`` – алгоритм сжатия для datasets (``None`` → без сжатия).
* ``overwrite``   – удалять существующий файл перед началом записи.

Потокобезопасность достигается с помощью ``threading.Lock`` вокруг операций
``append`` (``h5py`` не потокобезопасен).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Mapping, Sequence
from filelock import FileLock

from mwlab.data_gen.base import Batch, MetaBatch, Outputs, Writer
from mwlab.io.backends.hdf5_backend import HDF5Backend
from mwlab.io.touchstone import TouchstoneData

__all__ = ["HDF5Writer"]


class HDF5Writer(Writer):
    """Сохраняет батчи TouchstoneData в монолитный HDF5."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        path: str | Path,
        *,
        compression: str | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and overwrite:
            self.path.unlink()
        # backend откроем в __enter__, чтобы файл не оставался открытым до контекста
        self.comp = compression
        self._backend: HDF5Backend | None = None
        self._lock = threading.Lock()
        self._file_lock = FileLock(str(self.path) + ".lock")

    # ---------------------------------------------------------------- context
    def __enter__(self):  # noqa: D401
        self._backend = HDF5Backend(self.path, mode="a", in_memory=False)
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        try:
            self.flush()
        finally:
            if self._backend is not None:
                self._backend.close()
        return False

    # ---------------------------------------------------------------- write
    def write(self, outputs: Outputs, meta: MetaBatch, params: Batch):  # noqa: D401
        if not (len(outputs) == len(meta) == len(params)):
            raise ValueError("HDF5Writer: входные последовательности разной длины")
        if self._backend is None:
            raise RuntimeError("HDF5Writer: backend не инициализирован (не вызван __enter__)")

        for ts_obj, meta_dct in zip(outputs, meta):
            if not isinstance(ts_obj, TouchstoneData):
                raise TypeError("HDF5Writer ожидает TouchstoneData в outputs")
            if meta_dct:
                if not isinstance(meta_dct, Mapping):
                    raise TypeError("meta_batch должен содержать dict или быть пустым")
                ts_obj.params.update(meta_dct)  # добавляем метаданные
            # потокобезопасный append
            with self._file_lock, self._lock:
                self._backend.append(ts_obj)

    # ---------------------------------------------------------------- flush
    def flush(self):  # noqa: D401
        if self._backend is not None and self._backend.h5 is not None:
            with self._lock:
                self._backend.h5.flush()  # type: ignore[attr-defined]
