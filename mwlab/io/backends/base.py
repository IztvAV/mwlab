#mwlab/io/backends/base.py
"""
Абстрактный базовый класс StorageBackend.

* read(idx)  -> TouchstoneData
* append(ts) -> None (может не поддерживаться, тогда NotImplementedError)
"""

from __future__ import annotations

import abc

from mwlab.io.touchstone import TouchstoneData


class StorageBackend(abc.ABC):
    """Минимальный контракт, который понимает TouchstoneDataset."""

    # --------- чтение
    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def read(self, idx: int) -> TouchstoneData: ...

    # --------- запись (опционально)
    def append(self, ts: TouchstoneData) -> None:  # noqa: D401
        """Добавление новой записи (может не поддерживаться)."""
        raise NotImplementedError
