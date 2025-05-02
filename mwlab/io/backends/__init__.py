#mwlab/io/backends/__init__.py
"""
mwlab.io.backends
-----------------
Пакет backend-ов хранения данных.

Импортируем «из коробки» FileBackend (по-умолчанию) и HDF5Backend.
Пользователь может писать свои backend-ы, достаточно реализовать класс,
унаследованный от StorageBackend.
"""

from .base import StorageBackend
from .file_backend import FileBackend
from .hdf5_backend import HDF5Backend

__all__ = ["StorageBackend", "FileBackend", "HDF5Backend"]
