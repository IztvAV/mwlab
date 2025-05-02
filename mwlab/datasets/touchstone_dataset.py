# datasets/touchstone_dataset.py
"""
TouchstoneDataset – теперь принимает ЛЮБОЙ StorageBackend.

Пример использования:
    backend = FileBackend(root="data/")
    ds = TouchstoneDataset(backend,
                           x_tf=X_SelectKeys(['w', 'gap']),
                           s_tf=S_Crop(1e9, 10e9))

    backend_h5 = HDF5Backend("train.h5", mode="r")
    ds_h5 = TouchstoneDataset(backend_h5)
"""

from __future__ import annotations

import numpy as np
import pathlib
from torch.utils.data import Dataset
from typing import Callable, Optional

from mwlab.io.backends import StorageBackend, FileBackend, HDF5Backend

def get_backend(path: pathlib.Path,
                pattern: str = "*.s?p") -> StorageBackend:
    """
    Возвращает подходящий backend по типу *path*.
    • Каталог           → FileBackend(root)
    • .h5 / .hdf5       → HDF5Backend(path, 'r')
    • .zarr             → ZarrBackend(path, 'r')   # когда появится
    • иначе ValueError
    """
    if path.is_dir():
        return FileBackend(path, pattern)
    if path.suffix in (".h5", ".hdf5"):
        return HDF5Backend(str(path), mode="r")
    raise ValueError(f"Не знаю, какой backend выбрать для {path}")


class TouchstoneDataset(Dataset):
    """
    Итератор над StorageBackend.

    * __len__   – длина backend
    * __getitem__ – читает TouchstoneData, применяет трансформы,
                    возвращает (x-dict, network|tensor)
    """

    def __init__(
            self,
            source: StorageBackend | str | pathlib.Path,
            *,
            x_keys=None,
            x_tf=None,
            s_tf=None,
            pattern="*.s?p"
    ):
        if isinstance(source, (str, pathlib.Path)):
            backend = get_backend(pathlib.Path(source), pattern)
        else:
            backend = source   # уже готовый backend

        self.backend = backend
        self.x_keys = x_keys
        self.x_tf = x_tf
        self.s_tf = s_tf

    def __len__(self):
        return len(self.backend)

    def __getitem__(self, idx):
        ts = self.backend.read(idx)  # TouchstoneData

        # ----------- X (скалярные параметры) -----------
        x = {k: ts.params.get(k, np.nan) for k in (self.x_keys or ts.params)}
        if self.x_tf:
            x = self.x_tf(x)

        # ----------- S-матрица --------------------------
        net = ts.network
        s_out = self.s_tf(net) if self.s_tf else net

        return x, s_out
