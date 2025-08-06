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

def get_backend(path: pathlib.Path, **backend_kwargs) -> StorageBackend:
    """
    Автоматически выбирает backend по типу *path*.
    • Каталог           → FileBackend(root, pattern=...)
    • .h5 / .hdf5       → HDF5Backend(path, **kwargs)
    • .zarr             → ZarrBackend(path, **kwargs)  # in future
    """
    if path.is_dir():
        # для FileBackend оставляем только 'pattern', остальное игнорируем
        pattern = backend_kwargs.pop("pattern", "*.s[0-9]*p")
        return FileBackend(path, pattern)
    if path.suffix in (".h5", ".hdf5"):
        return HDF5Backend(str(path), **backend_kwargs)
    raise ValueError(f"Не знаю, какой backend выбрать для {path}")


def _is_scalar_number(x) -> bool:
    """True для скалярных int/float/bool (включая numpy-скаляры)."""
    # np.isscalar(True) == True, np.isscalar(np.float32(1.)) == True
    if not np.isscalar(x):
        return False
    return isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_))


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
            x_numeric_only: bool = False,
            x_drop_private: bool = False,
            **backend_kwargs
    ):
        if isinstance(source, (str, pathlib.Path)):
            backend = get_backend(pathlib.Path(source), **backend_kwargs)
        else:
            backend = source # уже готовый backend

        self.backend = backend
        self.x_keys = x_keys
        self.x_tf = x_tf
        self.s_tf = s_tf
        self.x_numeric_only = bool(x_numeric_only)
        self.x_drop_private = bool(x_drop_private)

    def __len__(self):
        return len(self.backend)

    def __getitem__(self, idx):
        ts = self.backend.read(idx)  # TouchstoneData

        # ----------- X (скалярные параметры) -----------
        raw = {k: ts.params.get(k, np.nan) for k in (self.x_keys or ts.params)}

        if self.x_drop_private:
            raw = {k: v for k, v in raw.items() if not str(k).startswith("__")}

        if self.x_numeric_only:
            raw = {k: v for k, v in raw.items() if _is_scalar_number(v)}

        x = self.x_tf(raw) if self.x_tf else raw

        # ----------- S-матрица --------------------------
        net = ts.network
        s_out = self.s_tf(net) if self.s_tf else net

        return x, s_out
