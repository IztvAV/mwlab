#mwlab/io/backends/hdf5_backend.py
"""
HDF5Backend – быстрый монолитный файл с S-матрицами.

• Один датасет 's' формы (N, F, P, P) – complex64
• Один датасет 'f' формы (F,)         – float64  (частоты общие!)
• Атрибуты параметров хранятся в
  группе samples[i].attrs[...], если частоты разные – делаем группу/запись.

**Ограничение**: все записи должны иметь одинаковую частотную сетку и
одинаковое число портов (типичный случай для ML).
"""

from __future__ import annotations

import contextlib
import h5py
import numpy as np
from mwlab.io.touchstone import TouchstoneData
from .base import StorageBackend


class HDF5Backend(StorageBackend):
    """
    Чтение происходит в режиме swmr=True (safe-read-only для DataLoader>1).
    """

    def __init__(self, path: str, mode: str = "r"):
        self.path = path
        self.mode = mode
        # libver='latest' дает поддержку swmr
        self.h5 = h5py.File(path, mode, libver="latest", swmr=(mode == "r"))

        # Если файл новый – создаем корневую группу /samples
        if mode in ("w", "a") and "samples" not in self.h5:
            self.h5.create_group("samples")

    # ------------------------------------------------------- StorageBackend
    def __len__(self) -> int:
        return len(self.h5["/samples"])

    def read(self, idx: int) -> TouchstoneData:  # noqa: D401
        grp = self.h5[f"/samples/{idx}"]
        data = {
            "s": grp["s"][...],
            "f": grp["f"][...],
        }
        # параметры как attrs
        for k, v in grp.attrs.items():
            data[f"param/{k}"] = np.asarray(v)
        return TouchstoneData.from_numpy(data)

    # ------------------------------------------------------- запись новой
    def append(self, ts: TouchstoneData) -> None:
        if self.mode not in ("w", "a"):
            raise IOError("Файл открыт в режиме только чтение.")
        idx = len(self.h5["/samples"])
        grp = self.h5["/samples"].create_group(str(idx))

        dct = ts.to_numpy()
        # --- сохраняем основную S-матрицу и частоты
        grp.create_dataset("s", data=dct["s"], compression="gzip", shuffle=True)
        grp.create_dataset("f", data=dct["f"])

        # --- параметры → attrs (строки и числа)
        for k, data in dct.items():
            if not k.startswith("param/"):
                continue
            name = k[6:]  # убираем "param/"
            if data.dtype == "uint8":
                # строка → bytes → str
                grp.attrs[name] = data.tobytes().decode("utf-8")
            else:
                # скаляр float32
                grp.attrs[name] = data.item()

        # Flush для одновременных читателей
        self.h5.flush()

    # ------------------------------------------------------- контекстный менеджер
    def close(self):
        if self.h5 and self.h5.id:  # файл ещё не закрыт
            self.h5.close()

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        with contextlib.suppress(Exception):
            self.h5.close()
