#mwlab/io/backends/hdf5_backend.py
"""
HDF5Backend – монолитный файл с S‑матрицами.

• Каждому образцу соответствует группа /samples/<idx>
      - 's'               – (F,P,P) complex64
      - 'f'   (опц.)      – (F,)    float64   (если частоты нестандартные)
      - 'unit', 's_def', 'z0', 'comments' – метаданные сети
• Пользовательские параметры храним в attrs группы.

Чтение в режиме swmr=True; при необходимости используем .refresh()
для видимости новых данных без перезагрузки файла.
"""

from __future__ import annotations

import contextlib
import h5py
import skrf as rf

from mwlab.io.touchstone import TouchstoneData
from .base import StorageBackend

class HDF5Backend(StorageBackend):
    """StorageBackend, сохраняющий отдельные образцы (touchstone-данные) в один .h5‑файл."""

    def __init__(self, path: str, mode: str = "r"):
        """
        Parameters
        ----------
        path : str
            Имя .h5‑файла.
        mode : {'r', 'w', 'a'}
            Режим: чтение | новая запись | дозапись.
            • 'r'  → swmr=True (safe-read-only)
        """
        self.path = path
        self.mode = mode

        self.h5 = h5py.File(
            path,
            mode,
            libver="latest",     # libver='latest' дает поддержку swmr
            swmr=(mode == "r"),
        )

        # Для новых файлов создаем корневую группу /samples
        if mode in ("w", "a") and "samples" not in self.h5:
            self.h5.create_group("samples")

        # Флаг: файл открыт для swmr‑чтения
        self._reader_swmr = self.h5.swmr_mode

    # ------------------------------------------------ StorageBackend API
    def __len__(self) -> int:  # noqa: D401
        """Безопасно получаем длину; refresh() при необходимости."""
        if "samples" not in self.h5:
            return 0

        if self._reader_swmr and hasattr(self.h5, "refresh"):
            # для h5py ≥ 3.9 — обновляем метаданные
            self.h5.refresh()

        return len(self.h5["/samples"])

    def read(self, idx: int) -> TouchstoneData:
        """
        Чтение одной записи.
        Примечание: параметры собираем напрямую в dict, минуя from_numpy().
        """
        grp = self.h5[f"/samples/{idx}"]

        # -- восстановим rf.Network --------------------------------------
        s = grp["s"][...]
        f = grp["f"][...] if "f" in grp else grp.parent["common_f"][...]
        unit = grp["unit"][()].decode()
        s_def = grp["s_def"][()].decode()
        z0 = grp["z0"][...]

        freq = rf.Frequency.from_f(f, unit=unit)
        net = rf.Network(frequency=freq, s=s, z0=z0)
        net.s_def = s_def

        if "comments" in grp:
            net.comments = grp["comments"][()].tobytes().decode().split("\n")

        # -- параметры пользователя --------------------------------------
        params = {k: v for k, v in grp.attrs.items()}

        return TouchstoneData(net, params)

    # ------------------------------------------------ запись новой записи
    def append(self, ts: TouchstoneData) -> None:
        if self.mode not in ("w", "a"):
            raise IOError("Файл открыт в режиме только чтение.")

        idx = len(self)  # с учетом возможного refresh()
        grp = self.h5["/samples"].create_group(str(idx))

        dct = ts.to_numpy()

        # --- данные S‑матрицы и частоты ---------------------------------
        grp.create_dataset("s", data=dct["s"], compression="gzip", shuffle=True)
        # если у всех общие частоты – можно хранить в корне, но пока кладем внутрь
        grp.create_dataset("f", data=dct["f"])

        # --- системные метаданные ---------------------------------------
        grp.create_dataset("unit", data=dct["meta/unit"])
        grp.create_dataset("s_def", data=dct["meta/s_def"])
        grp.create_dataset("z0", data=dct["meta/z0"])
        if "meta/comments" in dct:
            grp.create_dataset("comments", data=dct["meta/comments"])

        # --- параметры пользователя (string / число) --------------------
        for k, v in ts.params.items():
            grp.attrs[k] = v

        # Сразу делаем файл доступным читателям
        self.h5.flush()
        if not self.h5.swmr_mode:
            # Включаем swmr после первой записи (требование HDF5)
            self.h5.swmr_mode = True

    # ------------------------------------------------ housekeeping
    def close(self):
        if getattr(self, "h5", None) and self.h5.id.valid:
            self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with contextlib.suppress(Exception):
            self.close()

    def __del__(self):
        # на случай, если забыли закрыть явно
        with contextlib.suppress(Exception):
            self.close()
