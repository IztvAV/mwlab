#mwlab/io/backends/hdf5_backend.py
"""
HDF5Backend – монолитный файл с S‑матрицами.

• Каждому образцу соответствует группа /samples/<idx>:
      - 's'               – (F,P,P) complex64
      - 'f'   (опц.)      – (F,)    float64   (если частоты нестандартные)
      - 'unit', 's_def', 'z0', 'comments' – метаданные сети
• Общая частотная сетка (если есть) хранится в /common_f
• Пользовательские параметры – в attrs соответствующей группы

Чтение может быть в режиме swmr=True; при необходимости используем .refresh()
для видимости новых данных без перезагрузки файла.
"""

from __future__ import annotations

import contextlib
import h5py
import numpy as np
import skrf as rf

from mwlab.io.touchstone import TouchstoneData
from .base import StorageBackend


class HDF5Backend(StorageBackend):
    """StorageBackend, сохраняющий отдельные TouchstoneData в один .h5‑файл."""

    def __init__(self, path: str, mode: str = "r", in_memory: bool = False):
        """
        Параметры
        ---------
        path : str
            Путь к .h5‑файлу.
        mode : {'r', 'w', 'a'}
            Режим открытия файла:
              - 'r'  → только чтение, swmr=True
              - 'w'  → перезапись
              - 'a'  → дозапись
        in_memory : bool
            Если True – загружает весь файл в оперативную память (без записи изменений на диск).
        """
        self.path = path
        self.mode = mode
        self.in_memory = in_memory

        kwargs = {}
        if in_memory:
            # Загружаем весь файл в память, без последующего сохранения
            kwargs.update(driver="core", backing_store=False)

        swmr_flag = (mode == "r") and not in_memory  # <- SWMR только когда не 'core'

        self.h5 = h5py.File(
            path,
            mode,
            libver="latest",    # libver='latest' дает поддержку swmr
            swmr=swmr_flag,
            **kwargs
        )

        # Создаем корневую группу /samples (при необходимости)
        if mode in ("w", "a") and "samples" not in self.h5:
            self.h5.create_group("samples")

        self._reader_swmr = self.h5.swmr_mode

    # --------------------------------------------------------- API StorageBackend
    def __len__(self) -> int:
        """Безопасное определение количества записей. Если нужно, обновляем метаданные."""
        if "samples" not in self.h5:
            return 0

        if self._reader_swmr and hasattr(self.h5, "refresh"):
            self.h5.refresh()

        return len(self.h5["/samples"])

    def read(self, idx: int) -> TouchstoneData:
        """
        Чтение одной записи из HDF5-файла.
        Формируется объект rf.Network, словарь параметров и все оборачивается в TouchstoneData.
        """
        grp = self.h5[f"/samples/{idx}"]

        s = grp["s"][...]
        # Используем индивидуальную или общую сетку частот
        f = grp["f"][...] if "f" in grp else self.h5["/common_f"][...]
        unit = grp["unit"][()].decode()
        s_def = grp["s_def"][()].decode()
        z0 = grp["z0"][...]

        freq = rf.Frequency.from_f(f, unit=unit)
        net = rf.Network(frequency=freq, s=s, z0=z0)
        net.s_def = s_def

        if "comments" in grp:
            net.comments = grp["comments"][()].tobytes().decode().split("\n")

        # Читаем пользовательские параметры из attrs
        params = {k: v for k, v in grp.attrs.items()}

        return TouchstoneData(net, params)

    def append(self, ts: TouchstoneData) -> None:
        """
        Сохраняет новый образец TouchstoneData в HDF5-файл.
        Если частотная сетка совпадает с существующей – сохраняется только один раз в /common_f.
        """
        if self.mode not in ("w", "a"):
            raise IOError("Файл открыт в режиме только чтение.")

        idx = len(self)
        grp = self.h5["/samples"].create_group(str(idx))

        dct = ts.to_numpy()

        # --- сохраняем матрицу S без сжатия (для ускорения чтения)
        grp.create_dataset("s", data=dct["s"], compression=None)

        # --- частотная сетка --------------------------------------------
        if "common_f" in self.h5:
            root_f = self.h5["/common_f"]
            # уже есть частоты – проверим совпадение
            if root_f.shape != dct["f"].shape or not np.allclose(root_f[...], dct["f"]):
                # не совпадают – сохраняем индивидуально
                grp.create_dataset("f", data=dct["f"])
        else:
            # первая запись – объявляем общую сетку
            self.h5.create_dataset("common_f", data=dct["f"])

        # --- метаданные -------------------------------------------------
        grp.create_dataset("unit", data=dct["meta/unit"])
        grp.create_dataset("s_def", data=dct["meta/s_def"])
        grp.create_dataset("z0", data=dct["meta/z0"])
        if "meta/comments" in dct:
            grp.create_dataset("comments", data=dct["meta/comments"])

        # --- параметры пользователя --------------------------------------
        for k, v in ts.params.items():
            grp.attrs[k] = v

        # Применяем изменения
        self.h5.flush()

        # Если режим еще не swmr – активируем (для безопасного чтения в другом процессе)
        if not self.h5.swmr_mode and self.h5.driver != "core":
            self.h5.swmr_mode = True

    # --------------------------------------------------------- housekeeping
    def close(self):
        """Закрывает файл, если он еще открыт."""
        if getattr(self, "h5", None) and self.h5.id.valid:
            self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with contextlib.suppress(Exception):
            self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()