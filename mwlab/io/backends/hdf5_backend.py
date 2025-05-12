#mwlab/io/backends/hdf5_backend.py
"""
HDF5Backend – монолитный *.h5‑файл с набором S‑матриц.

Если ``in_memory=True`` – Backend ведет себя как «CachedBackend»:
* При инициализации **однократно** читает ВСЕ записи из файла и
  сохраняет их в список ``self._cache``.
* Файл после загрузки закрывается, дальнейшие обращения идут напрямую
  к объектам Python → минимальные накладные расходы.
* В режиме ``in_memory`` разрешено ТОЛЬКО чтение (``mode='r'``).

Если ``in_memory=False`` – поведение прежнее: ленивое чтение через h5py.

Формат файла
============
/samples/<idx>/
    ├── s           : (F, P, P) complex64
    ├── f (опц.)    : (F,)       float64      – индивидуальная сетка
    ├── unit        : str      – 'Hz' | 'GHz' …
    ├── s_def       : str      – 'S' / 'T' / …
    ├── z0          : (P,)       float64      – опорные сопротивления
    └── comments    : bytes    – UTF‑8, разделитель \n (опц.)
/common_f           : (F,) float64 – общая частотная сетка (если есть)
attrs группы – произвольные пользовательские параметры (int/float/str).
"""

from __future__ import annotations

import contextlib
import pathlib
from typing import List, Dict

import h5py
import numpy as np
import skrf as rf

from mwlab.io.touchstone import TouchstoneData
from .base import StorageBackend

__all__ = ["HDF5Backend"]


class HDF5Backend(StorageBackend):
    """Backend для хранения/чтения *TouchstoneData* в одном *.h5*‑файле.

    Parameters
    ----------
    path : str | pathlib.Path
        Путь к HDF5‑файлу.
    mode : {"r", "w", "a"}, default "r"
        Режим открытия.
        * ``"r"`` – только чтение (по умолчанию).
          При ``in_memory=False`` открывается с ``swmr=True``.
        * ``"w"`` – пересоздать файл.
        * ``"a"`` – дозапись в существующий.
    in_memory : bool, default False
        * **True**  → превратить файл в «CachedBackend»:
          все записи загружаются один раз в память; файл закрывается.
          Разрешен ТОЛЬКО при ``mode='r'``.
        * **False** → ленивое чтение через h5py.
    """

    # ---------------------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ
    # ---------------------------------------------------------------------
    def __init__(self, path: str | pathlib.Path, mode: str = "r", *, in_memory: bool = False):
        self.path = str(path)
        self.mode = mode
        self._in_memory = in_memory

        if in_memory and mode != "r":
            raise ValueError("in_memory=True поддерживается только с mode='r'.")

        # ------------------------------------------------------------------
        # 1) in_memory=False  → классический путь (h5py File ↔ ленивое чтение)
        # 2) in_memory=True   → откроем файл, прочитаем все, закроем.
        # ------------------------------------------------------------------
        if not in_memory:
            # ---------- обычный (ленивый) режим
            swmr_flag = (mode == "r")  # SWMR недоступен для 'w'/'a'
            self.h5 = h5py.File(
                self.path,
                mode,
                libver="latest",
                swmr=swmr_flag,
            )

            # создаем /samples при необходимости (для 'w'/'a')
            if mode in ("w", "a") and "samples" not in self.h5:
                self.h5.create_group("samples")

            self._reader_swmr = self.h5.swmr_mode
            self._cache: List[TouchstoneData] | None = None  # нет кэша
        else:
            # ---------- режим полного кэша (CachedBackend)
            with h5py.File(self.path, "r", libver="latest", swmr=False) as h5:
                if "samples" not in h5:
                    raise RuntimeError("Файл не содержит группы '/samples'.")
                n = len(h5["/samples"])
                self._cache = [self._read_from_h5(h5, i) for i in range(n)]
            # файл более не нужен
            self.h5 = None
            self._reader_swmr = False  # не используется

    # ------------------------------------------------------------------
    # ВНУТРЕННИЙ МЕТОД: чтение одной записи из открытого h5py.File
    # ------------------------------------------------------------------
    @staticmethod
    def _read_from_h5(h5: h5py.File, idx: int) -> TouchstoneData:
        """Считывает запись *idx* из already‑opened h5py.File и возвращает TouchstoneData."""
        grp = h5[f"/samples/{idx}"]

        # --- матрица S и частоты -----------------------------------------
        s = grp["s"][...]  # без сжатия, просто копирование в NumPy
        f = grp["f"][...] if "f" in grp else h5["/common_f"][...]

        # --- метаданные сети ---------------------------------------------
        unit = grp["unit"][()].decode()
        s_def = grp["s_def"][()].decode()
        z0 = grp["z0"][...]

        freq = rf.Frequency.from_f(f, unit=unit)
        net = rf.Network(frequency=freq, s=s, z0=z0)
        net.s_def = s_def

        if "comments" in grp:
            net.comments = (
                grp["comments"][()].tobytes().decode().split("\n")
            )

        # --- пользовательские параметры ----------------------------------
        params = {k: v for k, v in grp.attrs.items()}

        return TouchstoneData(net, params)

    # ------------------------------------------------------------------
    # АПИ StorageBackend
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        if self._in_memory:
            return len(self._cache)  # type: ignore[arg-type]

        if "samples" not in self.h5:  # type: ignore[operator]
            return 0

        # в SWMR‑режиме можно освежить метаданные
        if self._reader_swmr and hasattr(self.h5, "refresh"):
            self.h5.refresh()
        return len(self.h5["/samples"])  # type: ignore[index]

    def read(self, idx: int) -> TouchstoneData:  # noqa: D401
        """Возвращает TouchstoneData по индексу *idx*."""
        if self._in_memory:
            return self._cache[idx]  # type: ignore[index]
        return self._read_from_h5(self.h5, idx)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # ЗАПИСЬ НОВОГО ОБРАЗЦА (недоступно в in_memory‑режиме)
    # ------------------------------------------------------------------
    def append(self, ts: TouchstoneData) -> None:
        if self._in_memory:
            raise IOError("Backend находится в режиме in_memory – только чтение.")

        if self.mode not in ("w", "a"):
            raise IOError("Файл открыт в режиме только чтение.")

        idx = len(self)
        grp = self.h5["/samples"].create_group(str(idx))  # type: ignore[index]

        dct = ts.to_numpy()

        # --- матрица S ----------------------------------------------------
        grp.create_dataset("s", data=dct["s"], compression=None)

        # --- частоты ------------------------------------------------------
        if "common_f" in self.h5:  # type: ignore[operator]
            root_f = self.h5["/common_f"]  # type: ignore[index]
            if root_f.shape != dct["f"].shape or not np.allclose(root_f[...], dct["f"]):
                grp.create_dataset("f", data=dct["f"])
        else:
            self.h5.create_dataset("common_f", data=dct["f"])  # type: ignore[arg-type]

        # --- метаданные сети ---------------------------------------------
        grp.create_dataset("unit", data=dct["meta/unit"])
        grp.create_dataset("s_def", data=dct["meta/s_def"])
        grp.create_dataset("z0", data=dct["meta/z0"])
        if "meta/comments" in dct:
            grp.create_dataset("comments", data=dct["meta/comments"])

        # --- пользовательские параметры ----------------------------------
        for k, v in ts.params.items():
            grp.attrs[k] = v

        # --- финализация --------------------------------------------------
        self.h5.flush()
        if not self.h5.swmr_mode:
            self.h5.swmr_mode = True

    # ------------------------------------------------------------------
    # HOUSEKEEPING
    # ------------------------------------------------------------------
    def close(self):
        """Корректно закрывает файл, если он все еще открыт."""
        if getattr(self, "h5", None) and self.h5 is not None and self.h5.id.valid:  # type: ignore[attr-defined]
            self.h5.close()

    # поддержка контекстного менеджера
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with contextlib.suppress(Exception):
            self.close()

    def __del__(self):
        # аварийное закрытие при GC, если пользователь забыл вызвать close()
        with contextlib.suppress(Exception):
            self.close()
