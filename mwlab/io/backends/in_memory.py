#mwlab/io/backends/in_memory.py
"""
In‑memory backend‑ы для Touchstone‑данных
========================================
Этот модуль предоставляет быстрые backend‑ы, работающие целиком в оперативной
памяти. Они устраняют накладные расходы на файловый I/O и унифицируют работу
с синтетическими и уже готовыми наборами.

Структура
---------
* **InMemoryBackendBase** – базовый миксин: хранит список `TouchstoneData`,
  умеет `append`, `dump_pickle`, `load_pickle`, `to_hdf5`.
* **RAMBackend** – «готовый список в RAM».
* **SyntheticBackend** – лениво или жадно генерирует данные по функции
  *factory(idx)*; поддерживает кэширование (полное или LRU‑N) и
  параллельную пред‑генерацию с помощью `ProcessPoolExecutor`.

Примеры использования
---------------------
```python
>>> from mwlab.io.backends import RAMBackend, SyntheticBackend
>>> from mwlab.io.touchstone import TouchstoneData

# --- RAMBackend -------------------------------------------------------
from pathlib import Path

# способ 1 — сразу из списка
meas = [TouchstoneData.load(p) for p in Path('vna_dumps').glob('*.s2p')]
backend = RAMBackend(meas)
print(len(backend), backend.read(0))

# способ 2 — постепенно через append()
backend2 = RAMBackend()
for p in Path('step_measurements').glob('*.s2p'):
    backend2.append(TouchstoneData.load(p))
print(len(backend2), backend2.read(-1))

backend.dump_pickle('meas.pkl')      # быстро сохранить снимок (snapshot)
backend_loaded = RAMBackend.load_pickle('meas.pkl')  # загрузить обратно
print(len(backend_loaded), backend_loaded.read(0))

# --- SyntheticBackend: полный пред‑кэш -------------------------------:

def coupling_matrix_factory(i: int) -> TouchstoneData:
    return synthesize_filter(i, order=6)

backend_syn = SyntheticBackend(
        length=100_000,
        factory=coupling_matrix_factory,  # передаем ссылку на функцию
        cache=True,                       # сохранить все в список
        workers=8                         # распараллелить генерацию
)
print(backend_syn[42])
backend_syn.to_hdf5('synthetic.h5')
print(backend_syn[42])
backend_syn.to_hdf5('synthetic.h5')

# --- SyntheticBackend: LRU‑кэш ---------------------------------------
online = SyntheticBackend(
        length=10_000,
        factory=lambda i: perturb_nominal(i, bw=100e6),
        cache=512,           # хранить 512 последних вызовов
)
```
"""

from __future__ import annotations

import pickle
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, Optional

import h5py

from mwlab.io.touchstone import TouchstoneData
from .base import StorageBackend

__all__ = [
    "InMemoryBackendBase",
    "RAMBackend",
    "SyntheticBackend",
]

# ────────────────────────────────────────────────────────────────────────────
#                             БАЗОВЫЙ КЛАСС
# ────────────────────────────────────────────────────────────────────────────

class InMemoryBackendBase(StorageBackend):
    """Миксин: хранение TouchstoneData в списке *self._data*.

    Реализует
    * ``__len__``, ``read`` – доступ по индексу;
    * ``append`` – добавление новой записи;
    * ``dump_pickle`` / ``load_pickle`` – быстрый snapshot на диск;
    * ``to_hdf5`` – экспорт в файл, совместимый с :class:`HDF5Backend`.
    """

    # ------------------------- утилиты --------------------------------
    @staticmethod
    def _read_pickle_data(path) -> List[TouchstoneData]:
        """Читает pickle-файл и возвращает список TouchstoneData."""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------- базовые методы ------------------------------------------
    def __len__(self):  # noqa: D401
        return len(self._data)  # type: ignore[attr-defined]

    def read(self, idx: int) -> TouchstoneData:
        if idx < 0:
            idx += len(self)
        return self._data[idx]  # type: ignore[attr-defined]

    def append(self, ts: TouchstoneData) -> None:
        self._data.append(ts)  # type: ignore[attr-defined]

    # ------- сериализация --------------------------------------------
    def dump_pickle(self, path: os.PathLike | str, *, protocol: int = 5) -> None:
        """Сохраняет backend в pickle‑файл (Python ≥ 3.8 — protocol 5)."""
        with open(path, "wb") as fh:
            pickle.dump(self._data, fh, protocol=protocol)  # type: ignore[attr-defined]


    # ------- экспорт в HDF5 ------------------------------------------
    def to_hdf5(self, path: os.PathLike | str, *, compression: str | None = None) -> None:
        """Экспортирует записи в *.h5*, совместимый с :class:`HDF5Backend`.

        Если ``compression`` is None → без сжатия; можно указать 'gzip', 'lzf', ...
        """
        with h5py.File(path, "w", libver="latest") as h5:
            root = h5.create_group("samples")
            for idx, ts in enumerate(self._data):  # type: ignore[attr-defined]
                grp = root.create_group(str(idx))
                dct = ts.to_numpy()

                grp.create_dataset("s", data=dct["s"], compression=compression)
                grp.create_dataset("f", data=dct["f"])
                grp.create_dataset("unit", data=dct["meta/unit"])
                grp.create_dataset("s_def", data=dct["meta/s_def"])
                grp.create_dataset("z0", data=dct["meta/z0"])
                if "meta/comments" in dct:
                    grp.create_dataset("comments", data=dct["meta/comments"])
                for k, v in ts.params.items():
                    grp.attrs[k] = v
            h5.flush()


# ────────────────────────────────────────────────────────────────────────────
#                 RAMBackend – статический список в памяти
# ────────────────────────────────────────────────────────────────────────────

class RAMBackend(InMemoryBackendBase):
    """Backend для *уже готового* списка TouchstoneData, живущего в RAM."""

    def __init__(self, data: Iterable[TouchstoneData] = ()):  # noqa: D401
        self._data: List[TouchstoneData] = list(data)

    # ---------- загрузка из pickle только для RAMBackend --------------
    @classmethod
    def load_pickle(cls, path) -> "RAMBackend":
        data = cls._read_pickle_data(path)
        if not isinstance(data, list) or not all(isinstance(x, TouchstoneData) for x in data):
            raise ValueError("Недопустимое содержимое pickle-файла")
        return cls(data)


# ────────────────────────────────────────────────────────────────────────────
#           SyntheticBackend – лениво / жадно генерируемые данные
# ────────────────────────────────────────────────────────────────────────────

class SyntheticBackend(InMemoryBackendBase):
    """Backend, который генерирует TouchstoneData по функции *factory(idx)*.

    Параметры
    ---------
    length : int
        Сколько «виртуальных» образцов содержит backend.
    factory : Callable[[int], TouchstoneData]
        Функция‑синтезатор: получает индекс *i* и возвращает TouchstoneData.
    cache : bool | int, default True
        * ``True``  → сгенерировать и хранить **все** записи (аналог RAMBackend).
        * ``int``   → LRU‑кэш последних *N* обращений.
        * ``False`` → не кешировать: каждый вызов `read(i)` заново вызывает фабрику.
    workers : int | None, default None
        Если ``cache is True`` – позволяет распараллелить начальную генерацию с
        помощью ``ProcessPoolExecutor(workers)``. При ``cache`` = *int* или
        ``False`` этот параметр игнорируется.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        length: int,
        factory: Callable[[int], TouchstoneData],
        *,
        cache: bool | int = True,
        workers: Optional[int] = None,
    ) -> None:
        self.length = int(length)
        self.factory = factory
        # --------------------------------------------------------------
        # append() допустим ТОЛЬКО при cache=True (полный список)
        self._allow_append = cache is True

        # ------------- типы кэшей -------------------------------------
        if cache is True:
            # eager‑кэш: генерируем все сразу
            if workers and workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    self._data: List[TouchstoneData] = list(ex.map(factory, range(length)))
            else:
                self._data = [factory(i) for i in range(length)]
            self._lru: OrderedDict[int, TouchstoneData] | None = None
        elif cache is False:
            self._data = []               # пустой список, не используем
            self._lru = None
        else:  # cache = int  -> LRU‑N
            self._data = []               # нет полного списка
            self._lru = OrderedDict()
            self._max_lru = int(cache)

    # ------------------------------------------------------------------
    def __len__(self):  # noqa: D401
        return self.length

    def read(self, idx: int) -> TouchstoneData:  # noqa: D401
        if idx < 0:
            idx += self.length
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        # --- полный кэш ------------------------------------------------
        if hasattr(self, "_data") and idx < len(self._data):
            # либо eager‑кэш, либо append‑нутый элемент при cache=True
            return self._data[idx]

        # --- LRU‐кэш ---------------------------------------------------
        if self._lru is not None and idx in self._lru:
            self._lru.move_to_end(idx)
            return self._lru[idx]

        # --- генерируем -----------------------------------------------
        ts = self.factory(idx)

        # сохраняем
        if self._lru is not None:
            self._lru[idx] = ts
            self._lru.move_to_end(idx)
            # ограничиваем размер
            while len(self._lru) > self._max_lru:
                self._lru.popitem(last=False)
        return ts

    # ------------------------------------------------------------------
    def append(self, ts: TouchstoneData) -> None:
        """Добавляем *готовый* образец в конец.

        Доступно только при ``cache is True`` (полный список)."""
        if not self._allow_append:
            raise OSError("append() доступен только при cache=True")
        self._data.append(ts)
        self.length += 1

    # ---------- загрузка из pickle не поддерживается ------------------
    @classmethod
    def load_pickle(cls, path):
        raise IOError(
            "Невозможно восстановить SyntheticBackend из pickle — "
            "сохраняйте/загружайте такой набор как RAMBackend."
        )

    # ------------------------------------------------------------------
    # dump_pickle / to_hdf5 наследуются без изменений
    # ------------------------------------------------------------------
