# mwlab/data_gen/sources/csv.py
"""
CsvSource – ParamSource, основанный на табличном CSV/TSV‑файле
============================================================
Файл выступает «расшаренным чек‑листом» между несколькими воркерами.
Столбцы служебных метаданных:

* ``__id``      – **уникальный идентификатор** строки (строка).  Если столбца
  нет, он создаётся автоматически («p0», «p1», …).
* ``__status``  – текущее состояние точки:
    ""          – *pending*  (ещё не обработана);
    "reserved"  – захвачена воркером, но ещё не завершена;
    "done"       – успешно завершена;
    "failed"     – генератор кинул исключение.
* ``__error``   – текст последней ошибки (заполняется в mark_failed).

Потокобезопасность обеспечивается instance‑lock‑ом.  Для truly multi‑process
шаринга потребуется внешний файл‑lock; здесь мы предполагаем, что каждый
процесс открывает *свой* CsvSource (pandas загружает файл целиком в RAM), а
изменения пишутся обратно при ``__exit__``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
from filelock import FileLock

from mwlab.data_gen.base import ParamDict, ParamSource

__all__ = ["CsvSource"]


class CsvSource(ParamSource):
    """Источник параметров из табличного файла CSV/TSV/…"""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        id_col: str = "__id",
        status_col: str = "__status",
        error_col: str = "__error",
        reserve_tag: str = "reserved",
    ) -> None:
        self.path = Path(path)
        self.delim = delimiter or ("," if self.path.suffix.lower() == ".csv" else "\t")
        self.id_col = id_col
        self.status_col = status_col
        self.error_col = error_col
        self.reserve_tag = reserve_tag

        # контейнеры будут созданы в __enter__
        self._df: pd.DataFrame
        self._iter: Iterator[pd.Series]
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- context
    def __enter__(self):  # noqa: D401
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        # читаем строковые значения как есть; NA -> ""
        self._df = pd.read_csv(self.path, sep=self.delim, keep_default_na=False)  # дать pandas вывести типы

        # гарантируем служебные колонки
        for col in (self.id_col, self.status_col, self.error_col):
            if col not in self._df.columns:
                self._df[col] = ""

        # --- назначаем уникальные __id, если отсутствуют ---
        ids = self._df[self.id_col].astype(str)
        need_gen = ids == ""
        if need_gen.any():
            base = self._df.index.astype(str).map(lambda x: f"p{x}")
            self._df.loc[need_gen, self.id_col] = base[need_gen]
        if self._df[self.id_col].duplicated().any():
            dups = self._df[self.id_col][self._df[self.id_col].duplicated()].unique()
            raise ValueError(f"CsvSource: дублирующиеся id: {dups[:5]!r} …")

        # итератор по ещё не завершённым строкам
        pending_mask = ~self._df[self.status_col].isin(["done", "failed"])
        self._iter = (
            row for _, row in self._df.loc[pending_mask].iterrows()
        )
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        lock = FileLock(str(self.path) + ".lock")
        with lock:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            self._df.to_csv(tmp, sep=self.delim, index=False)
            tmp.replace(self.path)
        return False

    # ---------------------------------------------------------------- iterator
    def __iter__(self) -> Iterator[ParamDict]:
        for row in self._iter:
            yield {
                k: v
                for k, v in row.items()
                if k not in (self.status_col, self.error_col)
            }

    # ---------------------------------------------------------------- length
    def __len__(self) -> int:
        if not hasattr(self, "_df"):
            raise RuntimeError("CsvSource must be used as a context manager")
        mask = ~self._df[self.status_col].isin(["done", "failed"])
        return int(mask.sum())

    # ---------------------------------------------------------------- hooks
    def reserve(self, ids: Sequence[str]):  # noqa: D401
        with self._lock:
            idxs = self._df[self._df[self.id_col].isin(ids)].index
            self._df.loc[idxs, self.status_col] = self.reserve_tag

    def mark_done(self, ids: Sequence[str]):  # noqa: D401
        with self._lock:
            idxs = self._df[self._df[self.id_col].isin(ids)].index
            self._df.loc[idxs, self.status_col] = "done"

    def mark_failed(self, ids: Sequence[str], exc: Exception):  # noqa: D401
        with self._lock:
            idxs = self._df[self._df[self.id_col].isin(ids)].index
            self._df.loc[idxs, self.status_col] = "failed"
            self._df.loc[idxs, self.error_col] = str(exc)
