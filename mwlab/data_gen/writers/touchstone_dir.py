# mwlab/data_gen/writers/touchstone_dir.py
"""
TouchstoneDirWriter – Writer, сохраняющий каждый TouchstoneData в отдельный файл
================================================================================
Пишет батч объектов :class:`mwlab.io.touchstone.TouchstoneData` в указанную
директорию.  Имя формируется шаблоном ``stem_format`` и порядковым номером.

Дополнительно Writer может обновлять словарь ``params`` внутри каждого
TouchstoneData объектa метаданными из ``meta_batch[i]`` перед сохранением.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Mapping, Sequence

from mwlab.data_gen.base import Batch, MetaBatch, Outputs, Writer
from mwlab.io.touchstone import TouchstoneData

__all__ = ["TouchstoneDirWriter"]


class TouchstoneDirWriter(Writer):
    """Сохраняет TouchstoneData в виде *.sNp*-файлов."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        root: str | Path,
        *,
        stem_format: str = "sample_{idx:06d}",
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stem_format = stem_format
        self.overwrite = overwrite

        self._counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write
    def write(self, outputs: Outputs, meta: MetaBatch, params: Batch):  # noqa: D401
        if not (len(outputs) == len(meta) == len(params)):
            raise ValueError("TouchstoneDirWriter: входные последовательности разной длины")

        # ВАЖНО: итерируем синхронно по outputs, meta, params — чтобы
        # извлекать __id текущего элемента батча (а не по глобальному индексу)
        for ts_obj, meta_dct, p_map in zip(outputs, meta, params):

            if not isinstance(ts_obj, TouchstoneData):
                raise TypeError("TouchstoneDirWriter ожидает TouchstoneData в outputs")
            if meta_dct:
                if not isinstance(meta_dct, Mapping):
                    raise TypeError("meta_batch должен содержать dict или быть пустым")
                ts_obj.params.update(meta_dct)  # дописываем метаданные

            # --- формируем уникальное имя ---
            with self._lock:
                idx = self._counter
                self._counter += 1
            # Поддерживаем stem_format с плейсхолдерами {idx} и (опционально) {id}
            item_id = ""
            try:
                item_id = str(p_map.get("__id", ""))
            except Exception:  # pragma: no cover
                item_id = ""
            try:
                stem = self.stem_format.format(idx=idx, id=item_id)
            except KeyError:
                # если в шаблоне нет {id} — подставляем только idx
                stem = self.stem_format.format(idx=idx)

            n_ports = ts_obj.network.number_of_ports
            fname = self.root / f"{stem}.s{n_ports}p"

            if fname.exists() and not self.overwrite:
                raise FileExistsError(fname)
            ts_obj.save(fname)

    # ------------------------------------------------------------------ flush
    def flush(self):  # noqa: D401
        """Файл не буферизуется – метод оставлен для совместимости."""
        pass
