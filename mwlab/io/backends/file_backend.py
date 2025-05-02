#mwlab/io/backends/file_backend.py
"""
FileBackend – реализация «один Touchstone-файл на запись».

Это текущее (старое) поведение, теперь оформлено через общий интерфейс.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

from mwlab.io.touchstone import TouchstoneData
from .base import StorageBackend


class FileBackend(StorageBackend):
    """Backend, который хранит набор путей к *.sNp."""

    def __init__(self, root: str | pathlib.Path, pattern: str = "*.s?p"):
        self.root = pathlib.Path(root)
        self.paths: list[pathlib.Path] = sorted(self.root.rglob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"В каталоге {root!s} нет файлов {pattern}")

    # -------------------------- API StorageBackend
    def __len__(self) -> int:  # noqa: D401
        return len(self.paths)

    def read(self, idx: int) -> TouchstoneData:  # noqa: D401
        return TouchstoneData.load(self.paths[idx])

    # -------------------------- запись новой выборки
    def append(self, ts: TouchstoneData) -> None:
        """
        Сохраняем TouchstoneData в ту же директорию *root* под уникальным именем.
        Имя формируем по шаблону «sample_{idx}.sNp».
        """
        next_idx = len(self.paths)
        suffix = f"s{ts.network.number_of_ports}p"

        # Поиск первого свободного имени
        while True:
            fname = self.root / f"sample_{next_idx:06d}.{suffix}"
            if not fname.exists():
                break
            next_idx += 1

        ts.save(fname)
        self.paths.append(fname)
