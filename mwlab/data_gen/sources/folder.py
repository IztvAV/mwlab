# mwlab/data_gen/sources/folder.py
"""
FolderSource – ParamSource, читающий файлы с параметрами из директории
=====================================================================
Каждый файл (``*.json``, ``*.yaml``) внутри каталога описывает **одну** точку
параметров.  Такой формат удобен, когда результаты другого инструмента уже
разложены по файлам или когда точки докидываются «на лету» сторонним процессом.

Особенности реализации под новый контракт
-----------------------------------------
* **Уникальный id**:  ``__id`` = относительный путь файла (str).  Это надёжно
  уникально в пределах каталога и остаётся понятным для логов.
* **Горячее пополнение**: при ``refresh=True`` каталог сканируется при каждом
  обращении, иначе – однократный начальный список.
* **Автопереезд**: после ``mark_done`` / ``mark_failed`` файл перемещается в
  подпапку *done/* или *failed/* (можно отключить флагом ``move_*``).
* **Потокобезопасность** через ``threading.Lock``; в многопроцессной среде
  возможны гонки на файловой системе, но они решаются за счёт атомарного
  ``Path.replace`` (если цель уже существует – выбираем уникальное имя
  ``"name__1.json"`` и пытаемся снова).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Iterator, Mapping, MutableMapping, Sequence

import yaml

from mwlab.data_gen.base import ParamDict, ParamSource

__all__ = ["FolderSource"]


# ---------------------------------------------------------------------------
# internal helper: atomic, collision‑safe rename
# ---------------------------------------------------------------------------

def _safe_replace(src: Path, dst: Path):
    """Пытается `Path.replace`, при коллизии добавляет суффикс ``__n``."""
    attempt = 0
    target = dst
    while True:
        try:
            src.replace(target)
            break
        except FileExistsError:  # pragma: no cover
            attempt += 1
            target = dst.with_name(f"{dst.stem}__{attempt}{dst.suffix}")


# ---------------------------------------------------------------------------
# FolderSource implementation
# ---------------------------------------------------------------------------

class FolderSource(ParamSource):
    """Читает параметры из набора JSON/YAML–файлов в директории."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        root: str | Path,
        *,
        pattern: str = "*.json",
        move_done: bool = True,
        move_failed: bool = True,
        done_dir: str = "done",
        failed_dir: str = "failed",
        refresh: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.pattern = pattern
        self.move_done = move_done
        self.move_failed = move_failed
        self.done_dir = self.root / done_dir
        self.failed_dir = self.root / failed_dir
        self.refresh = refresh

        self._queue: Deque[Path] = deque()
        self._lock = threading.Lock()
        self._id_to_path: dict[str, Path] = {}
        self._seen_paths = set()

    # ------------------------------------------------------------------ utils
    def _scan(self):
        """Сканирует каталог и кладёт новые файлы в очередь."""
        for f in sorted(self.root.glob(self.pattern)):
            if f in self._seen_paths:
                continue
            self._queue.append(f)
            self._seen_paths.add(f)

    @staticmethod
    def _read_file(path: Path) -> MutableMapping[str, Any]:
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(path.read_text())  # type: ignore[return-value]
        return json.loads(path.read_text())  # type: ignore[return-value]

    # ---------------------------------------------------------------- context
    def __enter__(self):  # noqa: D401
        self._scan()
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False

    # ---------------------------------------------------------------- iterator
    def __iter__(self) -> Iterator[ParamDict]:
        while True:
            with self._lock:
                if not self._queue:
                    if not self.refresh:
                        break  # список фиксирован
                    self._scan()
                    if not self._queue:
                        break  # всё равно пусто → конец
                try:
                    path = self._queue.popleft()
                except IndexError:
                    break
            id_ = str(path.relative_to(self.root))
            self._id_to_path[id_] = path
            try:
                payload = self._read_file(path)
                if not isinstance(payload, MutableMapping):
                    raise ValueError("file must contain a mapping at top level")
                payload = dict(payload)  # гарантируем mutability
                payload["__id"] = id_
                payload["__path"] = str(path)
                yield payload  # <- отдаём Runner-у
            except Exception as e:
                # сразу маркируем failed
                self.mark_failed([id_], e)

    # ---------------------------------------------------------------- length
    def __len__(self) -> int:  # noqa: D401
        with self._lock:
            self._scan()
            return len(self._queue)

    # ---------------------------------------------------------------- hooks
    def reserve(self, ids: Sequence[str]):  # noqa: D401, WPS110
        # локальная файловая система – резервирование не требуется
        pass

    def mark_done(self, ids: Sequence[str]):  # noqa: D401, WPS110
        if not self.move_done:
            return
        self.done_dir.mkdir(exist_ok=True)
        for id_ in ids:
            p = self._id_to_path.get(id_)
            if p and p.exists():
                _safe_replace(p, self.done_dir / p.name)

    def mark_failed(self, ids: Sequence[str], exc: Exception):
        if not self.move_failed:
            return
        self.failed_dir.mkdir(exist_ok=True)
        for id_ in ids:
            p = self._id_to_path.get(id_)
            if p is None:
                p = (self.root / id_)  # fallback
            if p.exists():
                _safe_replace(p, self.failed_dir / p.name)
