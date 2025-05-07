# mwlab/lightning/touchstone_ldm.py
"""
TouchstoneLDataModule
=====================

Обертка Lightning **(PyTorch Lightning ≥ 2.1, импорт `lightning as L`)**
над `TouchstoneTensorDataset`.  Датамодуль решает всю «рутину» вокруг
*.sNp/Touchstone*‑файлов:

* читает данные из каталога *.sNp*, контейнера HDF5 или собственного
  `StorageBackend`;
* разбивает набор на **train/val/test** (ratio или фиксированные длины);
* при желании ограничивает число примеров (`max_samples`) — удобно для быстрых
  экспериментов;
* умеет работать в «прямой» *(X → Y)* и «обратной» *(Y → X)* постановке
  (`swap_xy=True`);
* по запросу подготавливает **скейлеры** (`scaler_in`, `scaler_out`);
* в режиме *predict* отдает `meta`, чтобы
  `LightningModule.predict_step()` мог сразу вызвать
  `TouchstoneCodec.decode()`.

> **Внимание:** модуль не знает, какие именно тензоры внутри —
> эту ответственность берет на себя `TouchstoneCodec`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data._utils.collate import default_collate

import lightning as L

from mwlab.datasets.touchstone_tensor_dataset import TouchstoneTensorDataset
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.io.backends import StorageBackend


# ─────────────────────────────────────────────────────────────────────────────
#                            TouchstoneLDataModule
# ─────────────────────────────────────────────────────────────────────────────
class TouchstoneLDataModule(L.LightningDataModule):
    """
    Lightning‑обёртка вокруг `TouchstoneTensorDataset`.

    Параметры
    ---------
    source : str | pathlib.Path | StorageBackend
        Каталог *.sNp*, файл .h5/LMDB или уже инициализированный backend.
    codec : TouchstoneCodec
        Отвечает за преобразование TouchstoneData ↔ тензоры.
    batch_size : int, default=32
        Размер батча для всех даталоадеров.
    num_workers : int, default=0
        Количество воркеров `DataLoader`.
    pin_memory : bool, default=True
        Передавать ли тензоры сразу в CUDA pinned memory.
    val_ratio : float, default=0.2
        Доля валидационного набора.
    test_ratio : float, default=0.1
        Доля тестового набора.
    max_samples : int | None
        Ограничить общее число сэмплов (None → использовать все).
    seed : int, default=42
        Фиксируем split для воспроизводимости.
    swap_xy : bool, default=False
        Переставлять ли X и Y (обратная задача Y → X).
    cache_size : int | None, default=0
        LRU‑кэш у `TouchstoneTensorDataset` (0 → кэш выкл., None → безлимит).
    scaler_in / scaler_out : torch.nn.Module | None
        Скейлеры для входа / выхода. Датамодуль умеет сам вызывать
        `scaler.fit()` (см. ниже).
    base_ds_kwargs : dict | None
        Дополнительные аргументы, которые будут проброшены в базовый
        `TouchstoneDataset` (например `{"s_tf": ..., "x_keys": [...]}`).
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        source: Union[str, Path, StorageBackend],
        *,
        codec: TouchstoneCodec,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        max_samples: Optional[int] = None,
        seed: int = 42,
        swap_xy: bool = False,
        cache_size: Optional[int] = 0,
        scaler_in: Optional[torch.nn.Module] = None,
        scaler_out: Optional[torch.nn.Module] = None,
        base_ds_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        # ---- user‑defined --------------------------------------------------
        self.source = source
        self.codec = codec

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        if self.val_ratio + self.test_ratio > 1.0:
            raise ValueError("val_ratio + test_ratio must be <= 1.0")

        self.max_samples = max_samples
        self.seed = int(seed)

        self.swap_xy = bool(swap_xy)
        self.cache_size = cache_size
        self.base_kwargs = base_ds_kwargs or {}

        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

        # ---- will be filled in setup() ------------------------------------
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self.predict_ds: Optional[Dataset] = None

    # ---------------------------------------------------------------- utils
    def _make_dataset(self, *, return_meta: bool) -> TouchstoneTensorDataset:
        """Фабрика датасета с нужными флагами."""
        return TouchstoneTensorDataset(
            source=self.source,
            codec=self.codec,
            swap_xy=self.swap_xy,
            return_meta=return_meta,
            cache_size=self.cache_size,
            base_kwargs=self.base_kwargs,
        )

    # custom collate that keeps meta safe ---------------------------------
    @staticmethod
    def _collate_with_meta(batch: List[tuple]) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        xs, ys, metas = zip(*batch)  # type: ignore
        x_coll = default_collate(xs)
        y_coll = default_collate(ys)
        return x_coll, y_coll, list(metas)  # список длиной batch

    def _fit_scaler(
        self, scaler: torch.nn.Module, ds: Dataset, *, take: str
    ) -> None:
        """
        Итеративно подгоняет скейлер без загрузки всего датасета в память.

        Parameters
        ----------
        scaler  : объект с методом `.fit(tensor)`
        ds : выборка (обычно train_ds)
        take    : 'x' | 'y' — какую часть батча использовать
        """
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
        with torch.no_grad():
            bufs: List[torch.Tensor] = []
            for b in loader:
                bufs.append(b[0] if take == "x" else b[1])
            data = torch.cat(bufs, dim=0)
            scaler.fit(data)

    # ================================================================= setup
    def setup(self, stage: str | None = None):
        """
        * **fit** / `None` — строит train/val/test + фит скейлеров;
        * **validate**/**test** — наборы должны быть готовы после fit;
        * **predict** — отдельный датасет с meta.
        """
        if stage in ("fit", None):
            full_ds: Dataset = self._make_dataset(return_meta=False)

            # ограничиваем количество примеров (debug/fast_dev)
            if self.max_samples is not None:
                full_ds = Subset(
                    full_ds, range(min(len(full_ds), int(self.max_samples)))
                )

            # защита «пустой датасет»
            if len(full_ds) == 0:
                raise ValueError("TouchstoneLDataModule: датасет пуст")

            # — train / val / test split
            n_total = len(full_ds)
            n_val = math.floor(n_total * self.val_ratio)
            n_test = math.floor(n_total * self.test_ratio)
            n_train = n_total - n_val - n_test

            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_ds,
                lengths=[n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # — подгоняем скейлеры (считаем статистики)
            if self.scaler_in is not None:
                self._fit_scaler(self.scaler_in, self.train_ds, take="x")

            if self.scaler_out is not None:
                self._fit_scaler(self.scaler_out, self.train_ds, take="y")

        # validate/test вызываются после fit → наборы уже существуют
        if stage == "validate" and self.val_ds is None:
            raise RuntimeError("setup('fit') должно быть вызвано перед validate")

        if stage == "test" and self.test_ds is None:
            raise RuntimeError("setup('fit') должно быть вызвано перед test")

        # — predict
        if stage == "predict" and self.predict_ds is None:
            self.predict_ds = self._make_dataset(return_meta=True)


    # ================================================================= loaders
    def _loader(self, ds: Dataset, *, shuffle: bool = False, with_meta: bool = False):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_with_meta if with_meta else None,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)

    def predict_dataloader(self):
        return self._loader(self.predict_ds, with_meta=True)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}(source={self.source}, "
            f"samples={self.max_samples or 'all'}, "
            f"batch={self.batch_size}, "
            f"val={self.val_ratio}, test={self.test_ratio}, "
            f"swap_xy={self.swap_xy})"
        )