# mwlab/mwfilter_lightning/touchstone_ldm.py
"""
TouchstoneLDataModule
=====================

Обертка Lightning **(PyTorch Lightning ≥ 2.1, импорт `mwfilter_lightning as L`)**
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
* дополнительно предоставляет публичные методы:
   - ``get_dataset(split="train", meta=False)``
   - ``get_dataloader(split="val", meta=True, shuffle=False)``

> **Внимание:** модуль не знает, какие именно тензоры внутри —
> эту ответственность берет на себя `TouchstoneCodec`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Sequence

import torch
from torch.utils.data import (
    Dataset, DataLoader, Subset
)
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
    Lightning‑обертка вокруг `TouchstoneTensorDataset`.

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

        # ------------- пользовательские аргументы ------------------------
        self.source = source
        self.codec = codec

        self.batch_size   = int(batch_size)
        self.num_workers  = int(num_workers)
        self.pin_memory   = bool(pin_memory)

        self.val_ratio    = float(val_ratio)
        self.test_ratio   = float(test_ratio)
        if self.val_ratio + self.test_ratio > 1.0:
            raise ValueError("val_ratio + test_ratio must be <= 1")

        self.max_samples  = max_samples
        self.seed         = int(seed)

        self.swap_xy      = bool(swap_xy)
        self.cache_size   = cache_size
        self.base_kwargs  = base_ds_kwargs or {}

        self.scaler_in    = scaler_in
        self.scaler_out   = scaler_out

        # ------------- будут заполнены в setup() -------------------------
        self.idx_train: List[int] | None = None
        self.idx_val:   List[int] | None = None
        self.idx_test:  List[int] | None = None

        self.train_ds:    Dataset | None = None
        self.val_ds:      Dataset | None = None
        self.test_ds:     Dataset | None = None
        self.predict_ds:  Dataset | None = None

    # ======================================================================
    #                         HELPERS
    # ======================================================================
    def _build_base(self, *, meta: bool) -> TouchstoneTensorDataset:
        """
        Конструирует *новый* TouchstoneTensorDataset
        с нужным флагом `return_meta`.
        """
        return TouchstoneTensorDataset(
            source       = self.source,
            codec        = self.codec,
            swap_xy      = self.swap_xy,
            return_meta  = meta,
            cache_size   = self.cache_size,
            base_kwargs  = self.base_kwargs,
        )

    def _view(self, idxs: Sequence[int], *, meta: bool) -> Dataset:
        """
        Возвращает `Subset(base_dataset, idxs)`.
        Базовый датасет создаётся с флагом `meta`.
        """
        return Subset(self._build_base(meta=meta), list(idxs))

    @staticmethod
    def _collate_with_meta(batch: List[tuple]
                           ) -> tuple[torch.Tensor, torch.Tensor, List[Dict[str,Any]]]:
        xs, ys, metas = zip(*batch)        # type: ignore
        return default_collate(xs), default_collate(ys), list(metas)

    def _fit_scaler(self, scaler: torch.nn.Module, ds: Dataset, *, take: str):
        """
        Однопроходный `scaler.fit` без загрузки всего ds в память.
        """
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
        with torch.no_grad():
            bufs = []
            for x, y in loader:
                bufs.append(x if take == "x" else y)
            scaler.fit(torch.cat(bufs, dim=0))

    # ======================================================================
    #                                 setup
    # ======================================================================
    def setup(self, stage: str | None = None):
        """
        * **fit** / `None` — создает split (индексы) + фит скейлеров;
        * **validate**/**test** — split уже готов; просто собираем view‑ы;
        * **predict** — полный датасет с meta.
        """

        # ------------------------ FIT -------------------------------------
        if stage in (None, "fit"):
            # ---- если split еще не строили → строим ----------------------
            if self.idx_train is None:
                base_no_meta = self._build_base(meta=False)

                # ограничение общего числа сэмплов
                if self.max_samples is not None:
                    base_no_meta = Subset(
                        base_no_meta,
                        range(min(len(base_no_meta), int(self.max_samples)))
                    )

                if len(base_no_meta) == 0:
                    raise ValueError("TouchstoneLDataModule: датасет пуст")

                # --------- random split (но сохраняем индексы!) ----------
                n_total = len(base_no_meta)
                n_val   = math.floor(n_total * self.val_ratio)
                n_test  = math.floor(n_total * self.test_ratio)
                n_train = n_total - n_val - n_test

                g = torch.Generator().manual_seed(self.seed)
                perm = torch.randperm(n_total, generator=g).tolist()

                self.idx_train = perm[:n_train]
                self.idx_val   = perm[n_train:n_train + n_val]
                self.idx_test  = perm[n_train + n_val:]

                # скейлеры считаем по train‑части
                if self.scaler_in or self.scaler_out:
                    tmp_train = self._view(self.idx_train, meta=False)
                    if self.scaler_in:
                        self._fit_scaler(self.scaler_in, tmp_train, take="x")
                    if self.scaler_out:
                        self._fit_scaler(self.scaler_out, tmp_train, take="y")

            # -------- собираем view‑датасеты (meta=False) ---------------
            self.train_ds = self._view(self.idx_train, meta=False)
            self.val_ds   = self._view(self.idx_val,   meta=False)
            self.test_ds  = self._view(self.idx_test,  meta=False)

        # ------------------------ VALIDATE / TEST -------------------------
        if stage in ("validate", "test"):
            if self.idx_train is None:
                raise RuntimeError("setup('fit') должен быть вызван до validate/test")

        # ------------------------ PREDICT ---------------------------------
        if stage == "predict" and self.predict_ds is None:
            # здесь нужен meta=True для auto_decode
            self.predict_ds = self._build_base(meta=True)

    # ======================================================================
    #                    PUBLIC helpers
    # ======================================================================
    def get_dataset(self, split: str = "train", *, meta: bool = False) -> Dataset:
        """
        Возвращает датасет нужного сплита.

        split ∈ {'train','val','test','full'}
        meta  – передавать ли Touchstone‑meta (нужно для codec.decode()).
        """
        mapping = {
            "train": self.idx_train,
            "val":   self.idx_val,
            "test":  self.idx_test,
            "full":  (self.idx_train or []) + (self.idx_val or []) + (self.idx_test or []),
        }
        if mapping[split] is None:
            raise RuntimeError("setup('fit') еще не выполнялся")
        return self._view(mapping[split], meta=meta)

    def get_dataloader(
        self,
        split: str = "train",
        *,
        meta: bool = False,
        shuffle: bool | None = None,
    ) -> DataLoader:
        """
        Быстро получить DataLoader для нужного сплита.
        shuffle по‑умолчанию включается *только* для train.
        """
        if shuffle is None:
            shuffle = (split == "train")
        ds = self.get_dataset(split, meta=meta)
        return self._loader(ds, shuffle=shuffle, with_meta=meta)

    # ======================================================================
    #                            loaders
    # ======================================================================
    def _loader(self, ds: Dataset, *, shuffle: bool = False, with_meta: bool = False):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn = (self._collate_with_meta if with_meta else None)
        )

    # -------- интерфейсы Lightning (оставлены для совместимости) ----------
    def train_dataloader(self):   return self._loader(self.train_ds, shuffle=True)
    def val_dataloader(self):     return self._loader(self.val_ds)
    def test_dataloader(self):    return self._loader(self.test_ds)
    def predict_dataloader(self): return self._loader(self.predict_ds, with_meta=True)

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return (f"{self.__class__.__name__}(source={self.source}, "
                f"samples={self.max_samples or 'all'}, batch={self.batch_size}, "
                f"val={self.val_ratio}, test={self.test_ratio}, swap_xy={self.swap_xy})")
