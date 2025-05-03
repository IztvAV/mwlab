# mwlab/lightning/touchstone_ldm.py

from pathlib import Path
import math, torch
from typing import Union
import lightning as L
from torch.utils.data import DataLoader, random_split, Subset

from mwlab.datasets.touchstone_tensor_dataset import TouchstoneTensorDataset
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.io.backends import StorageBackend

class TouchstoneLDataModule(L.LightningDataModule):
    """
    Lightning-обёртка вокруг TouchstoneTensorDataset.

    • Принимает готовый TouchstoneCodec  (централизуем всю «логику форматов»);
    • Поддерживает прямую и обратную задачу (swap_xy=True);
    • Возвращает meta-объект в predict-loader-е — удобно декодировать
      результат сразу в TouchstoneData внутри `predict_step`.
    """

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        source: Union[str, Path, StorageBackend],
        *,
        codec: TouchstoneCodec,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        max_samples: int | None = None,
        seed: int = 42,
        swap_xy: bool = False,
        cache_size: int | None = 0,
        scaler_in: torch.nn.Module | None = None,
        scaler_out: torch.nn.Module | None = None,
        base_ds_kwargs: dict | None = None,
    ):
        super().__init__()
        self.source          = source
        self.codec         = codec
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.pin_memory    = pin_memory
        self.val_ratio     = val_ratio
        self.test_ratio    = test_ratio
        self.max_samples   = max_samples
        self.seed          = seed
        self.swap_xy       = swap_xy
        self.cache_size    = cache_size
        self.scaler_in     = scaler_in
        self.scaler_out    = scaler_out
        self.base_kwargs   = base_ds_kwargs or {}

        self.train_ds:   torch.utils.data.Dataset | None = None
        self.val_ds:     torch.utils.data.Dataset | None = None
        self.test_ds:    torch.utils.data.Dataset | None = None
        self.predict_ds: torch.utils.data.Dataset | None = None

    # -------------------------------------------------------------- private
    def _make_dataset(self, *, return_meta: bool) -> TouchstoneTensorDataset:
        return TouchstoneTensorDataset(
            source         = self.source,
            codec        = self.codec,
            swap_xy      = self.swap_xy,
            return_meta  = return_meta,
            cache_size   = self.cache_size,
            base_kwargs  = self.base_kwargs,
        )

    # ---------------------------------------------------------------- setup
    def setup(self, stage: str | None = None):
        full_ds = self._make_dataset(return_meta=False)

        # ограничиваем размер набора (для быстрых экспериментальных запусков)
        if self.max_samples is not None:
            full_ds = Subset(full_ds, list(range(min(len(full_ds),
                                                   int(self.max_samples)))))

        if stage in ("fit", None):
            n_total = len(full_ds)
            n_val   = math.floor(n_total * self.val_ratio)
            n_test  = math.floor(n_total * self.test_ratio)
            n_train = n_total - n_val - n_test
            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_ds,
                lengths=[n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # ------ скейлеры ------------------------------------------------
            if self.scaler_in is not None:
                xs = torch.stack([x for x, _ in self.train_ds])
                self.scaler_in.fit(xs)
            if self.scaler_out is not None:
                ys = torch.stack([y for _, y in self.train_ds])
                self.scaler_out.fit(ys)

        # ---------- validate / test: проверяем, что наборы уже есть -------
        elif stage == "validate":
            if self.val_ds is None:
                raise RuntimeError("val_ds не инициализирован. Сначала вызовите setup('fit').")
        elif stage == "test":
            if self.test_ds is None:
                raise RuntimeError("test_ds не инициализирован. Сначала вызовите setup('fit').")

        # ---------- predict ----------------------------------------------
        elif stage == "predict":
            # отдельный набор, в котором meta нужны
            if self.predict_ds is None:
                self.predict_ds = self._make_dataset(return_meta=True)

    # ----------------------------------------------------------- dataloaders
    def _loader(self, ds, shuffle=False):
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)

    def predict_dataloader(self):
        return self._loader(self.predict_ds)

    # ---------------------------------------------------------------- view
    def __repr__(self):
        return (f"{self.__class__.__name__}(source={self.source}, swap_xy={self.swap_xy}, "
                f"batch={self.batch_size}, val_ratio={self.val_ratio}, "
                f"test_ratio={self.test_ratio})")
