# mwlab/lightning/touchstone_ldm.py

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split, Subset
from mwlab.datasets import TouchstoneTensorDataset

class TouchstoneLDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        max_samples: int | None = None,
        seed: int = 42,
        scaler_in=None,
        scaler_out=None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_samples = max_samples
        self.seed = seed
        self.scaler_in = scaler_in
        self.scaler_out = scaler_out
        self.dataset_kwargs = dataset_kwargs

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.predict_ds = None

    def setup(self, stage: str):
        full_dataset = TouchstoneTensorDataset(self.root, **self.dataset_kwargs)

        if self.max_samples is not None:
            full_dataset = Subset(full_dataset, list(range(min(self.max_samples, len(full_dataset)))))

        if stage == "fit" or stage is None:
            total = len(full_dataset)
            n_val = int(total * self.val_ratio)
            n_test = int(total * self.test_ratio)
            n_train = total - n_val - n_test

            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_dataset,
                lengths=[n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # Fit скейлеров по train
            if self.scaler_in:
                x_tensors = torch.stack([x for x, _ in self.train_ds])
                self.scaler_in.fit(x_tensors)
            if self.scaler_out:
                y_tensors = torch.stack([y for _, y in self.train_ds])
                self.scaler_out.fit(y_tensors)

        elif stage == "validate":
            if self.val_ds is None:
                raise RuntimeError("val_ds не инициализирован. Сначала вызовите setup('fit').")

        elif stage == "test":
            if self.test_ds is None:
                raise RuntimeError("test_ds не инициализирован. Сначала вызовите setup('fit').")

        elif stage == "predict":
            # fallback: если не задана predict_ds, используем test_ds
            if self.predict_ds is None:
                self.predict_ds = self.test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def __repr__(self):
        return (f"{self.__class__.__name__}(root={self.root}, batch_size={self.batch_size}, "
                f"scaler_in={self.scaler_in}, scaler_out={self.scaler_out}, "
                f"val_ratio={self.val_ratio}, test_ratio={self.test_ratio})")
