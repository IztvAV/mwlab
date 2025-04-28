# tests/test_touchstone_ldm.py

import pathlib
import pytest
from torch.utils.data import DataLoader

from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.lightning.touchstone_ldm import TouchstoneLDataModule
from mwlab.datasets.touchstone_tensor_dataset import TouchstoneTensorDataset
from mwlab.nn.scalers import StdScaler, MinMaxScaler

# ------------------------ fixtures ------------------------
@pytest.fixture(scope="module")
def dataset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "Data" / "Filter12"


@pytest.fixture(scope="module")
def codec(dataset_dir) -> TouchstoneCodec:
    """Формируем Codec по сырому набору файлов (real+imag)."""
    raw = TouchstoneDataset(dataset_dir)
    return TouchstoneCodec.from_dataset(raw)


# 1. setup + dataloaders ---------------------------------------------
def test_ldm_setup_and_dataloaders(dataset_dir, codec):
    ldm = TouchstoneLDataModule(
        root=str(dataset_dir),
        codec=codec,
        batch_size=4,
        val_ratio=0.2,
        test_ratio=0.1,
        scaler_in=StdScaler(dim=0),
        scaler_out=MinMaxScaler(dim=(0, 1)),
    )
    ldm.setup("fit")

    # наборы созданы
    assert ldm.train_ds and ldm.val_ds and ldm.test_ds

    # корректные загрузчики
    assert isinstance(ldm.train_dataloader(), DataLoader)
    assert isinstance(ldm.val_dataloader(), DataLoader)
    assert isinstance(ldm.test_dataloader(), DataLoader)

    # у скейлеров появились параметры после fit()
    assert hasattr(ldm.scaler_in, "mean")
    assert hasattr(ldm.scaler_out, "data_range")


# 2. predict_dataloader --------------------------------------------
def test_ldm_predict_stage(dataset_dir, codec):
    ldm = TouchstoneLDataModule(root=dataset_dir, codec=codec, batch_size=2)
    ldm.setup("predict")
    assert isinstance(ldm.predict_dataloader(), DataLoader)


# 3. ограничение max_samples ----------------------------------------
def test_ldm_max_samples(dataset_dir, codec):
    ldm = TouchstoneLDataModule(
        root=dataset_dir,
        codec=codec,
        max_samples=3,
        batch_size=1,
    )
    ldm.setup("fit")
    total = sum(len(ds) for ds in (ldm.train_ds, ldm.val_ds, ldm.test_ds))
    assert total <= 3

# 4 ─ repr / str ----------------------------------------------------
def test_ldm_repr_contains_info(dataset_dir, codec):
    ldm = TouchstoneLDataModule(root=dataset_dir, codec=codec, batch_size=4)
    txt = repr(ldm)
    for token in ("TouchstoneLDataModule", "batch=", "val_ratio=", "test_ratio="):
        assert token in txt


# 5 ─ validate / test без fit ----------------------------------------
def test_ldm_validate_error_before_fit(dataset_dir, codec):
    ldm = TouchstoneLDataModule(root=dataset_dir, codec=codec)
    with pytest.raises(RuntimeError, match="val_ds не инициализирован"):
        ldm.setup("validate")

def test_ldm_test_error_before_fit(dataset_dir, codec):
    ldm = TouchstoneLDataModule(root=dataset_dir, codec=codec)
    with pytest.raises(RuntimeError, match="test_ds не инициализирован"):
        ldm.setup("test")


# 6-a. predict_ds (авто-fallback) ------------------------------------
def test_ldm_predict_ds_auto(dataset_dir, codec):
    ldm = TouchstoneLDataModule(
        root=dataset_dir,
        codec=codec,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    ldm.setup("fit")      # создаёт train/val/test
    ldm.setup("predict")  # создаёт *новый* датасет с meta

    assert ldm.predict_ds is not None
    # теперь predict_ds должен быть отдельным объектом
    assert ldm.predict_ds is not ldm.test_ds


# 6-b ─ predict_ds (явно задан) ------------------------------------
def test_ldm_predict_ds_explicit(dataset_dir, codec):
    explicit_ds = TouchstoneTensorDataset(root=dataset_dir, codec=codec, return_meta=True)

    ldm = TouchstoneLDataModule(root=dataset_dir, codec=codec, val_ratio=0.1, test_ratio=0.1)
    ldm.setup("fit")

    ldm.predict_ds = explicit_ds
    ldm.setup("predict")

    assert ldm.predict_ds is explicit_ds

