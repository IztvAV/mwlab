# tests/test_touchstone_ldm.py
import pytest
from torch.utils.data import DataLoader

from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.lightning.touchstone_ldm import TouchstoneLDataModule
from mwlab.datasets.touchstone_tensor_dataset import TouchstoneTensorDataset
from mwlab.nn.scalers import StdScaler, MinMaxScaler

# ------------------------ fixtures ------------------------

@pytest.fixture(scope="module")
def codec(sample_dir) -> TouchstoneCodec:
    """Формируем Codec по сырому набору файлов (real+imag)."""
    raw = TouchstoneDataset(sample_dir)
    return TouchstoneCodec.from_dataset(raw)


# 1. setup + dataloaders ---------------------------------------------
def test_ldm_setup_and_dataloaders(sample_dir, codec):
    ldm = TouchstoneLDataModule(
        source=str(sample_dir),
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
def test_ldm_predict_stage(sample_dir, codec):
    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec, batch_size=2)
    ldm.setup("predict")
    assert isinstance(ldm.predict_dataloader(), DataLoader)


# 3. ограничение max_samples ----------------------------------------
def test_ldm_max_samples(sample_dir, codec):
    ldm = TouchstoneLDataModule(
        source=sample_dir,
        codec=codec,
        max_samples=3,
        batch_size=1,
    )
    ldm.setup("fit")
    total = sum(len(ds) for ds in (ldm.train_ds, ldm.val_ds, ldm.test_ds))
    assert total <= 3

# 4 ─ repr / str ----------------------------------------------------
def test_ldm_repr_contains_info(sample_dir, codec):
    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec, batch_size=4)
    ldm.setup('fit')
    txt = repr(ldm)
    for token in ("TouchstoneLDataModule", "batch=", "val=", "test="):
        assert token in txt


# 5 ─ validate / test без fit ----------------------------------------
def test_ldm_validate_error_before_fit(sample_dir, codec):
    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec)
    with pytest.raises(RuntimeError, match=r"setup\('fit'\) должно быть вызвано перед validate"):
        ldm.setup("validate")

def test_ldm_test_error_before_fit(sample_dir, codec):
    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec)
    with pytest.raises(RuntimeError, match=r"setup\('fit'\) должно быть вызвано перед test"):
        ldm.setup("test")


# 6-a. predict_ds (авто-fallback) ------------------------------------
def test_ldm_predict_ds_auto(sample_dir, codec):
    ldm = TouchstoneLDataModule(
        source=sample_dir,
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
def test_ldm_predict_ds_explicit(sample_dir, codec):
    explicit_ds = TouchstoneTensorDataset(source=sample_dir, codec=codec, return_meta=True)

    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec, val_ratio=0.1, test_ratio=0.1)
    ldm.setup("fit")

    ldm.predict_ds = explicit_ds
    ldm.setup("predict")

    assert ldm.predict_ds is explicit_ds

def test_ldm_swap_xy_shapes(sample_dir, codec):
    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec, swap_xy=True, batch_size=2)
    ldm.setup("fit")
    batch = next(iter(ldm.train_dataloader()))
    y, x = batch
    assert x.shape[-1] == len(codec.x_keys)
    assert y.shape[1] == len(codec.y_channels)  # (B, C, F)

def test_ldm_raises_on_empty_dataset(sample_dir, codec):
    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec, max_samples=0)
    with pytest.raises(ValueError, match="датасет пуст"):
        ldm.setup("fit")


def test_ldm_predict_returns_meta(sample_dir, codec):
    ldm = TouchstoneLDataModule(source=sample_dir, codec=codec, batch_size=2)
    ldm.setup("predict")
    batch = next(iter(ldm.predict_dataloader()))

    assert isinstance(batch, (tuple, list))
    assert len(batch) == 3  # (x, y, metas)

    metas = batch[2]
    assert isinstance(metas, list)
    assert all(isinstance(m, dict) for m in metas)
    assert all("params" in m for m in metas)

