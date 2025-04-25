# tests/test_touchstone_ldm.py

import pytest
from torch.utils.data import DataLoader

from mwlab.lightning.touchstone_ldm import TouchstoneLDataModule
from mwlab.nn.scalers import StdScaler, MinMaxScaler


@pytest.fixture(scope="module")
def dataset_dir():
    """Путь к директории с тестовыми *.sNp файлами."""
    import pathlib
    return pathlib.Path(__file__).parent.parent / "Data" / "Filter12"


# ---------------------------------------------------------------------------
# 1. Базовая проверка setup + train/val/test
# ---------------------------------------------------------------------------

def test_ldm_setup_and_dataloaders(dataset_dir):
    ldm = TouchstoneLDataModule(
        root=str(dataset_dir),
        batch_size=4,
        val_ratio=0.2,
        test_ratio=0.1,
        scaler_in=StdScaler(dim=0),
        scaler_out=MinMaxScaler(dim=(0, 1)),
    )

    ldm.setup("fit")

    # Наборы должны быть инициализированы
    assert ldm.train_ds is not None
    assert ldm.val_ds is not None
    assert ldm.test_ds is not None

    # Проверка загрузчиков
    assert isinstance(ldm.train_dataloader(), DataLoader)
    assert isinstance(ldm.val_dataloader(), DataLoader)
    assert isinstance(ldm.test_dataloader(), DataLoader)

    # Проверка скейлеров
    assert hasattr(ldm.scaler_in, "mean")
    assert hasattr(ldm.scaler_out, "data_range")


# ---------------------------------------------------------------------------
# 2. Проверка predict_dataloader
# ---------------------------------------------------------------------------

def test_ldm_predict_stage(dataset_dir):
    ldm = TouchstoneLDataModule(root=str(dataset_dir), batch_size=2)
    ldm.setup("predict")
    loader = ldm.predict_dataloader()
    assert isinstance(loader, DataLoader)

# ---------------------------------------------------------------------------
# 3. Проверка ограничения по max_samples
# ---------------------------------------------------------------------------

def test_ldm_max_samples(dataset_dir):
    ldm = TouchstoneLDataModule(
        root=str(dataset_dir),
        max_samples=3,
        batch_size=1,
    )
    ldm.setup("fit")
    total = sum(len(ds) for ds in (ldm.train_ds, ldm.val_ds, ldm.test_ds))
    assert total <= 3


# ---------------------------------------------------------------------------
# 4. Проверка __repr__ / __str__
# ---------------------------------------------------------------------------

def test_ldm_repr_contains_info(dataset_dir):
    ldm = TouchstoneLDataModule(
        root=str(dataset_dir),
        batch_size=4,
        val_ratio=0.2,
        test_ratio=0.1,
    )
    txt = repr(ldm)
    assert "TouchstoneLDataModule" in txt
    assert "batch_size=" in txt
    assert "val_ratio=" in txt
    assert "test_ratio=" in txt


# ---------------------------------------------------------------------------
# 5. Ошибка при вызове test/validate без fit
# ---------------------------------------------------------------------------

def test_ldm_validate_error_before_fit(dataset_dir):
    ldm = TouchstoneLDataModule(root=str(dataset_dir))
    with pytest.raises(RuntimeError, match="val_ds не инициализирован"):
        ldm.setup("validate")

def test_ldm_test_error_before_fit(dataset_dir):
    ldm = TouchstoneLDataModule(root=str(dataset_dir))
    with pytest.raises(RuntimeError, match="test_ds не инициализирован"):
        ldm.setup("test")


# ---------------------------------------------------------------------------
# 6. Поведение predict_ds: fallback и явное задание
# ---------------------------------------------------------------------------

def test_ldm_predict_ds_fallback(dataset_dir):
    ldm = TouchstoneLDataModule(
        root=str(dataset_dir),
        val_ratio=0.2,
        test_ratio=0.2,
    )
    ldm.setup("fit")
    ldm.setup("predict")

    # fallback: predict_ds должен совпадать с test_ds
    assert ldm.predict_ds is ldm.test_ds


def test_ldm_predict_ds_explicit(dataset_dir):
    from mwlab.datasets import TouchstoneTensorDataset

    ldm = TouchstoneLDataModule(
        root=str(dataset_dir),
        val_ratio=0.1,
        test_ratio=0.1,
    )
    ldm.setup("fit")

    # вручную задаём predict_ds
    explicit_ds = TouchstoneTensorDataset(dataset_dir)
    ldm.predict_ds = explicit_ds

    ldm.setup("predict")  # должно сохранить ручной выбор

    assert ldm.predict_ds is explicit_ds
    assert ldm.predict_ds is not ldm.test_ds
