# tests/test_touchstone_tensor_dataset.py
import pathlib
import pytest
import torch
import numpy as np

from mwlab import TouchstoneTensorDataset


@pytest.fixture(scope="module")
def dataset_dir():
    """Путь к директории с тестовыми *.sNp файлами."""
    return pathlib.Path(__file__).parent.parent / "Data" / "Filter12"


# ---------------------------------------------------------------------------
# 1. Базовая работоспособность и размеры тензоров
# ---------------------------------------------------------------------------

def test_basic_shapes(dataset_dir):
    ds = TouchstoneTensorDataset(dataset_dir)
    assert len(ds) > 0

    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == (len(ds.x_keys),)
    assert y.shape[0] == len(ds.y_channels)
    assert y.shape[1] == ds._freq_len


# ---------------------------------------------------------------------------
# 2. Автогенерация каналов «components → y_channels»
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("components", [["real", "imag"], ["db"], ["mag", "deg"]])
def test_components_autogeneration(dataset_dir, components):
    ds = TouchstoneTensorDataset(dataset_dir, components=components, y_channels=None)
    n_ports = ds._n_ports
    expected_c = len(components) * n_ports * n_ports
    assert len(ds.y_channels) == expected_c
    _, y = ds[0]
    assert y.shape[0] == expected_c


# ---------------------------------------------------------------------------
# 3. y_channels имеет приоритет над components
# ---------------------------------------------------------------------------

def test_y_channels_override(dataset_dir):
    custom = ["S11.real", "S12.imag"]
    ds = TouchstoneTensorDataset(dataset_dir, y_channels=custom, components=["db", "deg"])
    assert ds.y_channels == custom
    _, y = ds[0]
    assert y.shape[0] == len(custom)


# ---------------------------------------------------------------------------
# 4. Проверка формата y_channels
# ---------------------------------------------------------------------------

def test_y_channels_format(dataset_dir):
    ds = TouchstoneTensorDataset(dataset_dir)
    for tag in ds.y_channels:
        assert tag.startswith("S") and "." in tag
        i, j, comp = tag[1], tag[2], tag[4:]
        assert i.isdigit() and j.isdigit()
        assert comp in {"real", "imag", "db", "mag", "deg"}


# ---------------------------------------------------------------------------
# 5. Проверка значений Y: S11.real
# ---------------------------------------------------------------------------

def test_y_tensor_values(dataset_dir):
    ds = TouchstoneTensorDataset(dataset_dir, y_channels=["S11.real"])
    _, y = ds[0]
    _, net = ds._base[0]
    expected = net.s[:, 0, 0].real
    torch.testing.assert_close(y[0], torch.tensor(expected, dtype=torch.float32))


# ---------------------------------------------------------------------------
# 6. LRU-кэш (ограничение размера + вытеснение)
# ---------------------------------------------------------------------------

def test_lru_cache(dataset_dir):
    ds = TouchstoneTensorDataset(dataset_dir, cache_size=2)
    for i in range(min(4, len(ds))):
        _ = ds[i]
    assert len(ds._cache) <= 2

# ---------------------------------------------------------------------------
# 7. __repr__ / __str__ содержат ключевую информацию
# ---------------------------------------------------------------------------

def test_repr_str(dataset_dir):
    ds = TouchstoneTensorDataset(dataset_dir, cache_size=1)
    txt = str(ds)
    assert all(tok in txt for tok in ["samples=", "x_dim=", "y_shape=", "cache="])


# ---------------------------------------------------------------------------
# 8. Ошибка при пустой директории
# ---------------------------------------------------------------------------

def test_empty_dataset_dir(tmp_path):
    with pytest.raises(ValueError, match="Пустой TouchstoneDataset"):
        _ = TouchstoneTensorDataset(tmp_path)

