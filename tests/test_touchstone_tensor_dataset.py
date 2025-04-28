# tests/test_touchstone_tensor_dataset.py
"""
Проверки TouchstoneTensorDataset и миксина _CachedDataset.

Потребуются «живые» *.sNp; путь передает фикстура `dataset_dir`

Покрываем:
    ✓ корректная длина и форма X / Y;
    ✓ swap_xy   — инверсная постановка;
    ✓ return_meta – наличие meta с путём и params;
    ✓ LRU-кэш (_CachedDataset): ограничение размера и переиспользование объектов.
"""

import pathlib
import pytest

from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.datasets.touchstone_tensor_dataset import TouchstoneTensorDataset
from mwlab.codecs.touchstone_codec import TouchstoneCodec

# ─────────────────────────────────────────────────────────────────────────────
#                                       FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset_dir() -> pathlib.Path:
    """Путь к директории с тестовыми *.sNp файлами."""
    return pathlib.Path(__file__).parent.parent / "Data" / "Filter12"


@pytest.fixture(scope="module")
def codec(dataset_dir):
    """Генерируем Codec по содержимому директории (real/imag, union x_keys)."""
    raw = TouchstoneDataset(dataset_dir)
    return TouchstoneCodec.from_dataset(raw)


@pytest.fixture(scope="module")
def ds_direct(dataset_dir, codec):
    """Прямая задача X ➜ Y (swap_xy=False)."""
    return TouchstoneTensorDataset(
        root=dataset_dir,
        codec=codec,
        swap_xy=False,
        return_meta=False,
        cache_size=0,      # кэш выключен
    )


# ─────────────────────────────────────────────────────────────────────────────
#                                       HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _assert_shapes(ds: TouchstoneTensorDataset, idx: int = 0):
    """Проверка размерностей X-вектора и Y-тензора."""
    x, y = ds[idx]
    C, F = y.shape
    assert x.ndim == 1
    assert x.numel() == len(ds.codec.x_keys)
    assert C == len(ds.codec.y_channels)
    assert F == len(ds.codec.freq_hz)


# ─────────────────────────────────────────────────────────────────────────────
#                                       TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_length_matches_filecount(ds_direct, dataset_dir):
    """
    Количество элементов в датасете соответствует количеству файлов *.sNp.
    """
    n_files = sum(1 for p in dataset_dir.rglob("*.s?p"))
    assert len(ds_direct) == n_files


def test_shapes_direct(ds_direct):
    """
    Прямая задача (swap_xy=False):
      X — вектор признаков;
      Y — тензор (C, F).
    """
    _assert_shapes(ds_direct)


def test_shapes_inverse(dataset_dir, codec):
    """
    Обратная задача (swap_xy=True):
      меняем местами X и Y.
    """
    ds_inv = TouchstoneTensorDataset(
        root=dataset_dir,
        codec=codec,
        swap_xy=True,
        return_meta=False,
        cache_size=0,
    )
    y, x = ds_inv[0]  # (Y, X) в обратном порядке
    assert y.shape[0] == len(codec.y_channels)
    assert x.ndim == 1 and x.numel() == len(codec.x_keys)


def test_return_meta(dataset_dir, codec):
    """
    Возвращение метаданных (return_meta=True):
      meta содержит путь и параметры.
    """
    ds_meta = TouchstoneTensorDataset(
        root=dataset_dir,
        codec=codec,
        return_meta=True,
        cache_size=0,
    )
    x, y, meta = ds_meta[0]

    assert "orig_path" in meta or "path" in meta, "Missing path info in meta"
    assert isinstance(meta["params"], dict), "Missing params dict in meta"
    assert set(meta["params"]).issuperset(codec.x_keys), "Params keys mismatch"


def test_cache_behaviour(dataset_dir, codec):
    """
    Поведение кэша (_CachedDataset):
      • cache_size=2 → в памяти не более 2 элементов;
      • повторный доступ к индексу → тот же объект.
    """
    ds_cache = TouchstoneTensorDataset(
        root=dataset_dir,
        codec=codec,
        cache_size=2,
        return_meta=False,
    )

    # первый доступ — кэш пуст
    x0, y0 = ds_cache[0]
    first_id = id(x0)

    # второй доступ к тому же idx — тот же объект
    x0b, y0b = ds_cache[0]
    assert id(x0b) == first_id

    # запросим ещё два индекса — кэш не превышает лимит
    _ = ds_cache[1]
    _ = ds_cache[2]
    assert len(ds_cache._cache) <= 2
