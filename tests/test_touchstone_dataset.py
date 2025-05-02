"""
Тесты TouchstoneDataset: базовая логика, трансформы, DataLoader
"""

from __future__ import annotations

import numpy as np
import pytest
import skrf as rf
from torch.utils.data import DataLoader

from mwlab.io.backends import FileBackend
from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.transforms.x_transforms import X_SelectKeys
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.transforms import TComposite


# ---------- Фикстуры ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_info(sample_dir):
    """Вспомогательная информация по первым данным (параметры, частоты, число портов)."""
    file = next(sample_dir.glob("*.s?p"))
    ts = FileBackend(sample_dir).read(0)

    param_keys = list(ts.params.keys())
    selected_keys = param_keys[:3] if len(param_keys) >= 3 else param_keys

    fmin_hz, fmax_hz = ts.network.f[0], ts.network.f[-1]
    f_crop_min = fmin_hz + 0.25 * (fmax_hz - fmin_hz)
    f_crop_max = fmax_hz - 0.25 * (fmax_hz - fmin_hz)
    n_points = 50
    f_interp = rf.Frequency.from_f(f=np.linspace(f_crop_min, f_crop_max, n_points), unit='Hz')

    return {
        "selected_keys": selected_keys,
        "f_crop_min": f_crop_min,
        "f_crop_max": f_crop_max,
        "f_interp": f_interp,
        "ports": ts.network.number_of_ports
    }


# ---------- Тесты ----------------------------------------------------------------------------------------

def test_getitem_basic(sample_dir):
    """Базовая проверка __getitem__ без трансформов."""
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend)
    x, s = ds[0]
    assert isinstance(x, dict)
    assert isinstance(s, rf.Network)
    assert s.s.ndim == 3


def test_dataset_with_transforms(sample_dir, sample_info):
    """Проверка работы x_tf и s_tf на реальных данных."""
    backend = FileBackend(sample_dir)

    x_tf = TComposite([
        X_SelectKeys(sample_info["selected_keys"]),
    ])
    s_tf = TComposite([
        S_Crop(f_start=sample_info["f_crop_min"],
               f_stop=sample_info["f_crop_max"]),
        S_Resample(freq_or_n=sample_info["f_interp"]),
    ])

    ds = TouchstoneDataset(
        backend,
        x_keys=sample_info["selected_keys"],
        x_tf=x_tf,
        s_tf=s_tf,
    )

    x, s = ds[0]
    assert isinstance(x, dict)
    assert set(x.keys()) == set(sample_info["selected_keys"])
    assert all(isinstance(v, float) or np.isnan(v) for v in x.values())

    assert isinstance(s, rf.Network)
    assert s.s.ndim == 3
    assert s.s.shape[0] == 50
    assert s.s.shape[1] == sample_info["ports"]
    assert s.s.shape[2] == sample_info["ports"]


def test_missing_params_are_nan(sample_dir):
    """Если параметр отсутствует — он должен быть np.nan"""
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend, x_keys=["nonexistent_param"])
    x, _ = ds[0]
    assert "nonexistent_param" in x
    assert np.isnan(x["nonexistent_param"])


def test_fallback_behavior(sample_dir):
    """Проверка без x_keys и трансформов — fallback-поведение."""
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend)
    x, s = ds[0]
    assert isinstance(x, dict)
    assert all(isinstance(k, str) for k in x)
    assert all(isinstance(v, (float, int, str, type(np.nan))) for v in x.values())
    assert isinstance(s, rf.Network)
    assert s.s.ndim == 3


def simple_collate_fn(batch):
    xs, ss = zip(*batch)
    return list(xs), list(ss)

def test_dataloader_workers(sample_dir):
    """Проверка, что DataLoader работает с num_workers > 0."""
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend)

    loader = DataLoader(ds, batch_size=8, num_workers=2, collate_fn=simple_collate_fn)
    x_batch, s_batch = next(iter(loader))

    assert len(x_batch) == 8
    assert len(s_batch) == 8
    assert isinstance(x_batch[0], dict)
    assert hasattr(s_batch[0], "s")
