#tests/test_touchstone_dataset_transforms.py
"""
Базовые тесты TouchstoneDataset + проверки трансформов.
"""

from __future__ import annotations

import numpy as np
import pytest
import skrf as rf
from torch.utils.data import DataLoader

from mwlab.io.backends import FileBackend
from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.transforms.x_transforms import X_SelectKeys
from mwlab.transforms.s_transforms import (
    S_Crop,
    S_Resample,
    S_AddNoise,
    S_PhaseShiftDelay,
    S_PhaseShiftAngle,
    S_DeReciprocal,
    S_Z0Shift,
    S_Ripple,
    S_MagSlope,
)
from mwlab.transforms import TComposite


# ---------- Фикстуры ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_info(sample_dir):
    """Вспомогательная информация по первым данным (параметры, частоты, число портов)."""
    ts = FileBackend(sample_dir).read(0)

    param_keys = list(ts.params.keys())
    selected_keys = param_keys[:3] if len(param_keys) >= 3 else param_keys

    fmin_hz, fmax_hz = ts.network.f[0], ts.network.f[-1]
    f_crop_min = fmin_hz + 0.25 * (fmax_hz - fmin_hz)
    f_crop_max = fmax_hz - 0.25 * (fmax_hz - fmin_hz)
    n_points = 50
    f_interp = rf.Frequency.from_f(
        f=np.linspace(f_crop_min, f_crop_max, n_points), unit="Hz"
    )

    return {
        "selected_keys": selected_keys,
        "f_crop_min": f_crop_min,
        "f_crop_max": f_crop_max,
        "f_interp": f_interp,
        "ports": ts.network.number_of_ports,
    }


def _copy_net(sample_dir) -> rf.Network:
    return FileBackend(sample_dir).read(0).network.copy()

# ---------- Общие тесты TouchstoneDataset ----------------------------------------------------------------
def test_getitem_basic(sample_dir):
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend)
    x, s = ds[0]
    assert isinstance(x, dict)
    assert isinstance(s, rf.Network)
    assert s.s.ndim == 3


def test_dataset_with_transforms(sample_dir, sample_info):
    backend = FileBackend(sample_dir)

    x_tf = TComposite([X_SelectKeys(sample_info["selected_keys"])])
    s_tf = TComposite(
        [
            S_Crop(
                f_start=sample_info["f_crop_min"], f_stop=sample_info["f_crop_max"]
            ),
            S_Resample(freq_or_n=sample_info["f_interp"]),
        ]
    )

    ds = TouchstoneDataset(
        backend,
        x_keys=sample_info["selected_keys"],
        x_tf=x_tf,
        s_tf=s_tf,
    )

    x, s = ds[0]
    assert set(x.keys()) == set(sample_info["selected_keys"])
    assert isinstance(s, rf.Network)
    assert s.s.shape[0] == 50
    assert s.s.shape[1] == sample_info["ports"] == s.s.shape[2]


def test_missing_params_are_nan(sample_dir):
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend, x_keys=["nonexistent_param"])
    x, _ = ds[0]
    assert np.isnan(x["nonexistent_param"])


def test_fallback_behavior(sample_dir):
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend)
    x, s = ds[0]
    assert all(isinstance(k, str) for k in x)
    assert isinstance(s, rf.Network)


def simple_collate_fn(batch):
    xs, ss = zip(*batch)
    return list(xs), list(ss)


def test_dataloader_workers(sample_dir):
    backend = FileBackend(sample_dir)
    ds = TouchstoneDataset(backend)

    loader = DataLoader(
        ds, batch_size=8, num_workers=2, collate_fn=simple_collate_fn
    )
    x_batch, s_batch = next(iter(loader))
    assert len(x_batch) == 8 and len(s_batch) == 8


# ---------- НОВЫЕ ТЕСТЫ: аугментация ---------------------------------------------------------------------
def _get_sample_network(sample_dir) -> rf.Network:
    """Берём копию первой сети из датасета."""
    return FileBackend(sample_dir).read(0).network.copy()


def test_addnoise_nochange(sample_dir):
    """При нулевых σ сеть остаётся неизменной."""
    net = _get_sample_network(sample_dir)
    tf = S_AddNoise(sigma_db=0.0, sigma_deg=0.0)
    out = tf(net)
    np.testing.assert_allclose(out.s, net.s)


def test_addnoise_changes(sample_dir):
    """Ненулевой шум должен изменить хотя бы один элемент."""
    net = _get_sample_network(sample_dir)
    tf = S_AddNoise(sigma_db=0.2, sigma_deg=5.0)
    out = tf(net)
    assert np.any(np.abs(out.s - net.s) > 0)


def test_phaseshiftdelay_fixed(sample_dir):
    """Фиксированная задержка τ задаёт известный фазовый множитель."""
    net = _get_sample_network(sample_dir)
    tau_ps = 10.0
    tf = S_PhaseShiftDelay(tau_ps=tau_ps)
    out = tf(net)

    tau = tau_ps * 1e-12
    phase = np.exp(-1j * 2 * np.pi * net.f * tau)[:, None, None]
    np.testing.assert_allclose(out.s, net.s * phase)


def test_phaseshiftangle_fixed(sample_dir):
    """Фиксированный угол φ должен умножать всю матрицу на e^{jφ}."""
    net = _get_sample_network(sample_dir)
    phi_deg = 30.0
    tf = S_PhaseShiftAngle(phi_deg=phi_deg)
    out = tf(net)

    phi = np.deg2rad(phi_deg)
    expected = net.s * np.exp(1j * phi)
    np.testing.assert_allclose(out.s, expected)

# ----------------------------------------------------------------- Reciprocity
def test_dereciprocal(sample_dir):
    net = _copy_net(sample_dir)
    tf = S_DeReciprocal(sigma_db=0.2)
    out = tf(net)
    # off‑diagonal элементы должны отличаться
    i, j = 0, 1
    assert np.any(np.abs(out.s[:, i, j] - out.s[:, j, i]) > 1e-6)


# ----------------------------------------------------------------- Z0‑shift
def test_z0shift_changes_impedance(sample_dir):
    net = _copy_net(sample_dir)
    orig_z0 = net.z0.copy()
    tf = S_Z0Shift(delta_ohm=2.0)
    out = tf(net)
    assert np.all(np.abs(out.z0 - orig_z0) >= 1.9)


# ----------------------------------------------------------------- Ripple
def test_ripple_variation(sample_dir):
    net = _copy_net(sample_dir)
    tf = S_Ripple(amp_db=0.5, period_hz=2e9)
    out = tf(net)
    ratio = np.abs(out.s) / np.abs(net.s)
    # дисперсия коэффициента > 0 указывает на ripple
    assert np.var(ratio) > 0


# ----------------------------------------------------------------- Slope
@pytest.mark.parametrize("k_db", [2.0, -2.0])
def test_magslope(sample_dir, k_db):
    net = _copy_net(sample_dir)
    tf = S_MagSlope(slope_db_per_ghz=k_db)
    out = tf(net)

    mag_orig = 20 * np.log10(np.abs(net.s[:, 0, 0]) + 1e-12)
    mag_new = 20 * np.log10(np.abs(out.s[:, 0, 0]) + 1e-12)
    slope_est = np.polyfit(net.f / 1e9, mag_new - mag_orig, 1)[0]

    assert np.sign(slope_est) == np.sign(k_db)
    assert abs(slope_est) > 0.5  # существенный наклон