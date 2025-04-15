
import numpy as np
import tempfile
import pytest
from mwlab.fileio import load_touchstone, save_touchstone

@pytest.mark.parametrize("format", ["RI", "MA", "DB"])
@pytest.mark.parametrize("unit", ["HZ", "MHZ", "GHZ"])
def test_touchstone_save_load_roundtrip(format, unit):
    freqs = np.linspace(1e9, 2e9, 5)
    s = np.random.rand(5, 2, 2) + 1j * np.random.rand(5, 2, 2)

    with tempfile.NamedTemporaryFile(suffix=".s2p", delete=False) as tmp:
        save_touchstone(tmp.name, freqs, s, format=format, unit=unit)
        f2, s2, meta, _ = load_touchstone(tmp.name, annotations=True)

    np.testing.assert_allclose(f2, freqs, rtol=1e-6)
    assert meta["format"] == format
    assert meta["unit"] == unit
    assert meta["n_ports"] == 2
    assert s2.shape == s.shape
