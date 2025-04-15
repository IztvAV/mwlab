
import numpy as np
import tempfile
from mwlab.touchstone import TouchstoneFile

def test_load_save_roundtrip():
    freqs = np.linspace(1e9, 3e9, 5)
    s = np.random.rand(5, 2, 2) + 1j * np.random.rand(5, 2, 2)
    ts = TouchstoneFile.from_data(freqs, s, format='RI', unit='HZ')

    with tempfile.NamedTemporaryFile(suffix=".s2p", delete=False) as tmp:
        path = tmp.name

    ts.save(path, format='MA', unit='MHZ')
    ts_loaded = TouchstoneFile.load(path)

    assert ts_loaded.n_ports == 2
    assert ts_loaded.n_freqs == 5
    np.testing.assert_allclose(ts_loaded.freq, freqs, rtol=1e-6)

def test_to_format_conversion():
    freqs = np.array([1e9, 2e9])
    s = np.array([
        [[0.5 + 0.1j, 0.1 - 0.05j],
         [0.1 + 0.05j, 0.3 - 0.1j]],
        [[0.4 + 0.2j, 0.2 + 0.1j],
         [0.2 - 0.1j, 0.25 + 0.15j]]
    ])
    ts = TouchstoneFile.from_data(freqs, s)
    ts_ma = ts.to_format('MA')
    ts_db = ts.to_format('DB')

    assert ts_ma.format == 'MA'
    assert ts_db.format == 'DB'
    assert ts.format == 'RI'

def test_from_data_and_plot(monkeypatch):
    freqs = np.linspace(1e9, 5e9, 100)
    s = np.zeros((100, 2, 2), dtype=complex)
    s[:, 0, 0] = 0.5 * np.exp(1j * 2 * np.pi * freqs * 1e-9)
    s[:, 1, 1] = 0.3 + 0.2j

    ts = TouchstoneFile.from_data(freqs, s)

    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    ts.plot_s_db("S11")
    ts.plot_s_db("S22")

def test_invalid_element_plot_raises():
    freqs = np.array([1e9])
    s = np.array([[[0.5 + 0.1j]]])
    ts = TouchstoneFile.from_data(freqs, s)

    try:
        ts.plot_s_db("S21")  # Индексы вне диапазона
    except IndexError:
        pass
    else:
        assert False, "Expected IndexError"
