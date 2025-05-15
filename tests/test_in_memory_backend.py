# tests\test_in_memory_backend.py
import numpy as np
import pytest
import skrf as rf
from mwlab.io.touchstone import TouchstoneData
from mwlab.io.backends.in_memory import RAMBackend, SyntheticBackend

def create_dummy_ts(f_start=1e9, npoints=5):
    freq = rf.Frequency(f_start/1e9, (f_start + 1e9)/1e9, npoints, unit='GHz')
    s = np.zeros((npoints, 2, 2), dtype=complex)
    net = rf.Network(frequency=freq, s=s)
    return TouchstoneData(net, {"example_param": 42})

def test_ram_backend_append_and_read():
    backend = RAMBackend()
    ts = create_dummy_ts()
    backend.append(ts)
    assert len(backend) == 1
    out = backend.read(0)
    np.testing.assert_allclose(out.network.s, ts.network.s)
    assert out.params["example_param"] == 42

def test_ram_backend_pickle(tmp_path):
    ts_list = [create_dummy_ts(f_start=1e9 + i * 1e6) for i in range(3)]
    backend = RAMBackend(ts_list)
    pkl_path = tmp_path / "data.pkl"
    backend.dump_pickle(pkl_path)
    loaded = RAMBackend.load_pickle(pkl_path)
    assert len(loaded) == 3
    np.testing.assert_allclose(loaded.read(1).network.f, ts_list[1].network.f)

def test_ram_backend_read_negative_index():
    backend = RAMBackend([create_dummy_ts(), create_dummy_ts()])
    last = backend.read(-1)
    assert isinstance(last, TouchstoneData)

def test_synthetic_backend_eager():
    backend = SyntheticBackend(
        length=3,
        factory=create_dummy_ts,
        cache=True
    )
    assert len(backend) == 3
    ts = backend.read(2)
    assert isinstance(ts, TouchstoneData)

def test_synthetic_backend_lru():
    backend = SyntheticBackend(
        length=100,
        factory=create_dummy_ts,
        cache=2
    )
    ts0 = backend.read(0)
    ts1 = backend.read(1)
    ts0_again = backend.read(0)
    assert isinstance(ts0_again, TouchstoneData)

def test_synthetic_backend_append_disallowed_when_no_cache():
    backend = SyntheticBackend(
        length=1,
        factory=create_dummy_ts,
        cache=False
    )
    with pytest.raises(OSError):
        backend.append(create_dummy_ts())

def test_synthetic_backend_pickle_not_supported():
    with pytest.raises(IOError):
        SyntheticBackend.load_pickle("some.pkl")

def test_synthetic_backend_read_negative_index():
    backend = SyntheticBackend(
        length=3,
        factory=create_dummy_ts,
        cache=True
    )
    ts = backend.read(-1)
    assert isinstance(ts, TouchstoneData)
