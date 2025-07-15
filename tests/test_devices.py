# tests/test_devices.py
"""
Юнит-тесты для *mwlab.filters.devices* (Device / Filter).

Фокус
-----
* корректность прямого **f → Ω** и обратного **Ω → f** (LP / HP / BP / BR);
* работоспособность ``sparams`` (NumPy-backend);
* round-trip через *TouchstoneData*;
* содержание словаря параметров устройства;
* отлов ошибок валидации конструкторов.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import skrf as rf

from mwlab.filters.cm import CouplingMatrix
from mwlab.filters.devices import Filter
from mwlab.filters.topologies import get_topology, TopologyError, Topology

# ───────────────────────────────────── fixtures ────────────────────────────
@pytest.fixture(scope="module")
def topo4() -> Topology:
    return get_topology("folded", order=4)


@pytest.fixture(scope="module")
def cm4(topo4):
    m_vals = {
        "M1_2": 1.05, "M2_3": 0.90, "M3_4": 1.05, "M1_4": 0.25,
        "M1_5": 0.60, "M4_6": 0.80,
    }
    return CouplingMatrix(topo4, M_vals=m_vals, Q=7000)


F_HZ = np.linspace(1.0e9, 3.0e9, 51)

# ───────────────────────── 1. фабрики Filter.* ────────────────────────────
@pytest.mark.parametrize(
    "factory, args, kwargs, spec",
    [
        (Filter.lp,  (2.5,),          dict(unit="GHz"), "cut"),
        (Filter.hp,  (2.2,),          dict(unit="GHz"), "cut"),
        (Filter.bp,  (2.1, 0.25),     dict(unit="GHz"), "bw"),
        (Filter.br,  (2.1, 0.25),     dict(unit="GHz"), "bw"),
        (Filter.bp_edges, (2.0, 2.2), dict(unit="GHz"), "edges"),
        (Filter.br_edges, (2.0, 2.2), dict(unit="GHz"), "edges"),
    ],
)
def test_factories(cm4, factory, args, kwargs, spec):
    flt: Filter = factory(cm4, *args, **kwargs)
    assert isinstance(flt, Filter)
    assert flt.ports == 2
    assert flt._spec == spec

# ─────────────────── 2. f ↔ Ω: прямое и обратное ──────────────────────────
@pytest.mark.parametrize(
    "flt_constructor, omega_formula",
    [
        # LP
        (
            lambda cm: Filter.lp(cm, 2.0, unit="GHz"),
            lambda f, f0: f / f0,
        ),
        # HP
        (
            lambda cm: Filter.hp(cm, 2.0, unit="GHz"),
            lambda f, f0: f0 / f,
        ),
        # BP
        (
            lambda cm: Filter.bp(cm, 2.0, 0.2, unit="GHz"),
            lambda f, fb: (fb[0] / fb[1]) * (f / fb[0] - fb[0] / f),
        ),
        # BR
        (
            lambda cm: Filter.br(cm, 2.0, 0.2, unit="GHz"),
            lambda f, fb: (fb[1] / fb[0]) / (f / fb[0] - fb[0] / f),
        ),
    ],
)
def test_forward_reverse_mapping(cm4, flt_constructor, omega_formula):
    filt: Filter = flt_constructor(cm4)
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = filt._omega(F_HZ)

    args = (filt.f0, filt.bw) if filt.kind in {"BP", "BR"} else (filt.f0,)
    with np.errstate(divide='ignore', invalid='ignore'):
        ref = omega_formula(F_HZ, args)

    # сравниваем только конечные значения (без inf / nan)
    finite = np.isfinite(ref) & np.isfinite(omega)
    assert np.allclose(omega[finite], ref[finite], rtol=1e-12, atol=1e-12)

    # обратное Ω → f
    f_back = filt.freq_grid(omega, unit="GHz", as_rf=False)
    assert np.allclose(f_back * 1e9, F_HZ, atol=1e-3)

# ────────────────────────── 3. sparams smoke ───────────────────────────────
def test_sparams_shape_dtype(cm4):
    filt = Filter.bp(cm4, 2.0, 0.25, unit="GHz")
    S = filt.sparams(F_HZ, backend="numpy")
    assert S.shape == (F_HZ.size, 2, 2)
    assert S.dtype == np.complex64

# ──────────────────── 4. Touchstone round-trip ─────────────────────────────
@pytest.mark.parametrize(
    "factory",
    [
        lambda cm: Filter.lp(cm, 1.8, unit="GHz"),
        lambda cm: Filter.bp(cm, 2.0, 0.25, unit="GHz"),
        lambda cm: Filter.br(cm, 2.0, 0.25, unit="GHz"),
    ],
)
def test_touchstone_roundtrip(cm4, factory):
    filt = factory(cm4)
    ts = filt.to_touchstone(F_HZ)
    restored = Filter.from_touchstone(ts)

    assert restored.kind == filt.kind
    assert math.isclose(restored.f0, filt.f0, rel_tol=1e-12)
    assert restored._spec == filt._spec

    np.testing.assert_allclose(
        restored.cm.sparams(np.linspace(-1, 1, 11)),
        filt.cm.sparams(np.linspace(-1, 1, 11)),
    )

# ───────────────────── 5. ошибки валидации ────────────────────────────────
def test_invalid_kind(cm4):
    with pytest.raises(ValueError):
        Filter(cm4, kind="XYZ", f0=1.0)

def test_wrong_ports_in_topology():
    with pytest.raises(TopologyError):
        get_topology("folded", order=4, ports=3)

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(f0=2.0, bw=0.1, fbw=0.05),
        dict(bw=0.1),
    ],
)
def test_bad_combination(cm4, kwargs):
    with pytest.raises(ValueError):
        Filter(cm4, kind="BP", **kwargs)

# ──────────────── 6. freq_grid с rf.Frequency ─────────────────────────────
def test_freq_grid_as_rf(cm4):
    filt = Filter.hp(cm4, 2.0, unit="GHz")
    om = np.linspace(0.5, 2.0, 101)
    freq_obj = filt.freq_grid(om, unit="MHz", as_rf=True)
    assert isinstance(freq_obj, rf.Frequency)

    sel = np.sort(freq_obj.f[::25])  # f  ↑
    omega_ref = om[::25][::-1] if filt.kind == "HP" else om[::25]
    assert np.allclose(filt._omega(sel), omega_ref, rtol=1e-12, atol=1e-12)

# ────────────────── 7. проверка _device_params ─────────────────────────────
def test_device_params_content(cm4):
    filt = Filter.br_edges(cm4, 1.9, 2.1, unit="GHz")
    params = filt._device_params()

    bw_keys = {"bw", "fbw", "f_lower", "f_upper"}
    assert sum(k in params for k in bw_keys) == 2  # f_center + один из пары

    ts = filt.to_touchstone(F_HZ)
    assert ts.params["device"] == "Filter"
    assert "kind" in ts.params


