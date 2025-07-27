# tests/test_devices.py
"""
Юнит-тесты для *mwlab.filters.devices* (Device / Filter).

Проверяем:
* корректность прямого **f → Ω** и обратного **Ω → f** (LP / HP / BP / BR);
* работоспособность ``sparams`` (torch-backend);
* round-trip через TouchstoneData;
* содержание словаря параметров устройства;
* отлов ошибок валидации конструкторов.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import skrf as rf
import torch

from mwlab.filters.cm import CouplingMatrix
from mwlab.filters.devices import Filter
from mwlab.filters.topologies import get_topology, TopologyError, Topology

DT_C = torch.complex64  # ожидаемый комплексный dtype

# ─────────────────────────────────── fixtures ────────────────────────────
@pytest.fixture(scope="module")
def topo4() -> Topology:
    return get_topology("folded", order=4)


@pytest.fixture(scope="module")
def cm4(topo4):
    m_vals = {
        "M1_2": 1.05, "M2_3": 0.90, "M3_4": 1.05, "M1_4": 0.25,
        "M1_5": 0.60, "M4_6": 0.80,
    }
    # нормированная добротность (qu), а не Q!
    return CouplingMatrix(topo4, m_vals, qu=7000.0)


F_HZ = np.linspace(1.0e9, 3.0e9, 51)


# ───────────────────── 1. фабрики Filter.* ───────────────────────────────
@pytest.mark.parametrize(
    "factory, args, kwargs, spec",
    [
        (Filter.lp,        (2.5,),          dict(unit="GHz"), "cut"),
        (Filter.hp,        (2.2,),          dict(unit="GHz"), "cut"),
        (Filter.bp,        (2.1, 0.25),     dict(unit="GHz"), "bw"),
        (Filter.br,        (2.1, 0.25),     dict(unit="GHz"), "bw"),
        (Filter.bp_edges,  (2.0, 2.2),      dict(unit="GHz"), "edges"),
        (Filter.br_edges,  (2.0, 2.2),      dict(unit="GHz"), "edges"),
    ],
)
def test_factories(cm4, factory, args, kwargs, spec):
    flt: Filter = factory(cm4, *args, **kwargs)
    assert isinstance(flt, Filter)
    assert flt.ports == 2
    assert flt._spec == spec


# ─────────────── 2. f ↔ Ω: прямое и обратное -----------------------------
@pytest.mark.parametrize(
    "flt_constructor",
    [
        lambda cm: Filter.lp(cm, 2.0, unit="GHz"),
        lambda cm: Filter.hp(cm, 2.0, unit="GHz"),
        lambda cm: Filter.bp(cm, 2.0, 0.2, unit="GHz"),
        lambda cm: Filter.br(cm, 2.0, 0.2, unit="GHz"),
    ],
)
def test_forward_reverse_mapping(cm4, flt_constructor):
    filt: Filter = flt_constructor(cm4)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = filt._omega(F_HZ)

    # референс
    if filt.kind == "LP":
        ref = F_HZ / filt.f_edges[0]
    elif filt.kind == "HP":
        ref = filt.f_edges[1] / F_HZ
    elif filt.kind == "BP":
        ratio = F_HZ / filt.f0 - filt.f0 / F_HZ
        ref = (filt.f0 / filt.bw) * ratio
    else:  # BR
        ratio = F_HZ / filt.f0 - filt.f0 / F_HZ
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = (filt.bw / filt.f0) / ratio

    finite = np.isfinite(ref) & np.isfinite(omega)
    assert np.allclose(omega[finite], ref[finite], rtol=1e-12, atol=1e-12)

    # обратное Ω → f
    f_back = filt.freq_grid(omega, unit="GHz", as_rf=False)
    assert np.allclose(f_back * 1e9, F_HZ, atol=1e-3)


# ───────────────── 3. sparams smoke ▸ Torch ------------------------------
def test_sparams_shape_dtype(cm4):
    filt = Filter.bp(cm4, 2.0, 0.25, unit="GHz")
    S = filt.sparams(F_HZ)
    assert S.shape == (F_HZ.size, 2, 2)
    assert S.dtype == DT_C


# ─────────────── 4. Touchstone round‑trip -------------------------------
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

    omega_test = np.linspace(-1, 1, 11)
    S1 = restored.cm.sparams(omega_test).detach().cpu().numpy()
    S2 = filt.cm.sparams(omega_test).detach().cpu().numpy()
    np.testing.assert_allclose(S1, S2, rtol=1e-6, atol=1e-8)


# ───────────── 5. ошибки валидации ---------------------------------------
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


# ──────── 6. freq_grid с rf.Frequency ------------------------------------
def test_freq_grid_as_rf(cm4):
    filt = Filter.hp(cm4, 2.0, unit="GHz")
    om = np.linspace(0.5, 2.0, 101)
    freq_obj = filt.freq_grid(om, unit="MHz", as_rf=True)
    assert isinstance(freq_obj, rf.Frequency)

    sel = np.sort(freq_obj.f[::25])
    omega_ref = om[::25][::-1] if filt.kind == "HP" else om[::25]
    assert np.allclose(filt._omega(sel), omega_ref, rtol=1e-12, atol=1e-12)


# ────────────── 7. _device_params content --------------------------------
def test_device_params_content(cm4):
    filt = Filter.br_edges(cm4, 1.9, 2.1, unit="GHz")
    params = filt._device_params()

    # для BR/BP всегда есть f_center и один из: bw/fbw или f_lower/f_upper
    bw_keys = {"bw", "fbw", "f_lower", "f_upper"}
    assert sum(k in params for k in bw_keys) == 2  # f_center + один набор

    ts = filt.to_touchstone(F_HZ)
    assert ts.params["device"] == "Filter"
    assert "kind" in ts.params


# ───────────── 8. LP/HP — ввод через f_edges -----------------------------
def test_lp_hp_accept_edges(cm4):
    lp = Filter(cm4, kind="LP", f_edges=(2.3e9, None))
    assert math.isclose(lp.f_edges[0], 2.3e9)

    hp = Filter(cm4, kind="HP", f_edges=(None, 2.3e9))
    assert math.isclose(hp.f_edges[1], 2.3e9)


# ───────────── 9. неизвестная единица частоты ----------------------------
def test_unknown_unit_message(cm4):
    with pytest.raises(ValueError) as exc:
        Filter.lp(cm4, 2.3, unit="FooBar")
    msg = str(exc.value).lower()
    assert "foobar" in msg and ("ghz" in msg or "hz" in msg)


# ───────────── 10. BP/BR — двусторонняя ветвь ----------------------------
@pytest.mark.parametrize(
    "factory",
    [
        lambda cm: Filter.bp(cm, 2.0, 0.2, unit="GHz"),
        lambda cm: Filter.br(cm, 2.0, 0.2, unit="GHz"),
    ],
)
def test_two_sided_sign_encoding(cm4, factory):
    filt = factory(cm4)
    om = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    f = filt.freq_grid(om, unit="Hz", as_rf=False)

    assert f.size == om.size
    assert math.isclose(f[om == 0][0], filt.f0, rel_tol=1e-12)

    pos, neg = om > 0, om < 0
    assert np.allclose(f[pos] * f[neg][::-1], filt.f0 ** 2, rtol=1e-12, atol=1e-12)


# ───────────── 11. set_Q / Q‑property ------------------------------------
def test_set_Q_and_property(cm4):
    filt = Filter.bp(cm4, 2.0, 0.2, unit="GHz")
    filt.set_Q(70_000)                # FBW = 0.1 → qu = 7 000
    assert np.allclose(filt.Q, 70_000)
    assert np.allclose(filt.cm.qu, 7_000)
