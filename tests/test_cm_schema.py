# tests/test_cm_schema.py
import numpy as np
import torch
import pytest

from mwlab.filters.topologies import get_topology
from mwlab.filters.cm_schema import ParamSchema
from mwlab.filters.cm_core import build_M

# ------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def topo():
    # folded: order=4, ports=2  (diag + 5 links)
    return get_topology("folded", order=4)

@pytest.fixture(scope="module")
def schema_full(topo):
    # include все блоки
    return ParamSchema.from_topology(topo,
                                     include_qu="vec",
                                     include_phase=("a", "b"))

# ------------------------------------------------------- tests

def test_key_order(topo):
    schema = ParamSchema.from_topology(topo, include_qu="none", include_phase=())
    # diag → links(sorted)
    diag_expected = [f"M{i}_{i}" for i in range(1, topo.order + 1)]
    off_expected  = [f"M{i}_{j}" for (i, j) in sorted(topo.links)]
    assert schema.keys[: topo.order] == tuple(diag_expected)
    assert schema.keys[topo.order : schema.L_M] == tuple(off_expected)

def test_pack_strict_and_default(schema_full):
    # заполним все M = 0.5, qu = 700, phase_a/b = 0
    params = {k: 0.5 for k in schema_full.keys if k.startswith("M")}
    params.update({f"qu_{i}": 700 for i in range(1, schema_full.topo.order + 1)})
    params.update({f"phase_a{i}": 0.1 for i in range(1, schema_full.topo.ports + 1)})
    params.update({f"phase_b{i}": 0.0 for i in range(1, schema_full.topo.ports + 1)})

    vec = schema_full.pack(params, strict=True)
    assert vec.shape == (schema_full.size,)

    # unpack → pack round‑trip
    unpacked = schema_full.unpack(vec)
    vec2 = schema_full.pack(unpacked, strict=True)
    assert torch.allclose(vec, vec2)

    # strict mode: пропуск ключа вызывает ошибку
    bad = dict(params)
    bad.pop("M1_2")
    with pytest.raises(KeyError):
        schema_full.pack(bad, strict=True)

def test_pack_scalar_phase(schema_full):
    # phase_a / phase_b заданы скалярами — devono развернуться
    params = {"phase_a": 0.2, "phase_b": -0.1}
    vec = schema_full.pack(params, strict=False)
    unpacked = schema_full.unpack(vec)
    # все phase_a* должны равняться 0.2
    phase_a_keys = [k for k in schema_full.keys if k.startswith("phase_a")]
    assert all(abs(unpacked[k] - 0.2) < 1e-6 for k in phase_a_keys)

def test_assemble_correctness(schema_full):
    rng = torch.Generator().manual_seed(0)
    # случайный вектор
    vec = torch.randn(schema_full.size, generator=rng)
    M_real, qu, phase_a, phase_b = schema_full.assemble(vec)

    # --- размеры ---
    K = schema_full.topo.size
    assert M_real.shape == (K, K)
    assert qu.shape == (schema_full.topo.order,)
    assert phase_a.shape == (schema_full.topo.ports,)
    assert phase_b.shape == (schema_full.topo.ports,)

    # --- симметрия M ---
    assert torch.allclose(M_real, M_real.t(), atol=1e-7)

    # --- проверка build_M: reconstruct from packed values ---
    rows = torch.tensor(schema_full.m_rows)
    cols = torch.tensor(schema_full.m_cols)
    vals = vec[: schema_full.L_M]
    M_ref = build_M(rows, cols, vals, K)
    assert torch.allclose(M_real, M_ref)

def test_masks_lengths(schema_full):
    masks = schema_full.masks()

    # helper: сколько элементов соответствует slice‑ку
    def sl_len(sl):
        return sl.stop - sl.start

    # блок M
    assert masks["M"].sum() == sl_len(schema_full.slices["M"])

    # блок qu
    if schema_full.include_qu != "none":
        assert masks["qu"].sum() == sl_len(schema_full.slices["qu"])

    # блоки фаз
    if "a" in schema_full.include_phase:
        assert masks["phase_a"].sum() == sl_len(schema_full.slices["phase_a"])
    if "b" in schema_full.include_phase:
        assert masks["phase_b"].sum() == sl_len(schema_full.slices["phase_b"])


