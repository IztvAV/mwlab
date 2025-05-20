# test_doe.py
import pytest
import numpy as np

from mwlab.opt.design.space import (
    DesignSpace, ContinuousVar, IntegerVar, OrdinalVar, CategoricalVar
)
from mwlab.opt.design.samplers import (
    get_sampler, SobolSampler, HaltonSampler, LHSampler, LHSMaximinSampler, NormalSampler, FactorialFullSampler
)

# ----------------------------------------------------------------------------
# 1. Тесты пространства переменных
# ----------------------------------------------------------------------------

def test_space_basic_continuous():
    var = ContinuousVar(lower=0.0, upper=10.0)
    assert var.bounds() == (0.0, 10.0)
    assert var.to_unit(0.5) == 5.0
    assert var.from_unit(7.5) == 0.75
    js = var.to_json()
    assert js["lower"] == 0.0
    assert ContinuousVar.from_json(js).upper == 10.0

def test_space_integer_var():
    var = IntegerVar(lower=2, upper=8, step=2)
    vals = [var.to_unit(z) for z in np.linspace(0, 1, 5)]
    assert set(vals) <= {2, 4, 6, 8}
    assert var.from_unit(6) == (6-2)/(8-2)
    js = var.to_json()
    assert js["type"] == "integer"
    assert IntegerVar.from_json(js).lower == 2

def test_space_ordinal_categorical():
    ordv = OrdinalVar(levels=["low", "mid", "high"])
    catv = CategoricalVar(levels=["Cu", "Ag", "Au"])
    assert ordv.to_unit(0.0) == "low"
    assert ordv.to_unit(1.0) == "high"
    assert catv.from_unit("Ag") == 1 / 2
    js1 = ordv.to_json()
    js2 = catv.to_json()
    assert OrdinalVar.from_json(js1).levels == ["low", "mid", "high"]
    assert CategoricalVar.from_json(js2).levels == ["Cu", "Ag", "Au"]

def test_design_space_normalize_denormalize():
    space = DesignSpace({
        "x": ContinuousVar(0, 10),
        "n": IntegerVar(1, 5),
        "m": CategoricalVar(["a", "b", "c"]),
    })
    point = {"x": 5.0, "n": 3, "m": "b"}
    z = space.normalize(point)
    point2 = space.denormalize(z)
    assert point2["m"] == "b"
    assert abs(point2["x"] - 5.0) < 1e-8
    assert point2["n"] == 3

def test_design_space_from_center_delta():
    centers = {"x": 1.0, "y": -2.0}
    space = DesignSpace.from_center_delta(centers, delta=0.5)
    assert isinstance(space["x"], ContinuousVar)
    assert abs(space["x"].lower - 0.5) < 1e-10
    assert abs(space["y"].upper + 1.5) < 1e-10

def test_design_space_constraints():
    space = DesignSpace({"x": ContinuousVar(0, 1), "y": ContinuousVar(0, 1)})
    space.add_constraint(lambda p: p["x"] + p["y"] <= 1)
    pts = space.sample(8, sampler="sobol")
    assert all(p["x"] + p["y"] <= 1 for p in pts)

def test_design_space_serialization(tmp_path):
    space = DesignSpace({
        "a": ContinuousVar(0, 1, unit="m"),
        "b": IntegerVar(1, 3)
    })
    yaml = space.to_yaml()
    file = tmp_path / "space.yaml"
    file.write_text(yaml)
    loaded = DesignSpace.from_file(file)
    assert loaded["a"].unit == "m"
    assert loaded["b"].lower == 1

# ----------------------------------------------------------------------------
# 2. Тесты сэмплеров
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", [
    "sobol", "halton", "lhs", "lhs_maximin", "normal", "factorial_full"
])
def test_samplers_integration(sampler):
    space = DesignSpace({
        "w": ContinuousVar(-1, 1),
        "gap": ContinuousVar(-1, 1)
    })
    smp = get_sampler(sampler, rng=2024)
    pts = smp.sample(space, 8)
    assert isinstance(pts, list)
    assert all(isinstance(p, dict) and len(p) == 2 for p in pts)

def test_sampler_incremental_state():
    space = DesignSpace({"x": ContinuousVar(0, 1)})
    sampler = SobolSampler(rng=0)
    pts1 = sampler.sample(space, 2)
    state = sampler.state_dict()
    sampler2 = SobolSampler(rng=0)
    sampler2.load_state_dict(state)
    pts2 = sampler2.sample(space, 2)
    # должен продолжить ту же Sobol последовательность
    all_pts = pts1 + pts2
    assert len({tuple(p.values()) for p in all_pts}) == 4

def test_sampler_registry_aliases():
    smp = get_sampler("sobol")
    assert isinstance(smp, SobolSampler)
    smp2 = get_sampler("lhs")
    assert isinstance(smp2, LHSampler)
    smp3 = get_sampler("normal")
    assert isinstance(smp3, NormalSampler)

# ----------------------------------------------------------------------------
# 3. Misc
# ----------------------------------------------------------------------------

def test_categorical_sampling():
    space = DesignSpace({
        "c": CategoricalVar(levels=["aa", "bb", "cc"])
    })
    sampler = get_sampler("sobol")
    pts = sampler.sample(space, 8)
    assert all(p["c"] in ["aa", "bb", "cc"] for p in pts)

def test_sampling_with_integer_vars():
    space = DesignSpace({
        "x": IntegerVar(1, 10, step=2)
    })
    sampler = get_sampler("lhs")
    pts = sampler.sample(space, 7)
    for p in pts:
        x = p["x"]
        assert isinstance(x, int)
        assert 1 <= x <= 10
