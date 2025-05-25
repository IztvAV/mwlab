# tests/test_box_finder.py
"""
Юнит-тесты для mwlab.opt.analysis.box_finder.BoxFinder
=====================================================

•   Геометрический sanity-check: PASS-область — квадрат |x|≤0.1, |y|≤0.1.
•   Проверяются обе стратегии (`hypervolume_tol`, `axis_tol`)
    и оба режима симметрии (`sym`, `asym`).
•   Отдельно тестируются ситуации «полный FAIL», `yield < α`,
    а также валидация некорректных аргументов.
"""

from __future__ import annotations

import pytest

from mwlab.opt.design.space import ContinuousVar, DesignSpace
from mwlab.opt.analysis.box_finder import BoxFinder
from mwlab.opt.surrogates import BaseSurrogate
from mwlab.opt.objectives.specification import Specification

# ────────────────────────── заглушки ────────────────────────────
class SurIdentity(BaseSurrogate):
    """Surrogate-заглушка: возвращает точку как есть (dict)."""
    def predict(self, point, **kw):         # pragma: no cover
        return point
    def batch_predict(self, points, **kw):
        return points

class SpecSquare(Specification):
    """PASS, если каждая координата ≤0.1 по модулю."""
    def __init__(self): pass
    def is_ok(self, point):
        if "y" in point:
            return abs(point["x"]) <= 0.1 and abs(point["y"]) <= 0.1
        return abs(point["x"]) <= 0.1

class SpecFail(Specification):
    """Всегда FAIL."""
    def __init__(self): pass
    def is_ok(self, point): return False

class SpecPartial(Specification):
    """PASS только в правом-верхнем квадранте (yield≈0.25)."""
    def __init__(self): pass
    def is_ok(self, p):
        return (
            p["x"] > 0 and p.get("y", 1) > 0 and
            abs(p["x"]) <= 0.1 and abs(p.get("y", 0)) <= 0.1
        )

# ────────────────────────── helpers ─────────────────────────────
def make_space_2d():
    return DesignSpace({"x": ContinuousVar(-1, 1), "y": ContinuousVar(-1, 1)})

def make_space_1d():
    return DesignSpace({"x": ContinuousVar(-1, 1)})

def build_finder(
    *,
    strategy="hypervolume_tol",
    mode="sym",
    target_yield=0.99,
    n_lhs=2048,
    n_max=8192,
):
    return BoxFinder(
        strategy=strategy,
        mode=mode,
        target_yield=target_yield,
        n_lhs=n_lhs,
        n_max=n_max,
        conf_level=0.80,     # более «мягкий» CI — ускоряет тест
        rng=123,
        zoom_ratio=1.1,
    )

# ────────────────────────── tests ───────────────────────────────
@pytest.mark.parametrize("strategy", ["hypervolume_tol", "axis_tol"])
def test_sym_box(strategy):
    space = make_space_2d()
    finder = build_finder(strategy=strategy, mode="sym")
    cube = finder.find({"x": 0.0, "y": 0.0},
                       space,
                       SurIdentity(),
                       SpecSquare(),
                       delta_init=0.25)
    dx_neg, dx_pos = cube["x"]
    dy_neg, dy_pos = cube["y"]
    tol = 0.05 if strategy == "hypervolume_tol" else 0.12
    assert pytest.approx(abs(dx_neg), rel=tol) == 0.1
    assert pytest.approx(dx_pos,      rel=tol) == 0.1
    assert pytest.approx(abs(dy_neg), rel=tol) == 0.1
    assert pytest.approx(dy_pos,      rel=tol) == 0.1

@pytest.mark.parametrize("strategy", ["hypervolume_tol", "axis_tol"])
def test_asym_box(strategy):
    space = make_space_2d()
    center = {"x": 0.05, "y": -0.02}
    finder = build_finder(strategy=strategy, mode="asym")
    cube = finder.find(center, space, SurIdentity(), SpecSquare(), delta_init=0.25)
    # проверяем, что границы не выходят за пределы PASS-квадрата
    assert center["x"] + cube["x"][0] >= -0.1
    assert center["x"] + cube["x"][1] <=  0.1
    assert center["y"] + cube["y"][0] >= -0.1
    assert center["y"] + cube["y"][1] <=  0.1

def test_1d_axis_sym():
    space = make_space_1d()
    finder = build_finder(strategy="axis_tol", mode="sym")
    cube = finder.find({"x": 0.0}, space, SurIdentity(), SpecSquare(), delta_init=0.3)
    x_lo, x_hi = cube["x"]
    tol = 0.12
    assert pytest.approx(abs(x_lo), rel=tol) == 0.1
    assert pytest.approx(x_hi,      rel=tol) == 0.1

def test_fail_everywhere():
    space = make_space_2d()
    finder = build_finder(strategy="hypervolume_tol", mode="sym", n_lhs=256)
    with pytest.raises(RuntimeError):
        finder.find({"x": 0.0, "y": 0.0},
                    space, SurIdentity(), SpecFail(), delta_init=0.2)

def test_yield_less_than_one():
    """Проверяем, что при α<1 алгоритм выдаёт бокс >0.1 по «плюсовым» граням."""
    space = make_space_2d()
    finder = build_finder(strategy="axis_tol", mode="sym",
                          target_yield=0.2, n_lhs=256)
    cube = finder.find({"x": 0.0, "y": 0.0},
                       space, SurIdentity(), SpecPartial(), delta_init=0.2)
    # δ⁺ должно быть положительным и ≤0.2
    assert 0 < cube["x"][1] <= 0.2
    assert 0 < cube["y"][1] <= 0.2

def test_invalid_zoom_ratio():
    with pytest.raises(ValueError):
        BoxFinder(zoom_ratio=1.0)   # zoom должен быть >1

def test_repr_contains_mode():
    r = repr(build_finder(mode="asym"))
    assert "mode=asym" in r and "strategy=" in r


