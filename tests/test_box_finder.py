# tests/test_box_finder.py
"""
Проверяем mwlab.opt.analysis.box_finder.BoxFinder

- Геометрический sanity-check: условная «зона годности» — |x| ≤ 0.1  и  |y| ≤ 0.1
- Алгоритм должен найти куб примерно этой ширины, покрывая ровно/почти ровно область PASS.
- Проверяются режимы mode="sym", mode="asym", 1D, fail, yield<1, разные параметры и ошибки.
"""

import pytest

from mwlab.opt.design.space import DesignSpace, ContinuousVar
from mwlab.opt.analysis.box_finder import BoxFinder
from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.objectives.specification import Specification

# ─────────────────────────── Stubs ────────────────────────────

class SurIdentity(BaseSurrogate):
    """Surrogate-заглушка: возвращает dict-точку как есть."""
    def predict(self, point, **kw):
        return point
    def batch_predict(self, points, **kw):
        return points

class SpecSquare(Specification):
    """Specification-заглушка: pass, если |x|,|y| ≤ 0.1."""
    def __init__(self):
        pass
    def is_ok(self, point):
        # поддержим как 1D, так и 2D точки
        if "y" in point:
            return abs(point["x"]) <= 0.1 and abs(point["y"]) <= 0.1
        return abs(point["x"]) <= 0.1

class SpecFail(Specification):
    """Всегда fail."""
    def __init__(self): pass
    def is_ok(self, point): return False

class SpecPartial(Specification):
    """Pass только если x>0, y>0, и внутри квадрата."""
    def __init__(self): pass
    def is_ok(self, point):
        return (point["x"] > 0 and point.get("y", 1) > 0 and
                abs(point["x"]) <= 0.1 and abs(point.get("y", 0)) <= 0.1)

def build_finder(mode, **kw):
    return BoxFinder(
        strategy="lhs_global",
        mode=mode,
        target_yield=kw.get("target_yield", 1.0),
        n_lhs=kw.get("n_lhs", 2048),
        rng=kw.get("rng", 123),
        zoom_ratio=kw.get("zoom_ratio", 1.2)
    )

# ─────────────────────────── Fixtures ────────────────────────────

@pytest.fixture(scope="module")
def space2d():
    return DesignSpace({"x": ContinuousVar(-1, 1), "y": ContinuousVar(-1, 1)})

@pytest.fixture(scope="module")
def space1d():
    return DesignSpace({"x": ContinuousVar(-1, 1)})

@pytest.fixture(scope="module")
def surrogate():
    return SurIdentity()

@pytest.fixture(scope="module")
def spec2d():
    return SpecSquare()

@pytest.fixture(scope="module")
def spec1d():
    # Тот же класс работает для 1D
    return SpecSquare()

# ─────────────────────────── Main Tests ────────────────────────────

def test_sym_box(space2d, surrogate, spec2d):
    finder = build_finder(mode="sym")
    cube = finder.find({"x": 0.0, "y": 0.0}, space2d, surrogate, spec2d, delta_init=0.2)
    dx_neg, dx_pos = cube["x"]
    dy_neg, dy_pos = cube["y"]
    assert pytest.approx(abs(dx_neg), rel=0.05) == 0.1
    assert pytest.approx(dx_pos,   rel=0.05) == 0.1
    assert pytest.approx(abs(dy_neg), rel=0.05) == 0.1
    assert pytest.approx(dy_pos,   rel=0.05) == 0.1

def test_asym_box(space2d, surrogate, spec2d):
    finder = build_finder(mode="asym")
    cube = finder.find({"x": 0.05, "y": -0.02}, space2d, surrogate, spec2d, delta_init=0.2)
    x_lo, x_hi = cube["x"]
    y_lo, y_hi = cube["y"]
    cx = 0.5 * (x_hi + x_lo)
    cy = 0.5 * (y_hi + y_lo)
    assert abs(cx) <= 0.1 and abs(cy) <= 0.1
    assert -0.1 <= cx + x_lo <= 0.1
    assert -0.1 <= cx + x_hi <= 0.1
    assert -0.1 <= cy + y_lo <= 0.1
    assert -0.1 <= cy + y_hi <= 0.1

def test_boxfinder_1d(space1d, surrogate, spec1d):
    finder = build_finder(mode="sym")
    cube = finder.find({"x": 0.0}, space1d, surrogate, spec1d, delta_init=0.5)
    x_lo, x_hi = cube["x"]
    assert pytest.approx(abs(x_lo), rel=0.05) == 0.1
    assert pytest.approx(x_hi, rel=0.05) == 0.1

def test_boxfinder_fail_everywhere(space2d):
    finder = build_finder(mode="sym")
    surrogate = SurIdentity()
    spec = SpecFail()
    with pytest.raises(RuntimeError):
        finder.find({"x": 0.0, "y": 0.0}, space2d, surrogate, spec, delta_init=0.2)

def test_boxfinder_yield_less_than_one(space2d):
    # PASS — только в правом верхнем квадранте, т.е. теоретический yield=0.25
    finder = build_finder(mode="sym", target_yield=0.2, n_lhs=1024)
    surrogate = SurIdentity()
    spec = SpecPartial()
    cube = finder.find({"x": 0.0, "y": 0.0}, space2d, surrogate, spec, delta_init=0.2)
    dx_neg, dx_pos = cube["x"]
    dy_neg, dy_pos = cube["y"]
    # Ожидаемый размер чуть больше 0.1 (но не более 0.2)
    assert dx_pos > 0
    assert dy_pos > 0
    assert dx_pos <= 0.2 and dy_pos <= 0.2

def test_boxfinder_delta_init_dict(space2d, surrogate, spec2d):
    finder = build_finder(mode="sym")
    # передаем разные delta_init для разных переменных
    cube = finder.find(
        {"x": 0.0, "y": 0.0},
        space2d, surrogate, spec2d,
        delta_init={"x": 0.15, "y": 0.05}
    )
    dx_neg, dx_pos = cube["x"]
    dy_neg, dy_pos = cube["y"]

    # x — всегда максимум, который можно (0.1)
    assert pytest.approx(abs(dx_neg), rel=0.10) == 0.1
    assert pytest.approx(dx_pos, rel=0.10) == 0.1

    # y: может быть либо 0.05 (если delta_init ограничивает), либо 0.1 (если зона pass шире)
    assert 0.049 <= abs(dy_neg) <= 0.101
    assert 0.049 <= dy_pos <= 0.101

def test_boxfinder_custom_zoom_and_nlhs(space2d, surrogate, spec2d):
    finder = build_finder(mode="sym", n_lhs=512, zoom_ratio=1.05)
    cube = finder.find({"x": 0.0, "y": 0.0}, space2d, surrogate, spec2d, delta_init=0.2)
    dx_neg, dx_pos = cube["x"]
    dy_neg, dy_pos = cube["y"]
    # Не должно быть сильно хуже по размеру бокса
    assert abs(dx_neg) > 0.07 and abs(dy_neg) > 0.07

def test_boxfinder_repr(space2d, surrogate, spec2d):
    finder = build_finder(mode="sym")
    r = repr(finder)
    assert "BoxFinder" in r and "sym" in r
