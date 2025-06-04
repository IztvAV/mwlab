# tests/test_sensitivity.py
from __future__ import annotations

import importlib.util
import numpy as np
import pytest

from mwlab.opt.design.space import DesignSpace, ContinuousVar
from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.sensitivity import SensitivityAnalyzer


# ────────────────────────────────────────────────────────────────────────────
#  1. Dummy-суррогат: y = 1, если x1+x2 > 0, иначе 0
# ────────────────────────────────────────────────────────────────────────────
class DummySurrogate(BaseSurrogate):
    def predict(self, x, *, return_std: bool = False):
        y = 1.0 if (x["x1"] + x["x2"]) > 0.0 else 0.0
        return y

    def batch_predict(self, xs, *, return_std: bool = False):
        return [self.predict(x) for x in xs]


# ────────────────────────────────────────────────────────────────────────────
#  2. Dummy-Specification: ok, если y > 0.5
# ────────────────────────────────────────────────────────────────────────────
class DummySpec:
    def is_ok(self, y):
        return float(y) > 0.5


# ────────────────────────────────────────────────────────────────────────────
#  3. Общие фикстуры
# ────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def design_space():
    return DesignSpace({
        "x1": ContinuousVar(-1.0, 1.0),
        "x2": ContinuousVar(-1.0, 1.0),
    })


@pytest.fixture(scope="module")
def analyzer(design_space):
    sur = DummySurrogate()
    spec = DummySpec()
    return SensitivityAnalyzer(
        surrogate=sur,
        design_space=design_space,
        specification=spec,
    )


# ────────────────────────────────────────────────────────────────────────────
#  4. Morris
# ────────────────────────────────────────────────────────────────────────────
salib_missing = importlib.util.find_spec("SALib") is None


@pytest.mark.skipif(salib_missing, reason="SALib не установлен — пропускаем Morris/Sobol")
def test_morris(analyzer):
    df = analyzer.morris(N=4, plot=False)
    # Проверяем форму вывода
    assert list(df.columns) == ["mu_star", "sigma"]
    assert set(df.index) == {"x1", "x2"}
    # mu_star должны быть неотрицательны
    assert (df["mu_star"] >= 0).all()


# ────────────────────────────────────────────────────────────────────────────
#  5. Sobol
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(salib_missing, reason="SALib не установлен — пропускаем Morris/Sobol")
def test_sobol(analyzer):
    df = analyzer.sobol(n_base=64, plot=False)
    # Колонки
    assert {"S1", "ST"}.issubset(df.columns)
    # Индексы
    assert set(df.index) == {"x1", "x2"}
    # Индексы лежат в [0,1] (с учётом небольшой численной погрешности)
    assert (df["S1"].between(-0.05, 1.05)).all()
    assert (df["ST"].between(-0.05, 1.05)).all()


# ────────────────────────────────────────────────────────────────────────────
#  6. Active Subspace
# ────────────────────────────────────────────────────────────────────────────
def test_active_subspace(analyzer):
    lam, W = analyzer.active_subspace(k=2, n_samples=200, plot=False)
    # Проверяем размерности
    assert lam.shape == (2,)
    assert W.shape == (2, 2)
    # Собственные значения неотрицательны
    assert (lam >= -1e-12).all()
