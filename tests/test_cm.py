# tests/test_cm.py
"""pytest-тесты для mwlab.filters.cm (CouplingMatrix и утилиты)
================================================================

Покрываем:
* parse_m_key / make_perm / MatrixLayout
* CouplingMatrix.__post_init__ (валидация размеров qu/phase)
* tensor_M() — корректное заполнение симметричной матрицы
* sparams() — интеграция с ядром (форма, dtype)
* to_matrix() / from_matrix() — SL/TAIL/CUSTOM и round-trip
* to_dict() / from_dict()
* (опционально) plot_matrix() — пропускается, если нет matplotlib

Запуск: pytest -q tests/test_cm.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mwlab.filters.topologies import Topology, get_topology
from mwlab.filters.cm import (
    parse_m_key,
    make_perm,
    MatrixLayout,
    CouplingMatrix,
)
from mwlab.filters.cm_core import DEFAULT_DEVICE


# ────────────────────────────────────────────────────────────────────────────
#                            parse_m_key
# ────────────────────────────────────────────────────────────────────────────

def test_parse_m_key_basic():
    assert parse_m_key("M1_4") == (1, 4)
    assert parse_m_key("M4_1") == (1, 4)  # порядок меняется на (low,high)
    assert parse_m_key("M3_3") == (3, 3)  # диагональ допустима

    with pytest.raises(ValueError):
        parse_m_key("X1_2")
    with pytest.raises(ValueError):
        parse_m_key("M1-")
    with pytest.raises(ValueError):
        parse_m_key("M0_2")  # индексы > 0


# ────────────────────────────────────────────────────────────────────────────
#                            make_perm / MatrixLayout
# ────────────────────────────────────────────────────────────────────────────

def test_make_perm_tail_and_sl():
    perm_tail = make_perm(order=3, ports=2, layout=MatrixLayout.TAIL)
    assert perm_tail == [0, 1, 2, 3, 4]

    perm_sl = make_perm(order=4, ports=2, layout=MatrixLayout.SL)
    # external: S, R1,R2,R3,R4, L   → indices [4,0,1,2,3,5]
    assert perm_sl == [4, 0, 1, 2, 3, 5]

    with pytest.raises(ValueError):
        make_perm(order=3, ports=3, layout=MatrixLayout.SL)


def test_make_perm_custom_ok_and_errors():
    perm = [3, 0, 1, 4, 2]
    assert make_perm(3, 2, MatrixLayout.CUSTOM, permutation=perm) == perm

    with pytest.raises(ValueError):
        make_perm(3, 2, MatrixLayout.CUSTOM, permutation=[0, 1])  # wrong len
    with pytest.raises(ValueError):
        make_perm(3, 2, MatrixLayout.CUSTOM, permutation=[0, 1, 1, 3, 4])  # repeats


# ────────────────────────────────────────────────────────────────────────────
#                           fixtures for CouplingMatrix
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def topo_folded():
    return get_topology("folded", order=4)


@pytest.fixture()
def cm_example(topo_folded):
    mvals = {
        "M1_2": 1.05, "M2_3": 0.90, "M3_4": 1.05, "M1_4": 0.25,
        "M1_5": 0.60, "M4_6": 0.80,
        # диагонали опциональны, добавим одну для теста
        "M2_2": 0.0,
    }
    return CouplingMatrix(topo_folded, mvals, qu=700.0)


# ────────────────────────────────────────────────────────────────────────────
#                              CouplingMatrix init
# ────────────────────────────────────────────────────────────────────────────

def test_cm_init_qu_and_phase_validation(topo_folded):
    mvals = {"M1_2": 1.0, "M1_5": 0.5, "M4_6": 0.5}

    # OK: qu scalar
    CouplingMatrix(topo_folded, mvals, qu=500.0)

    # OK: qu vector len=order
    CouplingMatrix(topo_folded, mvals, qu=[500.0]*topo_folded.order)

    # error: wrong qu length
    with pytest.raises(ValueError):
        CouplingMatrix(topo_folded, mvals, qu=[1,2,3])

    # phase_a wrong length
    with pytest.raises(ValueError):
        CouplingMatrix(topo_folded, mvals, phase_a=[0.1, 0.2, 0.3])


# ────────────────────────────────────────────────────────────────────────────
#                              tensor_M
# ────────────────────────────────────────────────────────────────────────────

def test_tensor_M_symmetry(cm_example):
    M = cm_example.tensor_M(device=DEFAULT_DEVICE)
    # Симметрия
    assert torch.allclose(M, M.transpose(-1, -2))
    # Размерность
    K = cm_example.topo.size
    assert M.shape == (K, K)
    # Значения нескольких элементов
    assert torch.isclose(M[0, 1], torch.tensor(1.05))
    assert torch.isclose(M[0, 4], torch.tensor(0.60))
    # диагональ по умолчанию = 0, кроме заданных явно
    assert torch.isclose(M[1, 1], torch.tensor(0.0))


# ────────────────────────────────────────────────────────────────────────────
#                              sparams
# ────────────────────────────────────────────────────────────────────────────

def test_cm_sparams_shape_dtype(cm_example):
    omega = torch.linspace(-3, 3, 301)
    S = cm_example.sparams(omega, device=DEFAULT_DEVICE)
    assert S.dtype == torch.complex64
    assert S.shape == (301, cm_example.topo.ports, cm_example.topo.ports)


# ────────────────────────────────────────────────────────────────────────────
#                         to_matrix / from_matrix
# ────────────────────────────────────────────────────────────────────────────
def _assert_mvals_close(m1, m2, *, atol=1e-8, rtol=1e-6, zero_tol=1e-12):
    """Сравнение словарей M_ij с допуском."""
    nz1 = {k: float(v) for k, v in m1.items() if abs(v) > zero_tol}
    nz2 = {k: float(v) for k, v in m2.items() if abs(v) > zero_tol}
    assert nz1.keys() == nz2.keys(), f"Keys differ: {nz1.keys() ^ nz2.keys()}"
    for k in nz1:
        assert nz1[k] == pytest.approx(nz2[k], rel=rtol, abs=atol), k

def test_to_matrix_and_back(cm_example):
    # SL
    M_sl = cm_example.to_matrix(layout=MatrixLayout.SL)
    assert isinstance(M_sl, (torch.Tensor, np.ndarray))
    assert M_sl.shape[0] == cm_example.topo.size

    cm_back = CouplingMatrix.from_matrix(M_sl, topo=cm_example.topo,
                                         layout=MatrixLayout.SL, qu=cm_example.qu)
    _assert_mvals_close(cm_back.mvals, cm_example.mvals)

    # CUSTOM
    perm = list(range(cm_example.topo.size))[::-1]
    M_custom = cm_example.to_matrix(layout=MatrixLayout.CUSTOM, permutation=perm)
    cm_back2 = CouplingMatrix.from_matrix(M_custom, topo=cm_example.topo,
                                          layout=MatrixLayout.CUSTOM, permutation=perm)
    _assert_mvals_close(cm_back2.mvals, cm_example.mvals)


# ────────────────────────────────────────────────────────────────────────────
#                        to_dict / from_dict
# ────────────────────────────────────────────────────────────────────────────

def test_to_from_dict_round(cm_example):
    blob = cm_example.to_dict()
    cm2 = CouplingMatrix.from_dict(None, blob)
    # Топология восстановится; проверим числа
    for k, v in cm_example.mvals.items():
        assert pytest.approx(v, rel=1e-6) == cm2.mvals[k]
    assert cm2.topo.order == cm_example.topo.order
    assert cm2.topo.ports == cm_example.topo.ports


# ────────────────────────────────────────────────────────────────────────────
#                           plot_matrix (optional)
# ────────────────────────────────────────────────────────────────────────────

def test_plot_matrix_optional(cm_example):
    mpl = pytest.importorskip("matplotlib")
    fig = cm_example.plot_matrix(layout=MatrixLayout.TAIL, annotate=False, log=False)
    assert fig is not None
    mpl.pyplot.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])  # для ручного запуска
