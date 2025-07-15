# tests/test_cm.py
"""
Юнит-тесты для mwlab.filters.cm
-------------------------------

Запуск:
    pytest -q tests/test_cm.py
"""
from __future__ import annotations

import numpy as np
import pytest
import tempfile
from pathlib import Path
from typing import Sequence

from mwlab.filters.topologies import get_topology
from mwlab.filters.cm import (
    _build_M_complex,
    _phase_vec,
    _phase_diag,
    CouplingMatrix,
MatrixLayout,
    _get_cached_mats,
)
from mwlab.filters import io as cmio  # write_matrix / read_matrix
# ———————————————————————————————————————————————————————————————
#                              Fixtures
# ———————————————————————————————————————————————————————————————

torch_unavailable = False
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_unavailable = True

mpl_unavailable = False
try:
    import matplotlib.pyplot as _plt  # noqa: F401
except ModuleNotFoundError:
    mpl_unavailable = True

@pytest.fixture(scope="module")
def simple_topo():
    """Folded-топология на 3 резонатора → K = order+ports = 5."""
    return get_topology("folded", order=3)          # 3 res + 2 ports


@pytest.fixture(scope="module")
def m_vals():
    """Произвольный набор M-коэффициентов (самих значений хватит для smoke-тестов)."""
    return {
        "M1_2": 0.9,
        "M2_3": 1.0,
        "M1_3": 0.5,
        "M1_4": 0.8,
        "M3_5": 0.7,          # связи с портами
    }

@pytest.fixture(scope="module")
def cm_obj(simple_topo, m_vals):
    return CouplingMatrix(simple_topo, m_vals, Q=6000)

Ω = np.linspace(-2.0, 2.0, 101)
# ———————————————————————————————————————————————————————————————
#          _build_M_complex  — корректный учёт добротностей
# ———————————————————————————————————————————————————————————————
torch_unavailable = False
try:
    import torch  # noqa: F401  (нужен только факт импорта)
except ModuleNotFoundError:
    torch_unavailable = True


@pytest.mark.parametrize("Q", [None, 7000, [7000, 7100, 7200]])
@pytest.mark.parametrize(
    "backend",
    [
        "numpy",
        pytest.param(
            "torch",
            marks=pytest.mark.skipif(torch_unavailable, reason="torch not available"),
        ),
    ],
)
def test_build_M_complex_diag_loss(Q, backend, simple_topo):
    """
    • возвращаемый dtype — complex64;
    • на диагонали резонаторов появилось −j / (2 Q);
    • корректно обрабатываются Q = None | скаляр | список.
    """
    xp = __import__(backend)

    K = simple_topo.size
    M_real = xp.zeros((K, K), dtype=xp.float32)
    M_real[0, 1] = M_real[1, 0] = 1.23   # хотя бы один ненулевой Mij

    M_c = _build_M_complex(simple_topo, M_real, Q, xp, xp.complex64)
    assert M_c.dtype == xp.complex64
    assert M_c.shape == (K, K)

    # воображаемая часть диагонали (только резонаторы)
    if backend == "torch":
        diag_im = (
            M_c.imag.diagonal(offset=0, dim1=-2, dim2=-1)[..., : simple_topo.order]
        ).abs()
    else:  # NumPy
        diag_im = (
            M_c.imag.diagonal(offset=0, axis1=-2, axis2=-1)[..., : simple_topo.order]
        ).astype(float).__abs__()

    if Q is None:
        expected = xp.zeros_like(diag_im)
    elif isinstance(Q, (list, tuple)):
        expected = xp.asarray(1.0 / (2.0 * xp.asarray(Q)), dtype=diag_im.dtype)
    else:
        expected = xp.full(
            (simple_topo.order,), 1.0 / (2.0 * float(Q)), dtype=diag_im.dtype
        )

    assert xp.allclose(diag_im, expected)


# ———————————————————————————————————————————————————————————————
#                        Phase-helpers
# ———————————————————————————————————————————————————————————————
def test_phase_vec_and_diag_numpy():
    xp = np
    a = {1: 1.0, 2: 2.0}
    b = [0.1, 0.2]

    vec_a = _phase_vec(a, 2, xp, dtype=xp.float32)
    vec_b = _phase_vec(b, 2, xp, dtype=xp.float32)

    assert np.all(vec_a == np.array([1.0, 2.0], dtype=np.float32))
    assert vec_b.dtype == np.float32

    ω = np.linspace(-2, 2, 11)
    D = _phase_diag(xp, ω, vec_a, vec_b, xp.complex64)

    assert D.shape == (11, 2)
    assert D.dtype == np.complex64
    assert np.allclose(D[5], np.exp(-1j * vec_b).astype(np.complex64))  # ω = 0


# ———————————————————————————————————————————————————————————————
#                     cm_sparams — shape / dtype
# ———————————————————————————————————————————————————————————————
Ω = np.linspace(-1.5, 1.5, 201)


@pytest.mark.parametrize("method", ["inv", "solve"])
def test_numpy_shape_and_dtype(cm_obj, method):
    S = cm_obj.sparams(Ω, backend="numpy", method=method)
    assert S.shape == (201, 2, 2)
    assert S.dtype == np.complex64


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                torch_unavailable or not __import__("torch").cuda.is_available(),
                reason="CUDA недоступна или Torch не установлен",
            ),
        ),
    ],
)
def test_torch_vs_numpy_close(cm_obj, device):
    import torch

    ω_t = torch.as_tensor(Ω, device=device)
    S_np = cm_obj.sparams(Ω, backend="numpy")
    S_th = cm_obj.sparams(ω_t, backend="torch", device=device).cpu().numpy()
    np.testing.assert_allclose(S_np, S_th, rtol=1e-5, atol=1e-5)


def test_fix_sign_flag(cm_obj):
    S_pos = cm_obj.sparams(Ω, backend="numpy", fix_sign=True)
    S_raw = cm_obj.sparams(Ω, backend="numpy", fix_sign=False)
    assert np.allclose(S_pos[..., 0, 1], -S_raw[..., 0, 1])


# ———————————————————————————————————————————————————————————————
#                      Cache — одни и те же объекты
# ———————————————————————————————————————————————————————————————
def test_cached_mats_identity(simple_topo):
    import numpy as xp

    U1, R1, *_ = _get_cached_mats(
        simple_topo.order, simple_topo.ports, xp, dtype=xp.complex64
    )
    U2, R2, *_ = _get_cached_mats(
        simple_topo.order, simple_topo.ports, xp, dtype=xp.complex64
    )
    assert U1 is U2 and R1 is R2


# ———————————————————————————————————————————————————————————————
#      CouplingMatrix: tensor_M, serialization round-trip
# ———————————————————————————————————————————————————————————————
def test_tensor_M_backend(cm_obj):
    M_np = cm_obj._tensor_M("numpy")

    if torch_unavailable:
        pytest.skip("Torch не установлен")

    import torch

    M_th = cm_obj._tensor_M("torch", device="cpu")
    np.testing.assert_allclose(M_np, M_th.numpy(), rtol=0, atol=0)


def test_serialization_roundtrip(cm_obj):
    """
    • to_dict() → from_dict() возвращает эквивалентный объект;
    • результирующие S-параметры совпадают.
    """
    dct = cm_obj.to_dict()
    cm_back = CouplingMatrix.from_dict(cm_obj.topo, dct)
    np.testing.assert_allclose(
        cm_obj.sparams(Ω, backend="numpy"),
        cm_back.sparams(Ω, backend="numpy"),
    )


# ———————————————————————————————————————————————————————————————
#                       Негативные кейсы / ошибки
# ———————————————————————————————————————————————————————————————
def test_bad_Q_length(simple_topo):
    with pytest.raises(ValueError):
        _build_M_complex(
            simple_topo, np.zeros((5, 5)), [1, 2], np, np.complex64
        )   # order=3, а Q-элементов = 2


def test_bad_phase_length():
    with pytest.raises(ValueError):
        _phase_vec([1.0], 2, np, dtype=np.float32)   # ports = 2 → длина 1 некорректна


def test_bad_backend(cm_obj):
    with pytest.raises(ValueError):
        cm_obj.sparams(Ω, backend="foobar")   # неизвестный backend

# ————————————————————————————————————————————————————————————————
#                     MatrixLayout / permutation
# ————————————————————————————————————————————————————————————————

def _compare_sl_tail(cm: CouplingMatrix):
    """Helper: SL↔TAIL должны давать одну и ту же S‑матрицу."""
    S_tail = cm.sparams(Ω, backend="numpy")
    M_sl   = cm.to_matrix(MatrixLayout.SL)
    cm_sl  = CouplingMatrix.from_matrix(
        M_sl,
        topo=cm.topo,
        layout=MatrixLayout.SL,
        Q=cm.Q,  # ← передаём те же потери
        phase_a=cm.phase_a,
        phase_b=cm.phase_b,
    )
    S_sl   = cm_sl.sparams(Ω, backend="numpy")
    np.testing.assert_allclose(S_tail, S_sl, rtol=1e-6, atol=1e-6)



def test_layout_sl_tail_equivalence(cm_obj):
    """Быстрый smoke: SL‑матрица после импорта даёт те же S‑парам."""
    _compare_sl_tail(cm_obj)


# ————————————————————————————————————————————————————————————————
#                 from_matrix: симметризация / ошибки
# ————————————————————————————————————————————————————————————————

def test_from_matrix_force_sym(cm_obj):
    """Искажаем симметрию → force_sym=True чинит без ошибок."""
    M = cm_obj.to_matrix()
    M[0, 1] += 1e-6  # маленькое нарушение
    cm_fixed = CouplingMatrix.from_matrix(M, topo=cm_obj.topo, force_sym=True)
    S_fix = cm_fixed.sparams(Ω, backend="numpy")
    S_ref = cm_obj.sparams(Ω, backend="numpy")
    np.testing.assert_allclose(S_fix, S_ref, rtol=5e-4, atol=5e-4)


def test_from_matrix_sym_error(cm_obj):
    """При большом нарушении и force_sym=False бросается ValueError."""
    M = cm_obj.to_matrix()
    M[0, 2] += 0.1  # крупное искажение
    with pytest.raises(ValueError):
        CouplingMatrix.from_matrix(M, topo=cm_obj.topo, force_sym=False)


# ————————————————————————————————————————————————————————————————
#                       plot_matrix (matplotlib)
# ————————————————————————————————————————————————————————————————

@pytest.mark.skipif(mpl_unavailable, reason="matplotlib not installed")
def test_plot_matrix_returns_fig(cm_obj):
    fig = cm_obj.plot_matrix(layout=MatrixLayout.SL, log=True, annotate=False, figsize=(4, 4))
    # Проверяем, что figure создан и содержит AxesImage
    assert fig.axes, "Figure has no axes"
    img_count = sum(1 for ax in fig.axes for im in ax.images)
    assert img_count == 1
    import matplotlib.pyplot as plt
    plt.close(fig)


# ————————————————————————————————————————————————————————————————
#                 4. Torch vs NumPy (новая реализация)
# ————————————————————————————————————————————————————————————————

@pytest.mark.skipif(torch_unavailable, reason="torch not installed")
def test_torch_matches_numpy(cm_obj):
    import torch
    ω_t = torch.as_tensor(Ω, device="cpu")
    S_np = cm_obj.sparams(Ω, backend="numpy")
    S_th = cm_obj.sparams(ω_t, backend="torch", device="cpu").cpu().numpy()
    np.testing.assert_allclose(S_np, S_th, rtol=1e-5, atol=1e-5)


# ————————————————————————————————————————————————————————————————
#                 Файл I/O (ascii + json, SL и TAIL)
# ————————————————————————————————————————————————————————————————

def _roundtrip_io(cm: CouplingMatrix, layout: MatrixLayout, fmt: str):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"cm.{fmt}"
        cmio.write_matrix(cm, path, layout=layout, fmt=fmt)
        cm_back = cmio.read_matrix(path, topo=cm.topo, layout="auto")
        S1 = cm.sparams(Ω)
        S2 = cm_back.sparams(Ω)
        # ASCII-round-trip неизбежно даёт ~4·10⁻⁴ на |S|
        np.testing.assert_allclose(S1, S2, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("fmt", ["ascii", "json"])
@pytest.mark.parametrize("layout", [MatrixLayout.TAIL, MatrixLayout.SL])
def test_io_roundtrip(cm_obj, layout, fmt):
    _roundtrip_io(cm_obj, layout, fmt)


# ————————————————————————————————————————————————————————————————
#                     Перестановка CUSTOM
# ————————————————————————————————————————————————————————————————

def test_custom_permutation(cm_obj):
    order, ports = cm_obj.topo.order, cm_obj.topo.ports
    # z‑образная перестановка: R2 R1 P2 P1 R3
    perm: Sequence[int] = [1, 0, order + 1, order, 2]
    M_custom = cm_obj.to_matrix(MatrixLayout.CUSTOM, permutation=perm)
    cm_back = CouplingMatrix.from_matrix(
        M_custom,
        topo=cm_obj.topo,
        layout=MatrixLayout.CUSTOM,
        permutation=perm,
        Q=cm_obj.Q,  # ← передаём те же потери
        phase_a=cm_obj.phase_a,
        phase_b=cm_obj.phase_b,
    )
    np.testing.assert_allclose(
        cm_obj.sparams(Ω), cm_back.sparams(Ω), rtol=1e-6, atol=1e-6
    )