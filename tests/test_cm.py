# tests/test_cm.py
"""
Юнит-тесты для mwlab.filters.cm
-------------------------------

Запуск:
    pytest -q tests/test_cm.py
"""
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from mwlab.filters.topologies import get_topology
from mwlab.filters.cm import (
    _build_M_complex_batched,
    _match_and_broadcast_inputs,
    _get_cached_mats,
    cm_forward,
    cm_sparams,
    CouplingMatrix,
    MatrixLayout,
)
from mwlab.filters import io as cmio
from mwlab.filters.io import _guess_layout


# ───────────────────────────────────────────────────────────────────────
#                              Fixtures
# ───────────────────────────────────────────────────────────────────────

try:
    import torch  # noqa: F401
    torch_unavailable = False
except ModuleNotFoundError:
    torch_unavailable = True

try:
    import matplotlib.pyplot as _plt  # noqa: F401
    mpl_unavailable = False
except ModuleNotFoundError:
    mpl_unavailable = True


@pytest.fixture(scope="module")
def simple_topo():
    """Folded‑топология на 3 резонатора → K = order+ports = 5."""
    return get_topology("folded", order=3)


@pytest.fixture(scope="module")
def m_vals():
    """Произвольный набор M‑коэффициентов (значения важны только для smoke)."""
    return {
        "M1_2": 0.9,
        "M2_3": 1.0,
        "M1_3": 0.5,
        "M1_4": 0.8,
        "M3_5": 0.7,
    }


@pytest.fixture(scope="module")
def cm_obj(simple_topo, m_vals):
    # 6 000 — нормированная добротность (qu), а не физический Q
    return CouplingMatrix(simple_topo, m_vals, qu=6000)


Ω = np.linspace(-2.0, 2.0, 101)
Ω2 = np.linspace(-1.5, 1.5, 201)


# ───────────────────────────────────────────────────────────────────────
# helper: модуль мнимой части диагонали резонаторной подматрицы
# ───────────────────────────────────────────────────────────────────────
def _imag_diag_res(M, order, xp):
    if xp.__name__ == "torch":
        d = xp.diagonal(M[..., :order, :order], dim1=-2, dim2=-1)
        return xp.abs(d)
    d = np.diagonal(M[..., :order, :order], axis1=-2, axis2=-1)
    return np.abs(d)


# ───────────────────────────────────────────────────────────────────────
#   _build_M_complex_batched — добротности
# ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("qu", [None, 7000, [7000, 7100, 7200]])
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
def test_build_M_complex_batched_diag_loss(qu, backend, simple_topo):
    xp = __import__(backend)

    K = simple_topo.size
    M_real = xp.zeros((K, K), dtype=xp.float32)
    M_real[0, 1] = M_real[1, 0] = 1.23

    M_c = _build_M_complex_batched(simple_topo, M_real, qu, xp, xp.complex64)
    assert M_c.dtype == xp.complex64
    assert M_c.shape == (K, K)

    diag_im = _imag_diag_res(M_c, simple_topo.order, xp)

    if qu is None:
        expected = xp.zeros_like(diag_im)
    elif isinstance(qu, (list, tuple)):
        expected = xp.asarray(1.0 / xp.asarray(qu), dtype=diag_im.dtype)
    else:
        expected = xp.full((simple_topo.order,), 1.0 / float(qu), dtype=diag_im.dtype)

    assert xp.allclose(diag_im, expected)


# ───────────────────────────────────────────────────────────────────────
#   _match_and_broadcast_inputs — формы/broadcast/типы
# ───────────────────────────────────────────────────────────────────────
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
def test_match_and_broadcast_inputs_shapes(simple_topo, backend):
    xp = __import__(backend)
    order, ports = simple_topo.order, simple_topo.ports
    K = order + ports
    F = 17

    # Сделаем совместимые batch-формы:
    # M_real : (3, 2, K, K)
    # omega  : (3, 1, F)
    # qu     : скаляр
    # phase_a: (1, 1, ports)
    # phase_b: None
    M_real = xp.zeros((3, 2, K, K), dtype=xp.float32)

    vec = xp.linspace(-1, 1, F, dtype=xp.float32)  # (F,)
    omega = xp.reshape(xp.stack([vec, vec, vec], axis=0), (3, 1, F))

    qu = 5000.0
    phase_a = xp.ones((1, 1, ports), dtype=xp.float32)
    phase_b = None

    ret = _match_and_broadcast_inputs(
        simple_topo, M_real, omega, qu, phase_a, phase_b, xp, xp.complex64
    )
    (
        M_b,
        om_b,
        qu_b,
        pa_b,
        pb_b,
        B_shape,
        F_out,
        K_out,
        order_out,
        ports_out,
    ) = ret

    assert B_shape == (3, 2)
    assert M_b.shape == (3, 2, K, K)
    assert om_b.shape == (3, 2, F)
    assert qu_b.shape == (3, 2, order)
    assert pa_b.shape == (3, 2, ports)
    assert pb_b is None
    assert F_out == F and K_out == K and order_out == order and ports_out == ports


# ───────────────────────────────────────────────────────────────────────
#    cm_forward vs cm_sparams — эквивалентность
# ───────────────────────────────────────────────────────────────────────
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
def test_cm_forward_equiv_cm_sparams(cm_obj, backend):
    xp = __import__(backend)
    omega = xp.asarray(Ω, dtype=xp.float32)

    params = {
        "M_real": cm_obj._tensor_M(backend, device="cpu"),
        "qu": cm_obj.qu,
        "phase_a": cm_obj.phase_a,
        "phase_b": cm_obj.phase_b,
    }

    S1 = cm_forward(cm_obj.topo, params, omega, backend=backend, device="cpu")
    S2 = cm_sparams(
        cm_obj.topo,
        params["M_real"],
        omega,
        qu=params["qu"],
        phase_a=params["phase_a"],
        phase_b=params["phase_b"],
        backend=backend,
        device="cpu",
    )

    if backend == "torch":
        S1 = S1.cpu().numpy()
        S2 = S2.cpu().numpy()
    np.testing.assert_allclose(S1, S2, rtol=1e-6, atol=1e-6)


# ───────────────────────────────────────────────────────────────────────
#  Broadcast smoke: батч только в omega, скалярные qu/phase
# ───────────────────────────────────────────────────────────────────────
def test_broadcast_scalar_qu_phase(simple_topo):
    order, ports = simple_topo.order, simple_topo.ports
    K = order + ports

    M_real = np.zeros((K, K), dtype=np.float32)
    omega = np.linspace(-2, 2, 50, dtype=np.float32)[None, :]  # (1, F)

    params = {"M_real": M_real, "qu": 6000.0, "phase_a": 0.1, "phase_b": None}
    S = cm_forward(simple_topo, params, omega, backend="numpy")
    assert S.shape == (1, 50, ports, ports)


# ───────────────────────────────────────────────────────────────────────
#        Проверка фаз: скалярные и векторные коэффициенты phase_a/b
# ───────────────────────────────────────────────────────────────────────
def test_phase_application_numpy(cm_obj):
    omega = np.array([0.0], dtype=np.float32)
    base = cm_obj.sparams(omega, backend="numpy")

    params = {
        "M_real": cm_obj._tensor_M("numpy"),
        "qu": cm_obj.qu,
        "phase_a": [0.3, 0.7],
        "phase_b": [0.1, 0.2],
    }
    S_ph = cm_forward(cm_obj.topo, params, omega, backend="numpy")

    b = np.array(params["phase_b"], dtype=np.float32)
    D = np.exp(-1j * b).astype(np.complex64)
    expected = D[:, None] * base[0] * D[None, :]
    np.testing.assert_allclose(S_ph[0], expected, rtol=1e-6, atol=1e-6)


# ───────────────────────────────────────────────────────────────────────
#                     cm_sparams — shape / dtype
# ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["inv", "solve"])
def test_numpy_shape_and_dtype(cm_obj, method):
    S = cm_obj.sparams(Ω2, backend="numpy", method=method)
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

    ω_t = torch.as_tensor(Ω2, device=device)
    S_np = cm_obj.sparams(Ω2, backend="numpy")
    S_th = cm_obj.sparams(ω_t, backend="torch", device=device).cpu().numpy()
    np.testing.assert_allclose(S_np, S_th, rtol=1e-5, atol=1e-5)


def test_fix_sign_flag(cm_obj):
    S_pos = cm_obj.sparams(Ω2, backend="numpy", fix_sign=True)
    S_raw = cm_obj.sparams(Ω2, backend="numpy", fix_sign=False)
    assert np.allclose(S_pos[..., 0, 1], -S_raw[..., 0, 1])


def test_default_fix_sign_is_false(cm_obj):
    S_def = cm_obj.sparams(Ω2)
    S_raw = cm_obj.sparams(Ω2, fix_sign=False)
    np.testing.assert_allclose(S_def, S_raw)


# ───────────────────────────────────────────────────────────────────────
#                      Cache — одни и те же объекты
# ───────────────────────────────────────────────────────────────────────
def test_cached_mats_identity(simple_topo):
    import numpy as xp

    U1, R1, *_ = _get_cached_mats(
        simple_topo.order, simple_topo.ports, xp, dtype=xp.complex64
    )
    U2, R2, *_ = _get_cached_mats(
        simple_topo.order, simple_topo.ports, xp, dtype=xp.complex64
    )
    assert U1 is U2 and R1 is R2


# ───────────────────────────────────────────────────────────────────────
#      CouplingMatrix: tensor_M, serialization round‑trip
# ───────────────────────────────────────────────────────────────────────
def test_tensor_M_backend(cm_obj):
    M_np = cm_obj._tensor_M("numpy")

    if torch_unavailable:
        pytest.skip("Torch не установлен")

    import torch

    M_th = cm_obj._tensor_M("torch", device="cpu")
    np.testing.assert_allclose(M_np, M_th.numpy(), rtol=0, atol=0)


def test_serialization_roundtrip(cm_obj):
    dct = cm_obj.to_dict()
    cm_back = CouplingMatrix.from_dict(cm_obj.topo, dct)
    np.testing.assert_allclose(
        cm_obj.sparams(Ω2, backend="numpy"),
        cm_back.sparams(Ω2, backend="numpy"),
        rtol=1e-7,
        atol=1e-7,
    )


# ───────────────────────────────────────────────────────────────────────
#         to_matrix должен работать и на Torch‑backend
# ───────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(torch_unavailable, reason="torch not installed")
def test_to_matrix_torch_backend(cm_obj):
    import torch
    M_th = cm_obj.to_matrix(backend="torch", device="cpu")
    assert isinstance(M_th, torch.Tensor)
    assert M_th.dtype == torch.float32
    M_np = cm_obj.to_matrix(backend="numpy")
    np.testing.assert_allclose(M_np, M_th.numpy(), rtol=0, atol=0)


# ───────────────────────────────────────────────────────────────────────
#                       Негативные кейсы / ошибки
# ───────────────────────────────────────────────────────────────────────
def test_bad_qu_length(simple_topo):
    with pytest.raises(ValueError):
        _build_M_complex_batched(
            simple_topo, np.zeros((5, 5), dtype=np.float32), [1, 2], np, np.complex64
        )


def test_bad_phase_length_matcher(simple_topo):
    K = simple_topo.size
    M_real = np.zeros((K, K), dtype=np.float32)
    omega = np.linspace(-1, 1, 11, dtype=np.float32)
    phase_a = np.array([0.1], dtype=np.float32)  # неверная длина
    with pytest.raises(ValueError):
        _match_and_broadcast_inputs(simple_topo, M_real, omega, None, phase_a, None, np, np.complex64)


def test_bad_backend(cm_obj):
    with pytest.raises(ValueError):
        cm_obj.sparams(Ω2, backend="foobar")


def test_incompatible_broadcast_raises(simple_topo):
    # Несовместимые batch-формы: (2, K, K) vs (3, F)
    K = simple_topo.size
    order = simple_topo.order
    M_real = np.zeros((2, K, K), dtype=np.float32)
    omega = np.linspace(-1, 1, 33, dtype=np.float32)[None, :]  # (1, 33)
    qu = np.ones((4, order), dtype=np.float64)                 # (4, order) → несовместимая ось 4

    with pytest.raises((ValueError, RuntimeError)):
        _match_and_broadcast_inputs(simple_topo, M_real, omega, qu, None, None, np, np.complex64)


@pytest.mark.parametrize("F", [1, 5])
def test_singleton_dimensions(simple_topo, F):
    K = simple_topo.size
    M_real = np.zeros((K, K), dtype=np.float32)
    omega = np.linspace(-1, 1, F, dtype=np.float32)
    # Добавим потери, чтобы избежать сингулярности
    params = {"M_real": M_real, "qu": 6000.0}
    S = cm_forward(simple_topo, params, omega, backend="numpy")
    assert S.shape == (F, simple_topo.ports, simple_topo.ports)


# ───────────────────────────────────────────────────────────────────────
#                     MatrixLayout / permutation
# ───────────────────────────────────────────────────────────────────────
def _compare_sl_tail(cm: CouplingMatrix):
    S_tail = cm.sparams(Ω2, backend="numpy")
    M_sl = cm.to_matrix(MatrixLayout.SL)
    cm_sl = CouplingMatrix.from_matrix(
        M_sl,
        topo=cm.topo,
        layout=MatrixLayout.SL,
        qu=cm.qu,
        phase_a=cm.phase_a,
        phase_b=cm.phase_b,
    )
    S_sl = cm_sl.sparams(Ω2, backend="numpy")
    np.testing.assert_allclose(S_tail, S_sl, rtol=1e-6, atol=1e-6)


def test_layout_sl_tail_equivalence(cm_obj):
    _compare_sl_tail(cm_obj)


# ───────────────────────────────────────────────────────────────────────
#                 from_matrix: симметризация / ошибки
# ───────────────────────────────────────────────────────────────────────
def test_from_matrix_force_sym(cm_obj):
    """Искажаем симметрию → force_sym=True чинит без ощутимых ошибок."""
    M = cm_obj.to_matrix()
    M[0, 1] += 1e-6  # маленькое нарушение
    cm_fixed = CouplingMatrix.from_matrix(M, topo=cm_obj.topo, force_sym=True)
    S_fix = cm_fixed.sparams(Ω2, backend="numpy")
    S_ref = cm_obj.sparams(Ω2, backend="numpy")
    np.testing.assert_allclose(S_fix, S_ref, rtol=1e-3, atol=1e-3)


def test_from_matrix_sym_error(cm_obj):
    M = cm_obj.to_matrix()
    M[0, 2] += 0.1  # крупное искажение
    with pytest.raises(ValueError):
        CouplingMatrix.from_matrix(M, topo=cm_obj.topo, force_sym=False)


# ───────────────────────────────────────────────────────────────────────
#                       plot_matrix (matplotlib)
# ───────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(mpl_unavailable, reason="matplotlib not installed")
def test_plot_matrix_returns_fig(cm_obj):
    fig = cm_obj.plot_matrix(layout=MatrixLayout.SL, log=True, annotate=False, figsize=(4, 4))
    assert fig.axes, "Figure has no axes"
    img_count = sum(1 for ax in fig.axes for im in ax.images)
    assert img_count == 1
    import matplotlib.pyplot as plt
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
#                 Torch vs NumPy (дублирующий smoke)
# ───────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(torch_unavailable, reason="torch not installed")
def test_torch_matches_numpy(cm_obj):
    import torch
    ω_t = torch.as_tensor(Ω2, device="cpu")
    S_np = cm_obj.sparams(Ω2, backend="numpy")
    S_th = cm_obj.sparams(ω_t, backend="torch", device="cpu").cpu().numpy()
    np.testing.assert_allclose(S_np, S_th, rtol=1e-5, atol=1e-5)


# ───────────────────────────────────────────────────────────────────────
#                 Файл I/O (ascii + json, SL и TAIL)
# ───────────────────────────────────────────────────────────────────────
def _roundtrip_io(cm: CouplingMatrix, layout: MatrixLayout, fmt: str):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"cm.{fmt}"
        cmio.write_matrix(cm, path, layout=layout, fmt=fmt)
        cm_back = cmio.read_matrix(path, topo=cm.topo, layout="auto")
        S1 = cm.sparams(Ω2)
        S2 = cm_back.sparams(Ω2)
        # ASCII‑round‑trip даёт ≈ 10⁻³ на |S|
        np.testing.assert_allclose(S1, S2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("fmt", ["ascii", "json"])
@pytest.mark.parametrize("layout", [MatrixLayout.TAIL, MatrixLayout.SL])
def test_io_roundtrip(cm_obj, layout, fmt):
    _roundtrip_io(cm_obj, layout, fmt)


# ───────────────────────────────────────────────────────────────────────
#                     Перестановка CUSTOM
# ───────────────────────────────────────────────────────────────────────
def test_custom_permutation(cm_obj):
    order, ports = cm_obj.topo.order, cm_obj.topo.ports
    perm: Sequence[int] = [1, 0, order + 1, order, 2]  # Z‑образная
    M_custom = cm_obj.to_matrix(MatrixLayout.CUSTOM, permutation=perm)
    cm_back = CouplingMatrix.from_matrix(
        M_custom,
        topo=cm_obj.topo,
        layout=MatrixLayout.CUSTOM,
        permutation=perm,
        qu=cm_obj.qu,
        phase_a=cm_obj.phase_a,
        phase_b=cm_obj.phase_b,
    )
    np.testing.assert_allclose(
        cm_obj.sparams(Ω2), cm_back.sparams(Ω2), rtol=1e-6, atol=1e-6
    )


# ───────────────────────────────────────────────────────────────────────
#           JSON‑экспорт: корректная запись поля "ports"
# ───────────────────────────────────────────────────────────────────────
def test_json_exports_correct_ports(cm_obj):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cm.json")
        cmio.write_matrix(cm_obj, path, fmt="json")
        blob = json.loads(open(path, "r").read())
        assert blob["ports"] == cm_obj.topo.ports


# ───────────────────────────────────────────────────────────────────────
#                 _guess_layout — эвристика макета
# ───────────────────────────────────────────────────────────────────────
def _mk_tail(order=3, ports=3):
    K = order + ports
    M = np.zeros((K, K))
    for i in range(order - 1):
        M[i, i + 1] = M[i + 1, i] = 1.0
    for p in range(ports):
        M[order - 1, order + p] = M[order + p, order - 1] = 0.8 + 0.1 * p
    return M


def _mk_sl(order=3):
    K = order + 2
    M = np.zeros((K, K))
    for i in range(order - 1):
        M[i + 1, i + 2] = M[i + 2, i + 1] = 1.0
    M[1, 0] = M[0, 1] = 0.9
    M[-2, -1] = M[-1, -2] = 0.9
    return M


def _mk_tail_noise(order=4, ports=2, eps=1e-7):
    M = _mk_tail(order, ports)
    for i in range(order, order + ports):
        for j in range(i + 1, order + ports):
            M[i, j] = M[j, i] = eps
    return M


def test_guess_tail_multiport():
    M = _mk_tail(order=4, ports=3)
    assert _guess_layout(M) is MatrixLayout.TAIL


def test_guess_sl_twoport():
    M = _mk_sl(order=5)
    assert _guess_layout(M) is MatrixLayout.SL


def test_guess_ambiguous_returns_none():
    assert _guess_layout(np.zeros((4, 4))) is None


def test_tail_noise():
    M = _mk_tail_noise(eps=1e-7)
    assert _guess_layout(M) is MatrixLayout.TAIL


def test_tail_noise_rtol():
    M = _mk_tail_noise(eps=1e-4)
    assert _guess_layout(M, rtol=1e-3) is MatrixLayout.TAIL


def test_sl_with_corner_link():
    M = _mk_sl(order=6)
    M[1, -2] = M[-2, 1] = 0.3
    assert _guess_layout(M) is MatrixLayout.SL


def test_sl_rounding_noise():
    M = _mk_sl(order=4)
    M += 1e-6 * np.eye(M.shape[0])
    assert _guess_layout(M) is MatrixLayout.SL


@pytest.mark.parametrize("order,ports", [(2, 2), (3, 4), (5, 2)])
def test_random_matrix_returns_none(order, ports):
    K = order + ports
    rng = np.random.default_rng(order * 10 + ports)
    M = rng.standard_normal((K, K))
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    assert _guess_layout(M) is None
