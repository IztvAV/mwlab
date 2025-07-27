# tests/test_cm_core.py
"""pytest-тесты для модуля mwlab.filters.cm_core (Torch-only ядро)
====================================================================

Покрываем:
* prepare_inputs — формы, broadcast, ошибки;
* build_cmatrix   — добавление j/qu на диагонали резонаторов;
* _cached_mats    — кэширование и корректные размеры/типы;
* solve_sparams   — корректность формата S, алгоритмов solve/inv, фаз, fix_sign, autograd.

Запуск:  pytest -q tests/test_cm_core.py
"""
from __future__ import annotations

import math
import pytest
import torch

from mwlab.filters.cm_core import (
    prepare_inputs,
    build_cmatrix,
    solve_sparams,
    CoreSpec,
    _cached_mats,  # приватный, но полезно проверить базовые свойства
    CMError,
    DT_R,
    DT_C,
    DEFAULT_DEVICE,
)


# ----------------------------------------------------------------------------
# Общие фикстуры
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    # Для CI часто нужно явно зафиксировать CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def basic_spec():
    return CoreSpec(order=4, ports=2, method="auto", fix_sign=False)


@pytest.fixture()
def basic_data(device, basic_spec):
    """Возвращает (M_real, omega, qu) для простого кейса."""
    order, ports = basic_spec.order, basic_spec.ports
    K = order + ports
    M_real = torch.zeros(K, K, dtype=DT_R, device=device)
    # Связи: 1-2, 2-3, 3-4, 1-P1, 4-P2
    M_real[0, 1] = M_real[1, 0] = 1.0
    M_real[1, 2] = M_real[2, 1] = 0.9
    M_real[2, 3] = M_real[3, 2] = 1.0
    M_real[0, 4] = M_real[4, 0] = 0.8
    M_real[3, 5] = M_real[5, 3] = 0.9

    omega = torch.linspace(-3.0, 3.0, 801, device=device, dtype=DT_R)
    qu = torch.full((order,), 700.0, dtype=torch.float64, device=device)
    return M_real, omega, qu


# ----------------------------------------------------------------------------
# prepare_inputs
# ----------------------------------------------------------------------------

def test_prepare_inputs_shapes_and_broadcast(device, basic_spec):
    order, ports = basic_spec.order, basic_spec.ports
    K = order + ports

    # Батч: 5 экземпляров, частота одна, qu скаляр, phase_a/b None
    M_real = torch.zeros(5, K, K, dtype=DT_R, device=device)
    omega = torch.tensor([0.0], dtype=DT_R, device=device)
    qu = 1000.0

    prep = prepare_inputs(M_real, omega, order=order, ports=ports, qu=qu, device=device)

    assert prep.M_real.shape == (5, K, K)
    assert prep.omega.shape == (5, 1)
    assert prep.batch_shape == (5,)
    assert prep.F == 1
    assert prep.qu.shape[-1] == order
    assert prep.qu.shape[:-1] == prep.batch_shape
    assert prep.batch_shape == () or prep.batch_shape == (5,)
    assert prep.F == 1 and prep.K == K


def test_prepare_inputs_errors(device, basic_spec):
    order, ports = basic_spec.order, basic_spec.ports
    K = order + ports
    M_real = torch.zeros(K, K, dtype=DT_R, device=device)
    omega = torch.linspace(-1, 1, 11, device=device)

    # Неверный хвост у M_real
    with pytest.raises(CMError):
        prepare_inputs(M_real[:-1], omega, order=order, ports=ports, device=device)

    # Неверный размер qu
    with pytest.raises(CMError):
        prepare_inputs(M_real, omega, order=order, ports=ports, qu=torch.ones(order+1), device=device)

    # Неверный размер phase_a
    with pytest.raises(CMError):
        prepare_inputs(M_real, omega, order=order, ports=ports, phase_a=torch.ones(ports+1), device=device)


# ----------------------------------------------------------------------------
# build_cmatrix
# ----------------------------------------------------------------------------

def test_build_cmatrix_diagonal(device, basic_spec):
    order, ports = basic_spec.order, basic_spec.ports
    K = order + ports

    M_real = torch.zeros(K, K, dtype=DT_R, device=device)
    qu = torch.arange(1, order + 1, dtype=torch.float64, device=device) * 100.0

    M_c = build_cmatrix(M_real, qu, order)
    assert M_c.dtype == DT_C
    # проверяем добавление -j/qu на диагонали первых order элементов
    diag = M_c[..., torch.arange(order, device=device), torch.arange(order, device=device)]
    expected = 1j * (1.0 / qu)
    assert torch.allclose(diag, expected.to(dtype=DT_C))

    # проверяем, что портовая диагональ не изменилась
    p_diag = M_c[..., order:, order:].diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(p_diag, torch.zeros_like(p_diag))


# ----------------------------------------------------------------------------
# _cached_mats
# ----------------------------------------------------------------------------

def test_cached_mats(device, basic_spec):
    order, ports = basic_spec.order, basic_spec.ports
    U1, R1, I1 = _cached_mats(order, ports, str(device))
    U2, R2, I2 = _cached_mats(order, ports, str(device))

    # Проверяем кэш: идентичные объекты (lru_cache вернёт те же ссылки)
    assert U1.data_ptr() == U2.data_ptr()
    assert R1.data_ptr() == R2.data_ptr()
    assert I1.data_ptr() == I2.data_ptr()

    # Проверяем формы и типы
    K = order + ports
    assert U1.shape == (K, K) and R1.shape == (K, K) and I1.shape == (ports, ports)
    assert U1.dtype == DT_C and R1.dtype == DT_C and I1.dtype == DT_C
    # У/R/I должны быть без градиентов
    assert not U1.requires_grad and not R1.requires_grad and not I1.requires_grad


# ----------------------------------------------------------------------------
# solve_sparams
# ----------------------------------------------------------------------------

def test_solve_sparams_basic(device, basic_spec, basic_data):
    M_real, omega, qu = basic_data
    S = solve_sparams(basic_spec, M_real, omega, qu=qu, device=device)

    # Форма: (F, P, P)
    assert S.shape == (omega.shape[0], basic_spec.ports, basic_spec.ports)
    assert S.dtype == torch.complex64

    # |S11| и |S21| не должны быть все нули
    mags = torch.abs(S)
    assert torch.any(mags > 0)


def test_solve_sparams_fix_sign(device, basic_spec, basic_data):
    spec = CoreSpec(order=basic_spec.order, ports=basic_spec.ports, method="auto", fix_sign=True)
    M_real, omega, qu = basic_data
    S = solve_sparams(spec, M_real, omega, qu=qu, device=device)
    # Проверим, что S[0,1] = - S_no_fix[0,1]
    spec2 = CoreSpec(order=basic_spec.order, ports=basic_spec.ports, method="auto", fix_sign=False)
    S2 = solve_sparams(spec2, M_real, omega, qu=qu, device=device)
    assert torch.allclose(S[..., 0, 1], -S2[..., 0, 1])
    assert torch.allclose(S[..., 1, 0], -S2[..., 1, 0])


def test_solve_methods_equivalence(device, basic_spec, basic_data):
    # Сравним 'solve' и 'inv' на одном примере
    spec_solve = CoreSpec(basic_spec.order, basic_spec.ports, method="solve", fix_sign=False)
    spec_inv   = CoreSpec(basic_spec.order, basic_spec.ports, method="inv",   fix_sign=False)
    M_real, omega, qu = basic_data
    S1 = solve_sparams(spec_solve, M_real, omega, qu=qu, device=device)
    S2 = solve_sparams(spec_inv,   M_real, omega, qu=qu, device=device)

    assert torch.allclose(S1, S2, atol=1e-5, rtol=1e-5)


def test_apply_phases_effect(device, basic_spec, basic_data):
    M_real, omega, qu = basic_data
    phase_a = torch.tensor([0.1, -0.05], dtype=DT_R, device=device)  # P=2
    phase_b = torch.tensor([0.2, 0.0], dtype=DT_R, device=device)

    S_no = solve_sparams(basic_spec, M_real, omega, qu=qu, device=device)
    S_ph = solve_sparams(basic_spec, M_real, omega, qu=qu, phase_a=phase_a, phase_b=phase_b, device=device)

    # Эффект фаз: диагональные элементы S11/S22 могут меняться по фазе, сравним амплитуды
    assert not torch.allclose(S_no, S_ph)
    assert torch.allclose(torch.abs(S_no), torch.abs(S_ph), atol=1e-6)


def test_autograd(device, basic_spec, basic_data):
    # Проверяем, что градиенты по M_real проходят
    M_real, omega, qu = basic_data
    M_real = M_real.clone().requires_grad_(True)

    S = solve_sparams(basic_spec, M_real, omega, qu=qu, device=device)
    # скалярная функция потерь
    loss = torch.mean(torch.abs(S[..., 0, 1])**2)
    loss.backward()
    assert M_real.grad is not None
    # градиент не должен быть весь нулевой
    assert torch.any(M_real.grad != 0)


# ----------------------------------------------------------------------------
# Дополнительные кейсы
# ----------------------------------------------------------------------------

def test_scalar_omega(device, basic_spec, basic_data):
    M_real, _, qu = basic_data
    omega = torch.tensor(0.0, dtype=DT_R, device=device)
    S = solve_sparams(basic_spec, M_real, omega, qu=qu, device=device)
    assert S.shape == (1, basic_spec.ports, basic_spec.ports)


def test_batch_inputs(device, basic_spec, basic_data):
    M_real, omega, qu = basic_data
    # Создаём батч из 3 матриц и 2 разных сеток частот (совместимых по broadcast)
    M_b = M_real.unsqueeze(0).repeat(3, 1, 1)  # (3,K,K)
    omega_b = omega.unsqueeze(0).repeat(1, 1)  # (1,F)
    qu_b = qu.unsqueeze(0).repeat(3, 1)  # (3,order)
    # Теперь batch_shape = (3,)
    S = solve_sparams(basic_spec, M_b, omega_b, qu=qu_b, device=device)
    assert S.shape == (3, omega.shape[0], basic_spec.ports, basic_spec.ports)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])  # для быстрого ручного запуска
