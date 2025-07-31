# mwlab/filters/cm_core.py
"""
cm_core.py — вычислительное ядро MWLab.filters (Torch‑only)
============================================================

Назначение
----------
Чистые, независимые от высокоуровневых классов функции для расчёта S‑параметров
по расширенной **вещественной** матрице связи M_real и нормированной частоте Ω.

Ключевые особенности:
* **Только PyTorch** (float32 / complex64).
* Полная поддержка batch‑осей: входы могут иметь произвольное число ведущих
  измерений, которые будут автоматически согласованы (broadcast).
* Автоматический выбор алгоритма решения (solve / inv).
* Кэширование неизменных матриц U, R, I_p (по ключу: order, ports, device).
* Готовность к использованию в генераторах датасетов и nn.Module (autograd «как есть»).

API (основное):
---------------
- :func:`solve_sparams` — главная функция расчёта S(Ω).
- :func:`prepare_inputs` — приведение входов к корректным типам/формам.
- :func:`build_cmatrix` — формирование комплексной матрицы M̃.

Пример
-------
>>> import torch
>>> from mwlab.filters.cm_core import solve_sparams, CoreSpec
>>> order, ports = 4, 2
>>> K = order + ports
>>> M_real = torch.zeros(K, K, dtype=torch.float32)
>>> omega  = torch.linspace(-3, 3, 801)
>>> spec   = CoreSpec(order, ports)
>>> S = solve_sparams(spec, M_real, omega)  # (801, 2, 2)

"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Optional, Tuple

import torch

# -----------------------------------------------------------------------------
#                             Общие константы / типы
# -----------------------------------------------------------------------------

class CMError(ValueError):
    """Ошибки уровня вычислительного ядра (невалидные формы, типы и т.п.)."""

# Выбор устройства по умолчанию (удобно для интерактивной работы/экспериментов)
DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Жёстко фиксированные типы под расчёт (убираем лишнюю гибкость)
DT_R = torch.float32     # real dtype
DT_C = torch.complex64   # complex dtype


@dataclass(frozen=True)
class CoreSpec:
    """Описание устройства для ядра.

    Parameters
    ----------
    order : int
        Количество резонаторов.
    ports : int
        Количество портов.
    method : {"auto","solve","inv"}
        Выбор алгоритма решения линейной системы.
    fix_sign : bool
        Инвертировать ли знак S12/S21 для 2‑портовых устройств (IEEE convention).
    """

    order: int
    ports: int
    method: str = "auto"
    fix_sign: bool = False


class Prepared(NamedTuple):
    """Результат функции :func:`prepare_inputs`.

    Attributes
    ----------
    M_real : torch.Tensor
        (..., K, K)  вещественная матрица связи (float32).
    omega : torch.Tensor
        (..., F)     нормированная частотная сетка (float32).
    qu : torch.Tensor | None
        (..., order) или None.
    phase_a, phase_b : torch.Tensor | None
        (..., ports) или None.
    batch_shape : tuple[int,...]
        Общая broadcast‑форма всех batch‑осей.
    F : int
        Количество точек частотной сетки.
    K : int
        order + ports.
    """

    M_real: torch.Tensor
    omega: torch.Tensor
    qu: torch.Tensor | None
    phase_a: torch.Tensor | None
    phase_b: torch.Tensor | None
    batch_shape: Tuple[int, ...]
    F: int
    K: int
    order: int
    ports: int


# -----------------------------------------------------------------------------
#                               Вспомогательные утилиты
# -----------------------------------------------------------------------------

def _to_device_tensor(x, *, dtype, device):
    """Приводит объект к torch.Tensor на заданном device и dtype.

    * Если x уже Tensor — переводим при необходимости на нужный device/dtype.
    * Гарантируем хотя бы 1D: скаляр → shape (1,).
    """
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t
    # list/tuple/np/скаляр
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    return t

def _batch_part(shape: Tuple[int, ...], tail_dims: int) -> Tuple[int, ...]:
    """
    Возвращает batch-часть формы (все оси, кроме последних tail_dims).
    Если tail_dims == 0, возвращает исходную форму.
    """
    if tail_dims == 0:
        return shape
    return shape[:-tail_dims] if tail_dims <= len(shape) else ()

def _broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Находит общую broadcast-форму (как NumPy/PyTorch) для набора форм.
    Пустые кортежи и None игнорируются.
    """
    shapes = [tuple(s) for s in shapes if s is not None and len(s) > 0]
    if not shapes:
        return ()
    max_len = max(len(s) for s in shapes)
    out = []
    for i in range(1, max_len + 1):          # идём справа налево
        dims = []
        for s in shapes:
            dims.append(s[-i] if len(s) >= i else 1)
        m = max(dims)
        if any(d != 1 and d != m for d in dims):
            raise CMError(f"Невозможно broadcast-ить формы: {shapes}")
        out.append(m)
    return tuple(reversed(out))

def _expand_to(t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Расширяет тензор до нужной формы через broadcast (torch.expand).

    Если целевая форма имеет больше осей — добавляем ведущие оси длиной 1.
    Уменьшать число осей нельзя. Несовместимые размерности -> CMError.
    """
    if t.shape == torch.Size(shape):
        return t
    if t.ndim > len(shape):
        raise CMError(f"cannot expand from {tuple(t.shape)} to {shape}")
    if t.ndim < len(shape):
        t = t.view((1,) * (len(shape) - t.ndim) + tuple(t.shape))
    for got, need in zip(t.shape, shape):
        if got != need and got != 1:
            raise CMError(f"cannot expand from {tuple(t.shape)} to {shape}")
    return t.expand(shape)

# -----------------------------------------------------------------------------
#                              Подготовка входов
# -----------------------------------------------------------------------------

def prepare_inputs(
    M_real,
    omega,
    *,
    order: int,
    ports: int,
    qu=None,
    phase_a=None,
    phase_b=None,
    device: str | torch.device = DEFAULT_DEVICE,
) -> Prepared:
    """
    Приводит входные данные к корректным типам и формам (Torch, float32).

    Шаги:
    1. Переводим всё в тензоры нужного dtype/device.
    2. Проверяем хвостовые размерности:
       M_real[...,K,K], omega[...,F], qu[...,order], phase*[...,ports].
    3. Вычисляем общую batch-форму и делаем expand ко всем входам.

    Возвращает Prepared.
    """
    device = torch.device(device)
    K = order + ports

    # --- 1) к тензорам ----------------------------------------------------
    M_real_t = _to_device_tensor(M_real, dtype=DT_R, device=device)
    omega_t  = _to_device_tensor(omega,  dtype=DT_R, device=device)

    # qu: скаляр -> вектор длиной order
    qu_t = None
    if qu is not None:
        if not isinstance(qu, torch.Tensor) and not hasattr(qu, "__len__"):
            qu = [float(qu)] * order
        qu_t = _to_device_tensor(qu, dtype=torch.float64, device=device)  # деление точнее

    def _prep_phase(ph):
        if ph is None:
            return None
        if not isinstance(ph, torch.Tensor) and not hasattr(ph, "__len__"):
            ph = [float(ph)] * ports
        return _to_device_tensor(ph, dtype=DT_R, device=device)

    phase_a_t = _prep_phase(phase_a)
    phase_b_t = _prep_phase(phase_b)

    # --- 2) проверки хвостов ----------------------------------------------
    if M_real_t.shape[-2:] != (K, K):
        raise CMError(f"M_real: expected (...,{K},{K}), got {tuple(M_real_t.shape)}")

    if omega_t.shape[-1] <= 0:
        raise CMError("omega: пустая частотная сетка")
    F = omega_t.shape[-1]

    if qu_t is not None and qu_t.shape[-1] != order:
        raise CMError(f"qu: expected last dim {order}, got {tuple(qu_t.shape)}")

    if phase_a_t is not None and phase_a_t.shape[-1] != ports:
        raise CMError(f"phase_a: expected last dim {ports}, got {tuple(phase_a_t.shape)}")
    if phase_b_t is not None and phase_b_t.shape[-1] != ports:
        raise CMError(f"phase_b: expected last dim {ports}, got {tuple(phase_b_t.shape)}")

    # --- 3) общая batch-форма ---------------------------------------------
    batch_shape = _broadcast_shapes(_batch_part(M_real_t.shape, 2),
        _batch_part(omega_t.shape, 1),
        _batch_part(qu_t.shape, 1) if qu_t is not None else (),
        _batch_part(phase_a_t.shape, 1) if phase_a_t is not None else (),
        _batch_part(phase_b_t.shape, 1) if phase_b_t is not None else (),
    )

    def bs(tail: Tuple[int, ...]) -> Tuple[int, ...]:
        return batch_shape + tail

    # --- 4) expand ко всем -------------------------------------------------
    M_real_t = _expand_to(M_real_t, bs((K, K)))
    omega_t  = _expand_to(omega_t,  bs((F,)))
    if qu_t is not None:
        qu_t = _expand_to(qu_t, bs((order,)))
    if phase_a_t is not None:
        phase_a_t = _expand_to(phase_a_t, bs((ports,)))
    if phase_b_t is not None:
        phase_b_t = _expand_to(phase_b_t, bs((ports,)))

    return Prepared(
        M_real=M_real_t,
        omega=omega_t,
        qu=qu_t,
        phase_a=phase_a_t,
        phase_b=phase_b_t,
        batch_shape=batch_shape,
        F=F,
        K=K,
        order=order,
        ports=ports,
    )

# -----------------------------------------------------------------------------
#                Сборка реальной матрицы связи из вектора значений M
# -----------------------------------------------------------------------------
def build_M(
    rows: torch.Tensor,
    cols: torch.Tensor,
    m_vals: torch.Tensor,
    K: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Быстро формирует плотную симметричную матрицу M_real из вектора значений верхнего
    треугольника (включая диагональ резонаторов).

    Параметры
    ---------
    rows, cols : torch.LongTensor, shape (L,)
        0-based индексы строк и столбцов для верхнего треугольника.
    m_vals : torch.Tensor, shape (..., L)
        Значения M_{ij} в порядке (rows, cols). Допускаются batch-оси слева.
    K : int
        Полный размер матрицы (order + ports).
    out : torch.Tensor | None
        Опционально — заранее созданный тензор под результат (экономия аллокаций).

    Возвращает
    ----------
    torch.Tensor
        Плотная симметричная матрица (..., K, K) типа float32.
    """
    if rows.ndim != 1 or cols.ndim != 1 or rows.shape != cols.shape:
        raise ValueError("rows/cols должны быть 1D и одинаковой длины")
    if m_vals.shape[-1] != rows.numel():
        raise ValueError("m_vals.shape[-1] должно совпадать с длиной rows/cols")

    target_shape = m_vals.shape[:-1] + (K, K)
    if out is None:
        M = torch.zeros(target_shape, dtype=m_vals.dtype, device=m_vals.device)
    else:
        if out.shape != torch.Size(target_shape):
            raise ValueError("out имеет неверную форму")
        out.zero_()
        M = out

    # Заполняем верхний треугольник
    M[..., rows, cols] = m_vals
    # Отражаем
    M[..., cols, rows] = m_vals
    return M

# -----------------------------------------------------------------------------
#                        Формирование комплексной матрицы M̃
# -----------------------------------------------------------------------------
def build_cmatrix(M_real: torch.Tensor, qu: Optional[torch.Tensor], order: int) -> torch.Tensor:
    """Строит комплексную матрицу M̃ = M_real + j*(1/qu) на диагонали резонаторов.

    * M_real: (..., K, K), float32
    * qu: (..., order) или None (float64). Если None — без потерь.
    Возвращает: (..., K, K) complex64
    """
    # базовый комплексный тензор
    M_c = M_real.to(dtype=DT_C)

    if qu is None:
        return M_c

    # j / qu  (используем float64 → cast к complex64)
    alpha_j = (1.0 / qu) * (1j)
    alpha_j = alpha_j.to(dtype=DT_C)

    # добавляем к диагонали резонаторной подматрицы
    # subdiag: (..., order)
    # В torch нельзя писать диагональ в view напрямую для батчей, но можно использовать advanced indexing
    idx = torch.arange(order, device=M_c.device)
    # индексируем последний 2D блок: (..., order, order)
    # Собираем срез резонаторной части
    # Используем ... для batch, потом idx, idx
    # Формально: M_c[..., idx, idx] += alpha_j   — поддерживается в torch >=1.10
    M_c[..., idx, idx] = M_c[..., idx, idx] + alpha_j
    return M_c


# -----------------------------------------------------------------------------
#                        Кэш неизменных матриц (U, R, I_p)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _cached_mats(order: int, ports: int, device_key: str):
    """Возвращает тройку (U, R, I_p) на device.

    Все тензоры complex64, require_grad=False (они константы).
    Ключ кэша — (order, ports, device).
    """
    device = torch.device(device_key)
    K = order + ports
    with torch.no_grad():
        U = torch.zeros((K, K), dtype=DT_C, device=device)
        R = torch.zeros((K, K), dtype=DT_C, device=device)
        I_p = torch.eye(ports, dtype=DT_C, device=device)

        # U: единичная на резонаторной части
        U[:order, :order] = torch.eye(order, dtype=DT_C, device=device)
        # R: единичная на портовой части
        R[order:, order:] = torch.eye(ports, dtype=DT_C, device=device)

        U.requires_grad_(False)
        R.requires_grad_(False)
        I_p.requires_grad_(False)
    return U, R, I_p

@lru_cache(maxsize=128)
def _cached_Icols(K: int, P: int, device_key: str) -> torch.Tensor:
    """
    Возвращает последние P столбцов единичной матрицы размера K (float → complex).
    Кэшируется по (K, P, device).
    """
    device = torch.device(device_key)
    eye = torch.eye(K, dtype=DT_C, device=device)
    cols = eye[..., -P:]  # (K,P)
    cols.requires_grad_(False)
    return cols

def clear_core_cache() -> None:
    """Сбрасывает кэш неизменяемых матриц (U, R, I_p)."""
    _cached_mats.cache_clear()
    _cached_Icols.cache_clear()

def core_cache_info():
    """Возвращает информацию о кэше (_cached_mats.cache_info())."""
    return _cached_mats.cache_info()

# -----------------------------------------------------------------------------
#                            Фазовые множители
# -----------------------------------------------------------------------------
def _apply_phases(S: torch.Tensor,
                  omega: torch.Tensor,
                  phase_a: Optional[torch.Tensor],
                  phase_b: Optional[torch.Tensor]) -> torch.Tensor:
    """Применяет диагональные фазовые множители: S' = D * S * D^T.

    D = exp(-j * (ω * a + b)), где a,b заданы покомпонентно для портов.
    Все тензоры совместимы по batch-осям.
    """
    if phase_a is None and phase_b is None:
        return S

    B_shape = S.shape[:-3]
    F = S.shape[-3]
    P = S.shape[-1]

    zero = None  # лениво не создаём zero-тензор

    a = phase_a
    b = phase_b

    # theta: (..., F, P)
    # избегаем лишних аллокаций: считаем по частям
    if a is not None:
        theta = omega[..., None] * a[..., None, :]
        if b is not None:
            theta = theta + b[..., None, :]
    else:
        # a = 0
        theta = b[..., None, :] if b is not None else None

    if theta is None:  # обе фазы = 0
        return S

    D = torch.exp(-1j * theta).to(dtype=DT_C)

    # In-place умножения, чтобы не плодить большие временные массивы
    # S = D[..., :, None] * S * D[..., None, :]
    # Левая диагональ (строки)
    S.mul_(D.unsqueeze(-1))
    # Правая диагональ (столбцы)
    S.mul_(D.unsqueeze(-2))

    return S



# -----------------------------------------------------------------------------
#                           Главная функция ядра
# -----------------------------------------------------------------------------

def solve_sparams(
    spec: CoreSpec,
    M_real,
    omega,
    *,
    qu=None,
    phase_a=None,
    phase_b=None,
    device: str | torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """Расчёт комплексной матрицы S(Ω) для заданной матрицы связи.

    Parameters
    ----------
    spec : CoreSpec
        Описание устройства (order, ports, метод, fix_sign).
    M_real : array-like | torch.Tensor
        (..., K, K) вещественная матрица связи (K = order + ports).
    omega : array-like | torch.Tensor
        (..., F) нормированная частотная сетка.
    qu : None | scalar | array-like
        Приведённые добротности резонаторов (broadcastable до (..., order)).
    phase_a, phase_b : None | scalar | array-like
        Коэффициенты фазовых линий (broadcastable до (..., ports)).
    device : torch.device | str
        Устройство вычислений.

    Returns
    -------
    S : torch.Tensor
        (..., F, P, P) комплексная матрица S (complex64).
    """
    order, ports = spec.order, spec.ports
    device = torch.device(device)

    # 1) Подготовка входов
    prep = prepare_inputs(M_real, omega,
                          order=order, ports=ports,
                          qu=qu, phase_a=phase_a, phase_b=phase_b,
                          device=device)

    # 2) Комплексная матрица
    M_c = build_cmatrix(prep.M_real, prep.qu, order)  # (..., K, K)

    # 3) Константы
    U, R, I_p = _cached_mats(order, ports, str(device))  # (K,K),(K,K),(P,P)

    # Добавим batch-оси к U/R и ось F для всех KxK матриц
    # batch_ndim = prep.M_real.ndim - 2
    batch_ndim = M_c.ndim - 2  # сколько ведущих осей до K,K

    def _expand_const(mat):
        t = mat
        for _ in range(batch_ndim):
            t = t.unsqueeze(0)
        return t.unsqueeze(-3)  # ось F

    U_b = _expand_const(U)  # (..., 1, K, K)
    R_b = _expand_const(R)  # (..., 1, K, K)
    M_b = M_c.unsqueeze(-3)  # (..., 1, K, K)

    w = prep.omega.unsqueeze(-1).unsqueeze(-1)  # (..., F, 1, 1)
    A = R_b + (1j) * w * U_b - (1j) * M_b  # (..., F, K, K)

    # 5) Решение
    use_solve = (spec.method == "solve") or (spec.method == "auto" and ports <= 4)
    K = prep.K
    P = ports

    if use_solve:
        I_cols = _cached_Icols(K, P, str(device))
        B = I_cols.expand(A.shape[:-2] + I_cols.shape)  # (..., F, K, P)
        X = torch.linalg.solve(A, B)                    # (..., F, K, P)
        A_pp = X[..., -P:, :]                           # (..., F, P, P)
    else:
        A_inv = torch.linalg.inv(A)
        A_pp = A_inv[..., -P:, -P:]  # (..., F, P, P)

    I_p_sh = I_p.view((1,) * (A_pp.ndim - 2) + I_p.shape)
    S = I_p_sh - 2.0 * A_pp

    if spec.fix_sign and ports == 2:
        # чтобы не трогать кэш/grad, делаем копию только если требуется
        S = S.clone()
        S[..., 0, 1] *= -1.0
        S[..., 1, 0] *= -1.0

    S = _apply_phases(S, prep.omega, prep.phase_a, prep.phase_b)

    # Если batch_shape пуст и размер первой оси == 1, выжмем её
    #if prep.batch_shape == () and S.shape[0] == 1:
    #    S = S.squeeze(0)
    return S

# Алиас с более «говорящим» именем (если нужно)
sparams_core = solve_sparams

__all__ = [
    "CMError",
    "DEFAULT_DEVICE",
    "DT_R",
    "DT_C",
    "CoreSpec",
    "Prepared",
    "prepare_inputs",
    "build_cmatrix",
    "build_M",
    "solve_sparams",
    "sparams_core",
    "clear_core_cache",
    "core_cache_info",
]
