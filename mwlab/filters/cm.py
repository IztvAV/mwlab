# mwlab/filters/cm.py
"""
mwlab.filters.cm
================
Высоко-производительный, backend-независимый расчёт **S-параметров**
СВЧ-фильтров, диплексеров и мультиплексеров, описываемых *вещественной*
**расширенной матрицей связи** (coupling-matrix).

Компоненты модуля
-----------------
1. **`cm_sparams`** – низкоуровневый расчёт ядра
   • вход: матрица(ы) ``M ∈ ℝ^{…,K,K}``, сетка нормированных частот
     ``ω  (…,F)``;
   • опции: вектор/скаляр приведенных (нормированных) добротностей **qu**,
     фазовые сдвиги линий *(phase_a, phase_b)*;
   • backend: **NumPy** или **PyTorch** (CPU / CUDA);
   • выход: комплексная матрица ``S (…,F,P,P)`` с dtype **complex64**.

2. **`CouplingMatrix`** – контейнер *(Topology + Mᵢⱼ + qu + phase)*
   • проверяет согласованность коэффициентов с топологией;
   • строит тензор *M* нужного backend’а;
   • сериализуется в/из dict (`to_dict / from_dict`);
   • импорт / экспорт полной матрицы (`from_matrix / to_matrix`) c
     поддержкой различных макетов портов (`MatrixLayout.TAIL`, `SL`,
     `CUSTOM`);
   • быстрая визуализация `plot_matrix()` — тепловая карта *M* с
     настраиваемой палитрой (линейная или `SymLogNorm`).

3. **`MatrixLayout`** – перечисление раскладок портов во *внешней* матрице
   ``TAIL  → [R1 … Rn  P1 … Pp]`` (внутренний формат mwlab)
   ``SL    → [S  R1 … Rn  L]`` (классический Source–Load для 2-портов)
   ``CUSTOM`` – произвольная перестановка (задаётся списком индексов).

Особенности реализации
----------------------
* Единая реализация для **NumPy** и **PyTorch** (autograd-friendly).
* Автоматический выбор алгоритма обращения (*solve* vs *inv*).
* Кэш неизменных матриц **U**, **R**, **Iₚ** снижает аллокации ≈10×.
* Гибкая обработка симметрии при импорте (`force_sym`, `atol/rtol`).
* Полная диагностируемость: подробные `ValueError` с описанием причин.

Примеры использования
----------------------

```python
import numpy as np
from mwlab.filters.topologies import get_topology
from mwlab.filters.cm         import CouplingMatrix
import matplotlib.pyplot as plt

# 1) Топология «folded» на 4 резонатора
topo = get_topology("folded", order=4)

# 2) Определяем ненулевые элементы Mij (верхний треугольник)
M_vals = {
    "M1_2": 1.00, "M2_3": 0.90, "M3_4": 1.00, "M1_4": 0.20,  # резонатор-резонатор
    "M1_5": 0.85, "M4_6": 0.95,                              # связи c портами
}

cm = CouplingMatrix(topo, M_vals, qu=[7000]*4)

# 3) Считаем S-параметры (NumPy, CPU)
ω = np.linspace(-3.0, 3.0, 801, dtype=np.float32)       # нормированная сетка
S_np = cm.sparams(ω, backend="numpy")                   # (801, 2, 2)

# 4) То же на PyTorch + CUDA (если доступен)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
ω_t    = torch.linspace(-3, 3, 801, device=device)
S_t    = cm.sparams(ω_t, backend="torch", device=device)  # (801, 2, 2)

# 5) Визуализация матрицы связи
cm.plot_matrix(layout="SL", annotate=True, log=True)
plt.show()

# 6) Импорт / экспорт матрицы связи
# экспорт в формат «Source–Load»
M_sl = cm.to_matrix(layout="SL")  # ndarray (K,K)

# обратный импорт (в том числе из ASCII-файла)
cm_back = CouplingMatrix.from_matrix(M_sl, topo=topo, layout="SL", qu=cm.qu)
```
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple, List
from enum import Enum, auto

try:                                        # matplotlib не обязателен для core
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mcolors
    import matplotlib.ticker as _mticker
    _HAS_MPL = True
except ModuleNotFoundError:                 # pragma: no cover
    _HAS_MPL = False

import numpy as _np

from mwlab.filters.topologies import Topology

# ─────────────────────────────────────────────────────────────────────────────
#                Вспомогательные утилиты  (парсинг ключей M{i}_{j})
# ─────────────────────────────────────────────────────────────────────────────
def _parse_m_key(tag: str) -> tuple[int, int]:
    """
    Разбирает строку ключа матрицы связи ``"M<i>_<j>"`` и возвращает кортеж
    1‑based индексов ``(min(i, j), max(i, j))``.

    Поддерживает диагональные элементы ― допускается ``i == j``.
    Бросает ``ValueError``, если строка не соответствует формату или
    индексы не являются положительными целыми числами.
    """
    # базовая валидация префикса
    if not tag.startswith("M"):
        raise ValueError(f"ожидался ключ 'M<i>_<j>', получено {tag!r}")

    # попытка разобрать два целых индекса
    try:
        i_str, j_str = tag[1:].split("_", 1)
        i, j = int(i_str), int(j_str)
    except Exception as exc:  # неправильный формат или непарсится int
        raise ValueError(
            f"неверный ключ матрицы связи: {tag!r}  (ожидалось 'M<i>_<j>')"
        ) from exc

    # индексы должны быть положительными
    if i <= 0 or j <= 0:
        raise ValueError(f"индексы в {tag!r} должны быть положительными")

    # гарантируем (low, high) даже при i == j
    return (i, j) if i <= j else (j, i)

# ─────────────────────────────────────────────────────────────────────────────
#                               backend resolver
# ─────────────────────────────────────────────────────────────────────────────
def _lib(backend: str, *, device="cpu"):
    """
    Возвращает кортеж ``(xp, complex_dtype, default_kwargs)`` для
    NumPy / PyTorch backend-а.
    """
    backend = backend.lower()
    if backend == "numpy":
        import numpy as xp
        return xp, xp.complex64, {}

    if backend == "torch":
        import torch as xp  # type: ignore
        dev = xp.device(device) if not isinstance(device, xp.device) else device
        return xp, xp.complex64, {"device": dev}

    raise ValueError("backend должен быть 'numpy' или 'torch'")

# ─────────────────────────────────────────────────────────────────────────────
#                 XP-хелперы: diag / eye (безопасный complex64)
# ─────────────────────────────────────────────────────────────────────────────
def _xp_eye(xp, n, *, dtype, **kw):
    """
    Создаёт единичную матрицу с dtype `complex64`.
    Для старых Torch-версий, где `torch.eye(..., complex)` не
    поддерживается, используется float32 eye с дальнейшим `.to(dtype)`.
    """
    if xp.__name__ == "numpy":
        return xp.eye(n, dtype=dtype, **kw)
    try:
        return xp.eye(n, dtype=dtype, **kw)
    except (TypeError, RuntimeError):          # complex eye не поддерживается
        return xp.eye(n, dtype=xp.float32, **kw).to(dtype)

# ─────────────────────────────────────────────────────────────────────────────
#                       Кэш неизменных матриц (U, R, I_p)
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_MATS: Dict[Tuple[int, int, str, str, str], Tuple] = {}


def _get_cached_mats(order: int, ports: int, xp, *, dtype, **kw):
    """
    Возвращает три матрицы ``(U, R, I_ports)`` с учётом backend-а,
    устройства и dtype; кэшируются для экономии времени/памяти.
    """
    key = (order, ports, xp.__name__, str(kw.get("device", "cpu")), str(dtype))
    if key in _CACHE_MATS:
        return _CACHE_MATS[key]

    K = order + ports
    I_p = _xp_eye(xp, ports, dtype=dtype, **kw)
    I_r = _xp_eye(xp, order, dtype=dtype, **kw)

    U = xp.zeros((K, K), dtype=dtype, **kw)
    U[:order, :order] = I_r

    R = xp.zeros((K, K), dtype=dtype, **kw)
    R[order:, order:] = I_p

    _CACHE_MATS[key] = (U, R, I_p)
    return _CACHE_MATS[key]

# ─────────────────────────────────────────────────────────────────────────────
#                            helpers
# ─────────────────────────────────────────────────────────────────────────────
def _broadcast_to_xp(xp, arr, target_shape):
    """
    Безопасный broadcast для NumPy и Torch.

    Parameters
    ----------
    xp : модуль numpy или torch
    arr : ndarray/Tensor
    target_shape : tuple[int]

    Returns
    -------
    ndarray/Tensor с формой target_shape, разделяющий память (torch.expand)
    или создающий view (numpy.broadcast_to).
    """
    if xp.__name__ == "torch":
        # Torch: expand умеет broadcast без копии (как numpy.broadcast_to)
        return arr.expand(target_shape)
    return xp.broadcast_to(arr, target_shape)


def _as_xp_array(xp, obj, dtype, **kw):
    """
    Приводит объект (list/ndarray/Tensor/скаляр) к массиву/тензору backend-а xp.

    * Если obj уже в нужном backend-е, просто меняем dtype/device при необходимости.
    * Torch: используем xp.as_tensor(obj, dtype=..., device=...).
    * NumPy: xp.asarray(obj, dtype=...).

    Возвращает массив/тензор с минимум 1 измерением (скаляр → shape (1,)).
    """
    if xp.__name__ == "torch":
        arr = xp.as_tensor(obj, dtype=dtype, **kw)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr
    # NumPy
    arr = xp.asarray(obj, dtype=dtype, **kw)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _infer_batch_shape(*specs):
    """
    Вычисляет общую batch-форму для входов.

    specs — последовательность кортежей вида (shape, tail_dims),
    где:
        shape: полная форма массива
        tail_dims: сколько последних осей НЕ относятся к batch (их нужно отбросить
                   перед broadсast; например K,K или order, ports, F).
    Возвращает tuple[int] — общую batch-форму.

    Пример:
        shape=(32, 128, 6, 6), tail_dims=2  → batch part = (32,128)
        shape=(1, 128), tail_dims=1         → batch part = (1,)
    """
    bshape = ()
    for shape, tail in specs:
        if shape is None:
            continue
        if tail == 0:
            b_part = shape
        else:
            b_part = shape[:-tail] if tail <= len(shape) else ()
        # используем numpy для вычисления объединения форм
        bshape = _np.broadcast_shapes(bshape, b_part)
    return bshape


def _match_and_broadcast_inputs(
    topo,
    M_real,
    omega,
    qu,
    phase_a,
    phase_b,
    xp,
    cplx,
    **kw,
):
    """
        Приводит все входы к согласованным формам (batch‑broadcast).

        Возвращает
        ----------
        M_real : (..., K, K) float32
        omega  : (..., F)    float32
        qu     : (..., order) float64 | None
        phase_a: (..., ports) float32 | None
        phase_b: (..., ports) float32 | None
        B_shape: tuple[int]
            Общая batch-форма.
        F : int
            Длина частотной сетки.
        K : int
            order + ports.
        order : int
        ports : int
        """

    order, ports, K = topo.order, topo.ports, topo.size

    # -------------------- шаг 1. Приводим к backend-типам -------------------
    f32 = xp.float32 if xp.__name__ == "torch" else _np.float32
    f64 = xp.float64 if xp.__name__ == "torch" else _np.float64

    M_real = _as_xp_array(xp, M_real, dtype=f32, **kw)
    omega  = _as_xp_array(xp, omega,  dtype=f32, **kw)

    # qu: допускаем None/скаляр/вектор(order)/батч
    if qu is not None:
        if _np.isscalar(qu):
            qu = _as_xp_array(xp, [float(qu)] * order, dtype=f64, **kw)
        else:
            qu = _as_xp_array(xp, qu, dtype=f64, **kw)

    # phase_a/b: допускаем None/скаляр/вектор(ports)/батч
    def _prep_phase(ph):
        if ph is None:
            return None
        if _np.isscalar(ph):
            return _as_xp_array(xp, [float(ph)] * ports, dtype=f32, **kw)
        return _as_xp_array(xp, ph, dtype=f32, **kw)

    phase_a = _prep_phase(phase_a)
    phase_b = _prep_phase(phase_b)

    # -------------------- шаг 2. Проверки последних размерностей -----------
    if M_real.shape[-2:] != (K, K):
        raise ValueError(f"M_real: ожидалась форма (...,{K},{K}), получено {M_real.shape}")

    if omega.shape[-1] <= 0:
        raise ValueError("omega: пустая частотная сетка")
    F = omega.shape[-1]

    if qu is not None and qu.shape[-1] != order:
        raise ValueError(f"qu: последняя ось должна быть длиной {order}, получено {qu.shape}")

    if phase_a is not None and phase_a.shape[-1] != ports:
        raise ValueError(f"phase_a: последняя ось должна быть длиной {ports}, получено {phase_a.shape}")

    if phase_b is not None and phase_b.shape[-1] != ports:
        raise ValueError(f"phase_b: последняя ось должна быть длиной {ports}, получено {phase_b.shape}")

    # -------------------- шаг 3. Вычисляем общую batch-форму ---------------
    B_shape = _infer_batch_shape(
        (M_real.shape, 2),
        (omega.shape, 1),
        ((qu.shape if qu is not None else None), 1),
        ((phase_a.shape if phase_a is not None else None), 1),
        ((phase_b.shape if phase_b is not None else None), 1),
    )

    # -------------------- шаг 4. Broadcast всех входов к B_shape -----------
    # helper для вычисления final shape: B_shape + tail_dims
    def _bs(tail):
        return B_shape + tail

    M_real = _broadcast_to_xp(xp, M_real, _bs((K, K)))
    omega  = _broadcast_to_xp(xp, omega,  _bs((F,)))

    if qu is not None:
        qu = _broadcast_to_xp(xp, qu, _bs((order,)))
    if phase_a is not None:
        phase_a = _broadcast_to_xp(xp, phase_a, _bs((ports,)))
    if phase_b is not None:
        phase_b = _broadcast_to_xp(xp, phase_b, _bs((ports,)))

    return M_real, omega, qu, phase_a, phase_b, B_shape, F, K, order, ports


def _apply_phases(xp, S, omega, phase_a, phase_b, cplx):
    """
    Умножает матрицу S на диагональные фазовые матрицы, если заданы phase_a/b.

    Формула:  S' = D(ω) · S · D(ω)^T  (эквивалентно D[..., :, None] * S * D[..., None, :])

    omega: shape B...×F
    phase_a, phase_b: shape B...×P или None
    Возвращает S комплексного типа.
    """
    if phase_a is None and phase_b is None:
        return S

    # обеспечиваем, что phase_a/b не None
    B_shape = S.shape[:-3]  # (...), затем F,P,P
    F = S.shape[-3]
    P = S.shape[-1]

    f32 = xp.float32 if xp.__name__ == "torch" else _np.float32

    zero = xp.zeros(B_shape + (P,), dtype=f32, device=getattr(S, "device", None)) \
        if xp.__name__ == "torch" \
        else xp.zeros(B_shape + (P,), dtype=f32)

    a_vec = phase_a if phase_a is not None else zero
    b_vec = phase_b if phase_b is not None else zero

    # theta: B...×F×P
    theta = omega[..., None] * a_vec[..., None, :] + b_vec[..., None, :]

    if xp.__name__ == "torch":
        D = xp.exp(-1j * theta).to(cplx)
    else:
        D = _np.exp(-1j * _np.asarray(theta)).astype(cplx)

    # умножаем: D[..., :, None] * S * D[..., None, :]
    S = D[..., :, None] * S * D[..., None, :]
    return S

def _build_M_complex_batched(topo, M_real, qu, xp, cplx, **kw):
    """
    Формирует комплексную матрицу M̃ = M_real - j/qu на диагонали резонаторов.
    Предполагается, что M_real уже имеет форму (..., K, K).
    qu может быть None | скаляр | (..., order).
    """
    # --- базовая матрица ---
    M = xp.as_tensor(M_real, dtype=cplx, **kw) if xp.__name__ == "torch" \
        else xp.asarray(M_real, dtype=cplx, **kw)

    if qu is None:
        return M

    order = topo.order

    # Приводим qu к массиву нужного backend-а и длины 'order'
    if _np.isscalar(qu):
        qu_arr = xp.full((order,), float(qu), dtype=xp.float64 if xp.__name__ == "torch" else _np.float64, **kw)
    else:
        qu_arr = xp.as_tensor(qu, dtype=xp.float64, **kw) if xp.__name__ == "torch" \
            else xp.asarray(qu, dtype=_np.float64, **kw)
    if qu_arr.shape[-1] != order:
        raise ValueError(f"qu: последняя ось должна быть длиной {order}, получено {qu_arr.shape}")

    # j / qu
    alpha_j = (1.0 / qu_arr) * (1j)
    alpha_j = alpha_j.to(cplx) if xp.__name__ == "torch" else alpha_j.astype(cplx)

    # Добавляем к диагонали резонаторной части
    sub = M[..., :order, :order]
    if xp.__name__ == "torch":
        d = xp.diagonal(sub, dim1=-2, dim2=-1)
        d += alpha_j
    else:
        idx = _np.arange(order)
        # M[..., idx, idx] возвращает копию, поэтому пишем через явное присваивание
        M[..., idx, idx] = M[..., idx, idx] + alpha_j

    return M

# ─────────────────────────────────────────────────────────────────────────────
#                              LOW-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────
def cm_forward(
    topo: Topology,
    params: Mapping[str, object],
    omega,
    *,
    backend: str = "numpy",
    device: str = "cpu",
    method: str = "auto",    # 'auto' | 'inv' | 'solve'
    fix_sign: bool = False,
):
    """
    Батч‑расчёт S-параметров для фильтров/устройств, описываемых расширенной
    вещественной матрицей связи.

    Parameters
    ----------
    topo : Topology
        Объект топологии (order, ports, links).
    params : Mapping[str, Any]
        Словарь входных параметров. Обязателен ключ "M_real".
        Поддерживаемые ключи:
            * "M_real"  : (..., K, K)  вещественная матрица связи
            * "qu"      : None | скаляр | (..., order)
            * "phase_a" : None | скаляр | (..., ports)
            * "phase_b" : None | скаляр | (..., ports)
        Все тензоры/массивы могут иметь batch-оси, которые будут
        автоматически согласованы (broadcast) между собой и с omega.
    omega : array-like / torch.Tensor
        Нормированная частота: (..., F)
    backend : {"numpy", "torch"}
        Где считать: NumPy (CPU) или Torch (CPU/GPU).
    device : str
        Устройство для Torch (e.g., "cpu", "cuda:0"). Для NumPy игнорируется.
    method : {"auto", "inv", "solve"}
        Алгоритм обращения:
          * 'solve' – решаем A X = I_pp (быстрее при малом P),
          * 'inv'   – берём обратную A^{-1},
          * 'auto'  – по умолчанию 'solve' при ports ≤ 4.
    fix_sign : bool, default False
        Для 2‑портовых фильтров True инвертирует знак S12/S21 (опциональная
        IEEE-конвенция).

    Returns
    -------
    S : xp.ndarray / torch.Tensor
        Комплексная матрица S формы (..., F, P, P), dtype complex64.
    """
    xp, cplx, kw = _lib(backend, device=device)

    # -------- извлекаем параметры -----------------------------------------
    if "M_real" not in params:
        raise ValueError("params['M_real'] отсутствует (обязательный аргумент).")

    M_real  = params["M_real"]
    qu      = params.get("qu", None)
    phase_a = params.get("phase_a", None)
    phase_b = params.get("phase_b", None)

    # -------- согласуем формы, типы, выполняем broadcast -------------------
    M_real, omega, qu, phase_a, phase_b, B_shape, F, K, order, ports = \
        _match_and_broadcast_inputs(topo, M_real, omega, qu, phase_a, phase_b, xp, cplx, **kw)

    order, ports = topo.order, topo.ports

    # -------- строим комплексную матрицу M̃ --------------------------------
    M_c = _build_M_complex_batched(topo, M_real, qu, xp, cplx, **kw)

    # -------- кеш U, R, I_ports -------------------------------------------
    U, R, I_ports = _get_cached_mats(order, ports, xp, dtype=cplx, **kw)

    # растягиваем U/R до batch-формы, если нужно (добавляем оси спереди)
    while U.ndim < M_c.ndim:
        U, R = (t[None, ...] for t in (U, R))

    j = xp.asarray(1j, dtype=cplx, **kw)
    w = omega[..., None, None]  # (..., F, 1, 1)

    # A = R + j*ω*U - j*M̃
    A = R + j * w * U - j * M_c  # (..., F, K, K)

    # -------- решаем систему ----------------------------------------------
    use_solve = (method == "solve") or (method == "auto" and ports <= 4)

    if use_solve:
        # I_cols: (K,P)
        I_cols = _xp_eye(xp, K, dtype=cplx, **kw)[..., -ports:]  # (K,P)

        # батч: (B_total*F, K, K) × (B_total*F, K, P)
        A2 = A.reshape((-1, K, K))
        B2 = _broadcast_to_xp(xp, I_cols, A2.shape[:-2] + I_cols.shape)
        # NumPy: copy(), Torch: clone() — чтобы не переписать кэш
        if xp.__name__ == "torch":
            B2 = B2.clone()
        else:
            B2 = B2.copy()

        X = xp.linalg.solve(A2, B2).reshape(A.shape[:-2] + (K, ports))
        A_pp = X[..., -ports:, :]  # (..., F, P, P)
    else:
        A_inv = xp.linalg.inv(A)
        A_pp = A_inv[..., -ports:, -ports:]  # (..., F, P, P)

    # -------- базовая S ----------------------------------------------------
    I_p_reshaped = I_ports.reshape((1,) * (A_pp.ndim - 2) + I_ports.shape)
    S = I_p_reshaped - 2.0 * A_pp

    if fix_sign and ports == 2:
        if xp.__name__ == "torch":
            S = S.clone()
        else:
            S = S.copy()
        S[..., 0, 1] *= -1.0
        S[..., 1, 0] *= -1.0

    # -------- фазовые диагонали -------------------------------------------
    S = _apply_phases(xp, S, omega, phase_a, phase_b, cplx)

    return S

# ─────────────────────────────────────────────────────────────────────────────
#                   cm_sparams (тонкая обертка над cm_forward)
# ─────────────────────────────────────────────────────────────────────────────

def cm_sparams(
    topo: Topology,
    M_real,
    omega,
    *,
    qu=None,
    phase_a=None,
    phase_b=None,
    backend: str = "numpy",
    device: str = "cpu",
    method: str = "auto",
    fix_sign: bool = False,
):
    """
    Рассчитывает комплексную S-матрицу. Является тонкой оберткой над `cm_forward`.

    Параметры
    ----------
    topo : Topology
        Топология устройства.
    M_real : array-like | torch.Tensor
        Вещественная расширенная матрица связи формы (..., K, K),
        где K = order + ports.
    omega : array-like | torch.Tensor
        Нормированная частотная сетка формы (..., F).
    qu : None | scalar | array-like
        Приведённые добротности:
            - None → без потерь,
            - скаляр → один Q_u для всех резонаторов,
            - (..., order) → индивидуальные Q_u.
    phase_a, phase_b : None | scalar | array-like
        Коэффициенты фазовых линий для портов:
            - None       → нет фазовых сдвигов,
            - скаляр     → одинаково для всех портов,
            - (..., P)   → индивидуально для каждого порта.
    backend : {"numpy", "torch"}
        Где считать (NumPy CPU или Torch CPU/GPU).
    device : str
        Устройство для Torch (например "cuda:0"); для NumPy игнорируется.
    method : {"auto", "inv", "solve"}
        Алгоритм обращения матрицы A:
            * 'solve' – решаем AX = I (быстрее при малом числе портов),
            * 'inv'   – invert (надёжно при большом P),
            * 'auto'  – 'solve' если ports ≤ 4, иначе 'inv'.
    fix_sign : bool, default False
        (Опционально) инвертирует знак S12/S21 для 2‑портовой матрицы.

    Returns
    -------
    S : ndarray | torch.Tensor
        Комплексная матрица S (..., F, P, P) c dtype complex64.
    """
    params = {
        "M_real": M_real,
        "qu": qu,
        "phase_a": phase_a,
        "phase_b": phase_b,
    }
    return cm_forward(
        topo,
        params,
        omega,
        backend=backend,
        device=device,
        method=method,
        fix_sign=fix_sign,
    )

# ─────────────────────────────────────────────────────────────────────────────
#                               MatrixLayout
# ─────────────────────────────────────────────────────────────────────────────
class MatrixLayout(Enum):  # экспортируем в __all__ ниже
    """Где располагаются портовые строки/столбцы во *внешней* матрице.

    - **TAIL** – канонический для mwlab: ``(R1 … Rn, P1 … Pp)``;
    - **SL**   – классический «Source–Load» для 2-портовых фильтров
                 ``(S, R1 … Rn, L)``;
    - **CUSTOM** – пользователь задаёт произвольную перестановку.
    """

    TAIL   = auto()
    SL     = auto()
    CUSTOM = auto()

# ----------------------------------------------------------------------------
#                  helpers: permutation ↔ MatrixLayout
# ----------------------------------------------------------------------------
def _make_perm(
    order: int,
    ports: int,
    layout: MatrixLayout,
    permutation: Sequence[int] | None = None,
):
    """
    Формирует перестановочный вектор **perm** длиной *K = order + ports*,
    описывающий взаимное расположение строк/столбцов во *внешней* матрице
    связи относительно **канонического** (TAIL) порядка
    ``[R1 … Rn,  P1 … Pp]``.

    Семантика
    ---------
    * **perm[i] = j** означает, что **i‑я строка/столбец внешней матрицы**
      (той, в которой вы сохраняете/читаeте данные) должна быть взята из
      **j‑й строки/столбца канонической матрицы**.

    Соответственно, преобразования матриц выполняются так::

        P      = eye(K)[perm]        # матрица‑коммутатор
        M_ext  = P @ M_can @ P.T     # canonical → external
        M_can  = P.T @ M_ext @ P     # external  → canonical

    Параметры
    ---------
    order : int
        Количество резонаторов *n*.
    ports : int
        Количество портов *p* (сейчас поддерживается только *p = 2* для
        макета **SL**).
    layout : MatrixLayout
        Желаемый макет *внешней* матрицы:

        * **TAIL**   – канонический порядок ``[R1 … Rn  P1 … Pp]``
          → возвращается ``[0, 1, …, K‑1]``;

        * **SL**     – классический 2‑портовый порядок
          ``[S,  R1 … Rn,  L]``
          → возвращается ``[n, 0, 1, …, n‑1, n+1]``;

        * **CUSTOM** – произвольная перестановка, задаётся
          аргументом *permutation*.

    permutation : Sequence[int] | None
        Пользовательская перестановка **только** для ``layout=CUSTOM``.
        Должна быть перестановкой чисел `0 … K‑1` без повторов.

    Возвращает
    ----------
    list[int]
        Перестановка *perm* размера *K*.

    Примеры
    -------
    >>> from mwlab.filters.cm import _make_perm, MatrixLayout
    >>> _make_perm(order=4, ports=2, layout=MatrixLayout.SL)
    [4, 0, 1, 2, 3, 5]        # 4 резонатора + 2 порта

    >>> _make_perm(3, 2, MatrixLayout.TAIL)
    [0, 1, 2, 3, 4]

    >>> _make_perm(3, 2, MatrixLayout.CUSTOM,
    ...            permutation=[3, 0, 1, 4, 2])
    [3, 0, 1, 4, 2]

    Исключения
    ----------
    ValueError
        * Неподдерживаемый макет или количество портов для **SL**;
        * `layout == CUSTOM`, но *permutation* не задана или содержит ошибки.
    """
    if layout is MatrixLayout.TAIL:
        return list(range(order + ports))

    if layout is MatrixLayout.SL:
        if ports != 2:
            raise ValueError("layout 'SL' применим только к 2‑портовым матрицам")
        # external: S,   R1…Rn,           L
        # internal: R1…Rn, P1(=S), P2(=L)
        return [order] + list(range(order)) + [order + 1]

    # CUSTOM
    if permutation is None:
        raise ValueError("CUSTOM layout требует аргумент permutation")
    perm = list(permutation)
    if len(perm) != order + ports or sorted(perm) != list(range(order + ports)):
        raise ValueError("permutation должна быть перестановкой 0…K-1")
    return perm

# ─────────────────────────────────────────────────────────────────────────────
#            CouplingMatrix — высокоуровневый контейнер / фасад
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CouplingMatrix:
    """Контейнер «Topology + Mij + qu + фазовые линии»."""
    topo: Topology
    M_vals: Mapping[str, float]
    qu: float | Sequence[float] | _np.ndarray | "torch.Tensor" | None = None
    phase_a: Mapping[int, float] | Sequence[float] | None = None
    phase_b: Mapping[int, float] | Sequence[float] | None = None

    # ------------------------------------------------------------------ init
    def __post_init__(self):
        self.topo.validate_mvals(self.M_vals, strict=False)
        if self.qu is not None and not isinstance(self.qu, (int, float)):
            try:
                qlen = len(self.qu)  # type: ignore
            except TypeError:
                qlen = None
            if qlen is not None and qlen != self.topo.order:
                raise ValueError(
                    f"qu: ожидается {self.topo.order} значений, получено {qlen}"
                )
        for label, vec in (("phase_a", self.phase_a), ("phase_b", self.phase_b)):
            if isinstance(vec, Sequence) and len(vec) != self.topo.ports:
                raise ValueError(
                    f"{label}: длина {len(vec)} ≠ ports({self.topo.ports})"
                )

    # ---------------------------------------------------------------- helpers
    def _tensor_M(self, backend="numpy", *, device="cpu"):
        xp, _, kw = _lib(backend, device=device)
        K = self.topo.size
        dtype_f32 = xp.float32 if xp.__name__ == "torch" else _np.float32
        M = xp.zeros((K, K), dtype=dtype_f32, **kw)

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for key, val in self.M_vals.items():
            i, j = _parse_m_key(key)
            rows.append(i - 1)
            cols.append(j - 1)
            vals.append(float(val))

        if not rows:  # все нули
            return M

        if xp.__name__ == "torch":
            # Torch требует LongTensor индексы и Tensor значений
            idx_r = xp.tensor(rows, dtype=xp.long, **kw)
            idx_c = xp.tensor(cols, dtype=xp.long, **kw)
            vals_t = xp.tensor(vals, dtype=dtype_f32, **kw)
            M[idx_r, idx_c] = vals_t
            M[idx_c, idx_r] = vals_t
        else:  # NumPy
            M[rows, cols] = vals
            M[cols, rows] = vals

        return M

    # ---------------------------------------------------------------- API
    def sparams(
        self,
        omega,
        *,
        backend="numpy",
        device="cpu",
        method="auto",
        fix_sign=False,
    ):
        """Вычислить S-параметры (см. :pyfunc:`cm_sparams`)."""
        return cm_sparams(
            self.topo,
            self._tensor_M(backend, device=device),
            omega,
            qu=self.qu,
            phase_a=self.phase_a,
            phase_b=self.phase_b,
            backend=backend,
            device=device,
            method=method,
            fix_sign=fix_sign,
        )

    # ---------------------------------------------------------------- dump
    def to_dict(self) -> Dict[str, float]:
        """
        Сериализация в JSON-дружелюбный словарь:
        `order`, `ports`, `topology`, все `M…`, `qu…`, `phase_a…`, `phase_b…`.
        """
        out: Dict[str, float] = {
            "order": self.topo.order,
            "ports": self.topo.ports,
            "topology": self.topo.name or "",
            **{k: float(v) for k, v in self.M_vals.items()},
        }

        # qu
        if self.qu is not None:
            if isinstance(
                self.qu, (list, tuple, _np.ndarray)
            ) or (hasattr(_np, "ndarray") and isinstance(self.qu, _np.ndarray)):
                for idx, q in enumerate(self.qu, 1):
                    out[f"qu_{idx}"] = float(q)
            else:
                out["qu"] = float(self.qu)

        # phases
        for pref, mapping in (
            ("phase_a", self.phase_a),
            ("phase_b", self.phase_b),
        ):
            if mapping is None:
                continue
            if isinstance(mapping, Mapping):
                for p, v in mapping.items():
                    out[f"{pref}{p}"] = float(v)
            else:
                for idx, v in enumerate(mapping, 1):
                    out[f"{pref}{idx}"] = float(v)
        return out

    @classmethod
    def from_dict(cls, topo: Topology | None, d: Mapping[str, float | int | str]):
        """
        Восстановление CouplingMatrix из plain-словаря **d**
        (обратное действие к :meth:`to_dict`).

        Если аргумент *topo* == None, топология автоматически
        «доопределяется» по набору ненулевых коэффициентов M{i}_{j}.
        """
        # ------------------------------- 1. M-коэффициенты --------------------
        M_vals = {k: float(v) for k, v in d.items() if k.startswith("M")}
        if not M_vals:
            raise ValueError("from_dict: нет коэффициентов 'M…'")

        # ------------------------------- 2. qu-параметры -----------------------
        if "qu" in d:
            qu = float(d["qu"])
        else:
            q_keys = sorted((k for k in d if k.startswith("qu_")),
                            key=lambda s: int(s.split("_")[1]))
            qu = [float(d[k]) for k in q_keys] if q_keys else None

        # ------------------------------- 3. Фазовые сдвиги --------------------
        phase_a = {int(k[7:]): float(v) for k, v in d.items()
                   if k.startswith("phase_a")} or None
        phase_b = {int(k[7:]): float(v) for k, v in d.items()
                   if k.startswith("phase_b")} or None

        # ------------------------------- 4. Топология -------------------------
        if topo is None:
            # --- собираем все индексы, ищем order и ports --------------------
            pairs = [_parse_m_key(k) for k in M_vals]
            indices = {idx for ij in pairs for idx in ij}  # уникальные числа

            order_in_dict = int(d.get("order", 0))
            ports_in_dict = int(d.get("ports", 2))  # если есть

            # «order» — это всё, что ≤ наибольшего резонаторного индекса;
            # если order не задан, принимаем минимальное достаточное.
            max_idx = max(indices)
            #order = max(order_in_dict, max(i for i in indices
            #                               if i <= max_idx // 2))
            # «ports» берём так, чтобы хватило до max-index
            #ports = max(ports_in_dict, max_idx - order)
            order = order_in_dict or (max(i for i, j in pairs if i == j == 0)  # диагональ
                                      or max(i for i, j in pairs if i != j and i < j))
            ports = ports_in_dict or (max_idx - order)

            # формируем links: только верхний треугольник, убираем дубли
            links = sorted(set(pairs))

            topo = Topology(order=order,
                            ports=ports,
                            links=links,
                            name="inferred")

        return cls(topo,
                   M_vals,
                   qu=qu,
                   phase_a=phase_a,
                   phase_b=phase_b)

    # ------------------------------------------------------------------ to_matrix
    def to_matrix(
        self,
        layout: MatrixLayout = MatrixLayout.TAIL,
        *,
        permutation: Sequence[int] | None = None,
        backend: str = "numpy",
        device: str = "cpu",
    ):
        """Возвращает *реальную* квадратную матрицу нужного макета."""
        order, ports = self.topo.order, self.topo.ports
        xp, _, kw = _lib(backend, device=device)
        M_can = xp.asarray(self._tensor_M(backend, device=device))
        perm = _make_perm(order, ports, layout, permutation)
        # Torch требует свой dtype; для NumPy оставляем прежний
        dtype_eye = xp.float32 if xp.__name__ == "torch" else _np.float32
        P = xp.eye(order + ports, dtype=dtype_eye, **kw)[perm]
        return P @ M_can @ P.T

    # ------------------------------------------------------------------ from_matrix
    # ──────────────────────────────────────────────────────────────────────
    #  CouplingMatrix.from_matrix – импорт из «сырых» расширенных матриц
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def from_matrix(
            cls,
            M_ext,  # (K,K) ndarray / Tensor
            *,
            topo: Topology | None = None,
            layout: MatrixLayout = MatrixLayout.TAIL,
            permutation: Sequence[int] | None = None,  # для CUSTOM
            force_sym: bool = True,
            atol: float = 1e-8,
            rtol: float = 1e-5,
            # … и сразу пробрасываем в итоговый объект:
            qu=None,
            phase_a=None,
            phase_b=None,
    ) -> "CouplingMatrix":
        """
        Создаёт :class:`CouplingMatrix` из готовой **расширенной** матрицы связи.

        Parameters
        ----------
        M_ext : array-like | torch.Tensor
            Полная матрица размером ``K×K`` (``K = n + P``).
        topo : Topology | None, default *None*
            Если *None* – топология будет автоматически восстановлена по
            ненулевым `Mij` (принимается `ports=2`).
        layout : MatrixLayout, default **TAIL**
            Раскладка строк/столбцов во входной матрице.

            * **TAIL**   – внутренний канонический вид:
              ``[R1 … Rn  P1 … Pp]``
              (ничего переставлять не нужно);

            * **SL**     – классическая 2-портовая запись
              ``[S  R1 … Rn  L]`` *(только ports=2)*;

            * **CUSTOM** – произвольная перестановка,
              задаётся параметром *permutation*.

        permutation : Sequence[int] | None
            Нумерация **0-based**.  Элемент *permutation[i]* – индекс **во
            входной** (экспортной) матрице, который должен стать *i*-й строкой
            и столбцом канонической матрицы.
            Используется **только** при `layout=CUSTOM`.

        force_sym : bool, default *True*
            Если входная матрица несимметрична (обычно после экспорта в ASCII)
            и `force_sym=True`, она делается симметричной по формуле
            ``M := 0.5 * (M + Mᵀ)``.
            При `force_sym=False` бросаем `ValueError`.

        atol, rtol : float
            Допуски для `numpy.allclose` при проверке симметричности.

        qu, phase_a, phase_b
            Приведенные добротности резонаторов и фазовые коэффициенты линий – передаются
            без изменений в конструктор :class:`CouplingMatrix`.

        Returns
        -------
        CouplingMatrix
            Объект, эквивалентный переданной матрице.

        Notes
        -----
        * Если *topo* не указан, порядок *n* и количество портов *P* будут
          «угаданы» из максимального номера узла, а связи – из всех
          ненулевых элементов **верхнего** треугольника.
        * Поддерживаются как `numpy.ndarray`, так и `torch.Tensor`; в любом
          случае дальнейшая работа идёт через NumPy ― это всего лишь импорт.
        """

        # ────────────────────────── 0. Приводим к NumPy.float32 ────────────
        if hasattr(_np, "ndarray") and isinstance(M_ext, _np.ndarray):
            M_ext_np = _np.asarray(M_ext, dtype=_np.float32)
        else:  # torch / list / tuple
            M_ext_np = _np.asarray(M_ext, dtype=_np.float32)

        if M_ext_np.ndim != 2 or M_ext_np.shape[0] != M_ext_np.shape[1]:
            raise ValueError("M_ext должна быть квадратной матрицей (K×K)")

        K: int = int(M_ext_np.shape[0])

        # ────────────────────────── 1. Симметричность ─────────────────────
        if not _np.allclose(M_ext_np, M_ext_np.T, atol=atol, rtol=rtol):
            if force_sym:
                M_ext_np = 0.5 * (M_ext_np + M_ext_np.T)
            else:
                raise ValueError(
                    "Входная матрица не симметрична; "
                    "передайте force_sym=True или исправьте данные."
                )

        # ────────────────────────── 2. Перестановка → canonical ───────────
        # формируем вектор perm: *internal_index* → *external_index*
        if layout is MatrixLayout.TAIL:
            perm = list(range(K))  # уже canonical
        elif layout is MatrixLayout.SL:
            if K < 3:
                raise ValueError("SL-layout требует K≥3 (S, ≥1 резонатор, L)")
            if permutation is not None:
                raise ValueError("Для layout=SL параметр permutation не используется")
            order = K - 2
            perm = _make_perm(order, ports=2, layout=MatrixLayout.SL)
        elif layout is MatrixLayout.CUSTOM:
            if permutation is None:
                raise ValueError("CUSTOM-layout требует permutation=<seq[int]>")
            if len(permutation) != K or set(permutation) != set(range(K)):
                raise ValueError("Некорректная permutation — должен быть перестановочный вектор длиной K")
            perm = list(permutation)
        else:  # перестраховка на случай будущих расширений Enum
            raise ValueError(f"Неизвестный layout: {layout}")

        # матрица-перестановка  P  (rows = internal, cols = external)
        P = _np.eye(K, dtype=_np.float32)[perm]  # shape (K,K)

        # canonical   M_can = Pᵀ · M_ext · P
        M_can = P.T @ M_ext_np @ P

        # ────────────────────────── 3. Преобразуем в словарь Mij ───────────
        M_vals: Dict[str, float] = {}
        for i in range(K):
            for j in range(i, K):
                if abs(M_can[i, j]) < atol:
                    continue
                M_vals[f"M{i + 1}_{j + 1}"] = float(M_can[i, j])

        # ────────────────────────── 4. Если надо — вывести/догадаться Topology
        if topo is None:
            # ports → всё, что идёт после последнего ненулевого резонатора
            # здесь предполагаем ports=2 (самый частый кейс импорта SL)
            order = K - 2
            ports = 2
            links = []
            for k in M_vals:
                try:
                    i, j = _parse_m_key(k)
                except ValueError:
                    # диагональный элемент Mii — не ребро графа, пропускаем
                    continue
                if i != j:  # (i==j) уже отфильтровано, но оставим явную защиту
                    links.append((i, j))
            topo = Topology(order, ports, links=links, name="inferred")

        # ────────────────────────── 5. Финальный объект ────────────────────
        return cls(
            topo,
            M_vals,
            qu=qu,
            phase_a=phase_a,
            phase_b=phase_b,
        )

    # ------------------------------------------------------------------ визуализация
    def plot_matrix(
        self,
        *,
        layout: MatrixLayout = MatrixLayout.TAIL,
        log: bool | float = False,
        cmap: str = "coolwarm",
        annotate: bool = True,
        hide_zero: bool = True,
        figsize: Tuple[int, int] = (8, 6),
        backend: str = "numpy",
        device: str = "cpu",
    ) -> "matplotlib.figure.Figure":
        """Рисует тепловую карту коэффициентов матрицы.

        * **log** – ``True`` → `SymLogNorm(linthresh=0.02)`, либо число
          (linthresh) для настраиваемой k-зоны вокруг нуля.
        * **annotate** – печатать ли числовые значения (в *линейном* масштабе).
        * **hide_zero** – ячейки, отсутствующие в топологии, окрашивать в серый.
        * Возвращает объект ``Figure`` для дальнейшей кастомизации/сохранения.
        """
        if not _HAS_MPL:
            raise ImportError("Для plot_matrix необходим matplotlib.")

        M = _np.asarray(self.to_matrix(layout, backend=backend, device=device))
        order, ports = self.topo.order, self.topo.ports

        # ---------- маска «незначимых» элементов --------------------------
        mask = _np.zeros_like(M, dtype=bool)
        if hide_zero:
            tol = 1e-12
            mask |= _np.abs(M) < tol

        # ---------- нормировка цвета -------------------------------------
        vmin, vmax = _np.nanmin(M), _np.nanmax(M)
        if log:
            linthresh = 0.02 if log is True else float(log)
            norm = _mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        else:
            norm = _mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # ---------- подписи ------------------------------------------------
        if layout is MatrixLayout.SL and ports == 2:
            labels = ["S", *map(str, range(1, order + 1)), "L"]
        else:
            labels = [*map(str, range(1, order + 1)), *[f"P{i}" for i in range(1, ports + 1)]]

        fig, ax = _plt.subplots(figsize=figsize)
        im = ax.imshow(M, cmap=cmap, norm=norm)

        # маска серого для «нулевых»
        if hide_zero and mask.any():
            im.set_array(_np.ma.masked_where(mask, M))
            im.set_cmap(cmap)

        # подписи осей
        ax.set_xticks(_np.arange(M.shape[0]))
        ax.set_yticks(_np.arange(M.shape[0]))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # аннотации чисел (линейные!)
        if annotate and M.shape[0] <= 15:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if hide_zero and mask[i, j]:
                        continue
                    val = M[i, j]
                    if abs(val) < 1e-3:
                        text = f"{val:.2e}"
                    else:
                        text = f"{val:.2f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

        # цветовая шкала
        fig.colorbar(im, ax=ax, shrink=0.8)

        return fig


    # ---------------------------------------------------------------- repr
    def __repr__(self):
        ph_cnt = len(self.phase_a or []) + len(self.phase_b or [])
        qu_info = "None" if self.qu is None else (
            f"vec[{len(self.qu)}]" if isinstance(self.qu, (list, tuple, _np.ndarray))
            else f"{self.qu:g}")

        return (
            f"CouplingMatrix(order={self.topo.order}, "
            f"ports={self.topo.ports}, M={len(self.M_vals)}, "
            f"qu={qu_info}, "
            f"phases={ph_cnt})"
        )
