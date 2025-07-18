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
   • опции: вектор/скаляр добротностей **Q**, фазовые сдвиги линий
     *(phase_a, phase_b)*;
   • backend: **NumPy** или **PyTorch** (CPU / CUDA);
   • выход: комплексная матрица ``S (…,F,P,P)`` с dtype **complex64**.

2. **`CouplingMatrix`** – контейнер *(Topology + Mᵢⱼ + Q + phase)*
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

cm = CouplingMatrix(topo, M_vals, Q=[7000]*4)

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
cm_back = CouplingMatrix.from_matrix(M_sl, topo=topo, layout="SL", Q=cm.Q)
```
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple, List
from enum import Enum, auto

try:                                        # matplotlib не обязателен для core
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mcolors
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
    Разбирает строку ``"M{i}_{j}"`` → (i,j).  Бросает ValueError, если
    формат неверен или i==j.
    """
    if not tag.startswith("M"):
        raise ValueError(f"ожидался ключ 'M<i>_<j>', получено {tag!r}")
    try:
        i_str, j_str = tag[1:].split("_", 1)
        i, j = int(i_str), int(j_str)
    except Exception as exc:
        raise ValueError(f"неверный ключ матрицы связи: {tag!r}") from exc
    if i == j:
        raise ValueError(f"диагональный коэффициент должен называться 'M{i}{i}', "
                         f"а не {tag!r}")
    return (i, j) if i < j else (j, i)     # всегда (low, high)

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
def _xp_diag(xp, vec):
    """`diag(vec)` → (K,K) для NumPy и batched `torch.diag_embed`."""
    return xp.diag(vec) if xp.__name__ == "numpy" else xp.diag_embed(vec)


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
#                         helpers: фазовые коэффициенты
# ─────────────────────────────────────────────────────────────────────────────
def _phase_vec(val, ports: int, xp, *, dtype, **kw):
    """
    Приводит вход `val` к вектору длиной `ports`.

    * `None` → нуль-вектор;
    * `dict {порт: значение}` (1-base);
    * последовательность / тензор длиной `ports`.
    """
    out = xp.zeros(ports, dtype=dtype, **kw)
    if val is None:
        return out

    if isinstance(val, Mapping):               # словарь {p: value}
        for p, v in val.items():
            if not 1 <= p <= ports:
                raise ValueError(f"phase-coef: порт {p} вне диапазона 1…{ports}")
            out[p - 1] = float(v)
        return out

    if xp.__name__ == "torch" and xp.is_tensor(val):  # тензор (GPU/CPU)
        arr = val.to(dtype).reshape(-1)
    elif hasattr(xp, "as_tensor"):
        arr = xp.as_tensor(val, dtype=dtype, **kw).reshape(-1)
    else:
        arr = xp.asarray(val, dtype=dtype, **kw).reshape(-1)

    if arr.shape[0] != ports:
        raise ValueError("длина phase-вектора ≠ ports")
    return arr


def _phase_diag(xp, omega, a_vec, b_vec, dtype):
    """Диагональная матрица фаз: ``D = diag{e^{-j (a·ω + b)}}``."""
    theta = omega[..., None] * a_vec + b_vec       # → (..., F, P)
    if xp.__name__ == "torch":
        return xp.exp(-1j * theta).to(dtype)
    return _np.exp(-1j * _np.asarray(theta)).astype(dtype)

# ─────────────────────────────────────────────────────────────────────────────
#       helper: добавление потерь −j/(2 Q) на диагональ резонаторных узлов
# ─────────────────────────────────────────────────────────────────────────────
def _build_M_complex(topo: Topology, M_real, Q, xp, cplx, **kw):
    """
    Формирует комплексную матрицу ``M̃ = M_real - j / (2 Q)``.
    Поддерживает скаляр/вектор `Q`, numpy.ndarray и torch.Tensor.
    """
    # базовая матрица
    if xp.__name__ == "torch":
        M = xp.as_tensor(M_real, dtype=cplx, **kw)
    else:
        M = xp.asarray(M_real, dtype=cplx, **kw)

    K = topo.size
    if M.shape[-2:] != (K, K):
        raise ValueError(
            f"M_real: ожидалась матрица {K}×{K}, получено {M.shape[-2]}×{M.shape[-1]}"
        )
    if Q is None:
        return M

    order = topo.order
    idx = xp.arange(order, **kw)
    dtype_f64 = xp.float64 if xp.__name__ == "torch" else _np.float64

    is_tensor = xp.__name__ == "torch" and xp.is_tensor(Q) and Q.ndim == 1
    if isinstance(Q, (list, tuple)) or (isinstance(Q, _np.ndarray) and Q.ndim == 1) or is_tensor:
        Q_arr = xp.as_tensor(Q, dtype=dtype_f64, **kw) if is_tensor else xp.asarray(Q, dtype=dtype_f64, **kw)
        if Q_arr.shape[-1] != order:
            raise ValueError("Q: неверная длина")
        alpha = 1.0 / (2.0 * Q_arr)
    else:
        alpha = xp.full((order,), 1.0 / (2.0 * float(Q)), dtype=dtype_f64, **kw)

    alpha_j = (alpha * 1j).to(cplx) if xp.__name__ == "torch" else (alpha * 1j).astype(cplx)
    alpha_j = xp.broadcast_to(alpha_j, M.shape[:-2] + (order,))

    M[..., idx, idx] -= alpha_j
    return M

# ─────────────────────────────────────────────────────────────────────────────
#                              LOW-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────
def cm_sparams(
    topo: Topology,
    M_real,
    omega,
    *,
    Q=None,
    phase_a=None,
    phase_b=None,
    backend: str = "numpy",
    device: str = "cpu",
    method: str = "auto",    # 'auto' | 'inv' | 'solve'
    fix_sign: bool = False,
):
    """
    Рассчитывает комплексную S-матрицу размером ``(..., F, P, P)``.

    Параметры
    ---------
    topo     : :class:`mwlab.filters.topologies.Topology`
    M_real   : ndarray | torch.Tensor, вещественная матрица связи ``(...,K,K)``
    omega    : ndarray | torch.Tensor, нормированная частота ``(...,F)``
    Q        : None | скаляр | вектор длиной *order* | torch.Tensor
    phase_a, phase_b : коэффициенты фазовых линий (см. докстринг `_phase_vec`)
    backend  : ``'numpy'`` | ``'torch'``
    device   : строка устройства для Torch (``'cpu'`` | ``'cuda:0'`` …)
    method   : ``'inv'`` — прямое обращение,
               ``'solve'`` — решение `AX = I_pp`,
               ``'auto'`` — *solve* при `ports ≤ 4`, иначе *inv*.
    fix_sign : bool, default **False**
               Для 2‑портовых фильтров `True` инвертирует знак S₁₂ / S₂₁
               IEEE‑конвенция — `False`.
    """
    xp, cplx, kw = _lib(backend, device=device)

    order, ports, K = topo.order, topo.ports, topo.size

    # входы → backend-типы --------------------------------------------------
    omega = xp.as_tensor(omega, dtype=xp.float32, **kw) if hasattr(xp, "as_tensor") else xp.asarray(omega, dtype=_np.float32, **kw)
    M     = _build_M_complex(topo, M_real, Q, xp, cplx, **kw)

    # неизменные U, R, I_p -------------------------------------------------
    U, R, I_ports = _get_cached_mats(order, ports, xp, dtype=cplx, **kw)
    while U.ndim < M.ndim:                        # добавляем batch-оси
        U, R = (t[None, ...] for t in (U, R))

    j = xp.asarray(1j, dtype=cplx, **kw)
    w = omega[..., None, None]                    # (...,F,1,1)
    A = R + j * w * U - j * M                    # (...,F,K,K)

    # выбор алгоритма solve / inv -----------------------------------------
    use_solve = (
        method == "solve"
        or (method == "auto" and ports <= 4)
    )

    if use_solve:
        I_cols = _xp_eye(xp, K, dtype=cplx, **kw)[..., -ports:]        # (K,P)
        A2 = A.reshape((-1, K, K))
        B2 = xp.broadcast_to(I_cols, A2.shape[:-2] + I_cols.shape)
        B2 = B2.clone() if xp.__name__ == "torch" else B2.copy()
        X  = xp.linalg.solve(A2, B2).reshape(A.shape[:-2] + (K, ports))
        A_pp = X[..., -ports:, :]
    else:
        A_pp = xp.linalg.inv(A)[..., -ports:, -ports:]

    # базовая S ------------------------------------------------------------
    S = I_ports.reshape((1,) * (A_pp.ndim - 2) + I_ports.shape) - 2.0 * A_pp
    if fix_sign and ports == 2:
        S = S.clone() if xp.__name__ == "torch" else S.copy()
        S[..., 0, 1] *= -1.0
        S[..., 1, 0] *= -1.0

    # фазовые линии --------------------------------------------------------
    if phase_a is not None or phase_b is not None:
        a_vec = _phase_vec(phase_a, ports, xp, dtype=xp.float32, **kw)
        b_vec = _phase_vec(phase_b, ports, xp, dtype=xp.float32, **kw)
        D = _phase_diag(xp, omega, a_vec, b_vec, cplx)
        S = D[..., :, None] * S * D[..., None, :]

    return S.astype(cplx) if hasattr(S, "astype") else S.to(cplx)


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

def _make_perm(order: int, ports: int, layout: MatrixLayout,
               permutation: Sequence[int] | None = None):
    """Возвращает список индексов *perm* такой, что
    ``M_canon = P @ M_ext @ P.T``, где ``P = eye()[perm]``.
    """
    if layout is MatrixLayout.TAIL:
        return list(range(order + ports))

    if layout is MatrixLayout.SL:
        if ports != 2:
            raise ValueError("layout 'SL' применим только к 2-портовым устройствам")
        # внешний порядок: S, R1…Rn, L
        # внутренний      : R1…Rn,  S,     L
        perm = [*range(1, order + 1), 0, order + 1]
        return perm

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
    """Контейнер «Topology + Mij + Q + фазовые линии»."""
    topo: Topology
    M_vals: Mapping[str, float]
    Q: float | Sequence[float] | _np.ndarray | "torch.Tensor" | None = None
    phase_a: Mapping[int, float] | Sequence[float] | None = None
    phase_b: Mapping[int, float] | Sequence[float] | None = None

    # ------------------------------------------------------------------ init
    def __post_init__(self):
        self.topo.validate_mvals(self.M_vals, strict=False)
        if self.Q is not None and not isinstance(self.Q, (int, float)):
            try:
                qlen = len(self.Q)  # type: ignore
            except TypeError:
                qlen = None
            if qlen is not None and qlen != self.topo.order:
                raise ValueError(
                    f"Q: ожидается {self.topo.order} значений, получено {qlen}"
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
            Q=self.Q,
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
        `order`, `ports`, `topology`, все `M…`, `Q…`, `phase_a…`, `phase_b…`.
        """
        out: Dict[str, float] = {
            "order": self.topo.order,
            "ports": self.topo.ports,
            "topology": self.topo.name or "",
            **{k: float(v) for k, v in self.M_vals.items()},
        }

        # Q
        if self.Q is not None:
            if isinstance(
                self.Q, (list, tuple, _np.ndarray)
            ) or (hasattr(_np, "ndarray") and isinstance(self.Q, _np.ndarray)):
                for idx, q in enumerate(self.Q, 1):
                    out[f"Q_{idx}"] = float(q)
            else:
                out["Q"] = float(self.Q)

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

        # ------------------------------- 2. Q-параметры -----------------------
        if "Q" in d:
            Q = float(d["Q"])
        else:
            q_keys = sorted((k for k in d if k.startswith("Q_")),
                            key=lambda s: int(s.split("_")[1]))
            Q = [float(d[k]) for k in q_keys] if q_keys else None

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
            order = max(order_in_dict, max(i for i in indices
                                           if i <= max_idx // 2))
            # «ports» берём так, чтобы хватило до max-index
            ports = max(ports_in_dict, max_idx - order)

            # формируем links: только верхний треугольник, убираем дубли
            links = sorted(set(pairs))

            topo = Topology(order=order,
                            ports=ports,
                            links=links,
                            name="inferred")

        return cls(topo,
                   M_vals,
                   Q=Q,
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
            Q=None,
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

        Q, phase_a, phase_b
            Добротности резонаторов и фазовые коэффициенты линий – передаются
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
            # internal:   0…n-1 (R) , n (P1=S) , n+1 (P2=L)
            # external:   0=S , 1…n (R) , n+1=L
            perm = [i + 1 for i in range(order)] + [0, K - 1]
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
                if abs(M_can[i, j]) < 1e-12:  # нуль → пропускаем
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
            Q=Q,
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
        figsize: Tuple[int, int] = (5, 5),
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
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("узлы j")
        ax.set_ylabel("узлы i")

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
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Magnitude of M_ij")
        fig.tight_layout()
        return fig


    # ---------------------------------------------------------------- repr
    def __repr__(self):
        ph_cnt = len(self.phase_a or []) + len(self.phase_b or [])
        return (
            f"CouplingMatrix(order={self.topo.order}, "
            f"ports={self.topo.ports}, M={len(self.M_vals)}, "
            f"phases={ph_cnt})"
        )
