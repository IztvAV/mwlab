# mwlab/filters/cm.py
"""
cm.py — высокоуровневый фасад над ядром расчёта S-параметров
=============================================================

Содержимое модуля
-----------------
* **CouplingMatrix** — контейнер «Topology + значения M_ij + qu + фазы».
* **MatrixLayout** — перечисление раскладок портов во внешней матрице.
* **make_perm** — генерация перестановки строк/столбцов для макетов.
* **parse_m_key** — парсер ключей вида "M<i>_<j>" (1-based индексы).

Модуль не содержит тяжёлой математики — все вычисления выполняются в
`cm_core.solve_sparams`. Здесь же находятся методы импорта/экспорта полных
матриц связи (внутренний формат ↔ внешний), сериализация в словарь и обратно.

Особенности реализации
----------------------
* Только PyTorch: все матричные операции выполняются с использованием torch.
* По умолчанию dtype — float32 / complex64, device — auto("cuda"/"cpu").
* Проверка согласованности M_ключей с топологией выполняется мягко
  (`strict=False`), но с дополнительной валидацией диагоналей в самом классе.
* Методы `to_matrix` / `from_matrix` возвращают/принимают torch.Tensor;
  при необходимости пользователь может вызвать `.cpu().numpy()`.

"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Mapping, Sequence, Tuple, List, Optional

import torch

from .topologies import Topology, TopologyError
from .cm_core import (
    solve_sparams,
    CoreSpec,
    DEFAULT_DEVICE,
    DT_R,
)

# -----------------------------------------------------------------------------
#                            Парсер ключей M<i>_<j>
# -----------------------------------------------------------------------------

def parse_m_key(tag: str) -> Tuple[int, int]:
    """Разбирает строку ключа матрицы связи вида ``"M<i>_<j>"``.

    Возвращает кортеж 1-based индексов в порядке (min, max).
    Допускается i == j (диагональные элементы).

    Raises
    ------
    ValueError
        Если ключ не соответствует формату или индексы не положительные.
    """
    if not tag.startswith("M"):
        raise ValueError(f"Ожидался ключ 'M<i>_<j>', получено {tag!r}")
    try:
        i_str, j_str = tag[1:].split("_", 1)
        i, j = int(i_str), int(j_str)
    except Exception as exc:
        raise ValueError(f"Неверный ключ матрицы связи: {tag!r}") from exc
    if i <= 0 or j <= 0:
        raise ValueError(f"Индексы в {tag!r} должны быть положительными")
    return (i, j) if i <= j else (j, i)


# -----------------------------------------------------------------------------
#                        Раскладки внешних матриц (layout)
# -----------------------------------------------------------------------------

class MatrixLayout(Enum):
    """Где располагаются портовые строки/столбцы во *внешней* матрице.

    - **TAIL**  – канонический внутренний порядок MWLab: ``[R1 … Rn  P1 … Pp]``;
    - **SL**    – классический «Source–Load» (только p=2): ``[S  R1 … Rn  L]``;
    - **CUSTOM** – произвольная перестановка (задаётся пользователем).
    """

    TAIL = auto()
    SL = auto()
    CUSTOM = auto()


def make_perm(
    order: int,
    ports: int,
    layout: MatrixLayout,
    permutation: Optional[Sequence[int]] = None,
) -> List[int]:
    """Формирует перестановку **perm** длиной K = order + ports.

    Семантика: ``perm[i] = j`` означает, что **i‑я строка/столбец внешней матрицы**
    берётся из **j‑й строки/столбца канонической (TAIL) матрицы**.

    Преобразования:
    ``M_ext = P @ M_can @ P.T``, где ``P = eye(K)[perm]``.

    Parameters
    ----------
    order, ports : int
        Размерности.
    layout : MatrixLayout
        Желаемый макет.
    permutation : Sequence[int] | None
        Пользовательская перестановка для CUSTOM (0-based), должна быть
        перестановкой чисел ``0 … K‑1``.

    Returns
    -------
    list[int]
        Перестановка длиной K.
    """
    K = order + ports
    if layout is MatrixLayout.TAIL:
        return list(range(K))

    if layout is MatrixLayout.SL:
        if ports != 2:
            raise ValueError("layout 'SL' применим только к 2‑портовым матрицам")
        # external: [S,  R1 … Rn,  L]
        # internal: [R1 … Rn,  P1=S,  P2=L]
        return [order] + list(range(order)) + [order + 1]

    # CUSTOM
    if permutation is None:
        raise ValueError("CUSTOM layout требует permutation")
    perm = list(permutation)
    if len(perm) != K or sorted(perm) != list(range(K)):
        raise ValueError("permutation должна быть перестановкой 0…K-1")
    return perm


# -----------------------------------------------------------------------------
#                              CouplingMatrix
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class CouplingMatrix:
    """Контейнер «Topology + значения M_ij + qu + фазы».

    Параметры
    ----------
    topo : Topology
        Топология (граф связей).
    mvals : Mapping[str, float]
        Значения ненулевых коэффициентов расширенной матрицы связи ``M``.
        Ожидаются ключи вида "M<i>_<j>" (1-based, i ≤ j). Диагональные
        элементы (i==j) допустимы и не проверяются топологией.
    qu : float | Sequence[float] | None
        Приведённые добротности резонаторов (скаляр или вектор длиной order).
    phase_a / phase_b : float | Sequence[float] | Mapping[int,float] | None
        Коэффициенты фазовых линий. Могут задаваться:
          * скаляром — одинаково для всех портов;
          * последовательностью длиной ports;
          * словарём {port_index(1-based): value}.

    Примечание
    ----------
    Класс не хранит torch.Tensor — только питоновские структуры. Преобразование
    выполняется в методе :meth:`tensor_M` / :meth:`sparams`.
    """

    topo: Topology
    mvals: Mapping[str, float]
    qu: float | Sequence[float] | None = None
    phase_a: float | Sequence[float] | Mapping[int, float] | None = None
    phase_b: float | Sequence[float] | Mapping[int, float] | None = None

    # ------------------------------------------------------------------ init
    def __post_init__(self):
        # 1) Проверяем соответствие связей топологии (строгая полнота не требуется)
        self.topo.validate_mvals(self.mvals, strict=False)

        # 2) Дополнительная валидация ключей (индексы в допустимых пределах)
        K = self.topo.size
        for k in self.mvals:
            i, j = parse_m_key(k)  # может бросить ValueError
            if not (1 <= i <= K and 1 <= j <= K):
                raise ValueError(f"Ключ {k!r}: индексы выходят за предел 1…{K}")

        # 3) Проверка длины qu, phase_a, phase_b (если заданы последовательности)
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
            if isinstance(vec, Sequence) and not isinstance(vec, (str, bytes)):
                if len(vec) != self.topo.ports:
                    raise ValueError(
                        f"{label}: длина {len(vec)} ≠ ports({self.topo.ports})"
                    )

    # ---------------------------------------------------------------- helpers
    def _phase_to_list(self, ph, *, name: str) -> Optional[List[float]]:
        """Приводит phase_a/phase_b к списку длиной ports или None."""
        if ph is None:
            return None
        p = self.topo.ports
        if isinstance(ph, (int, float)):
            return [float(ph)] * p
        if isinstance(ph, Mapping):
            arr = [0.0] * p
            for key, val in ph.items():
                idx = int(key)
                if not (1 <= idx <= p):
                    raise ValueError(f"{name}: индекс {idx} вне диапазона 1…{p}")
                arr[idx - 1] = float(val)
            return arr
        # sequence
        if len(ph) != p:
            raise ValueError(f"{name}: длина {len(ph)} ≠ ports({p})")
        return [float(v) for v in ph]

    # ---------------------------------------------------------------- tensor_M
    def tensor_M(self, device: str | torch.device = DEFAULT_DEVICE) -> torch.Tensor:
        """Строит симметричную матрицу ``M_real`` формы (K,K) (torch.float32).

        Порты и резонаторы нумеруются 1-based (как в mvals), поэтому индексы
        уменьшаются на 1 при заполнении.
        """
        device = torch.device(device)
        K = self.topo.size
        M = torch.zeros((K, K), dtype=DT_R, device=device)

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for key, val in self.mvals.items():
            i, j = parse_m_key(key)
            rows.append(i - 1)
            cols.append(j - 1)
            vals.append(float(val))

        if rows:
            idx_r = torch.tensor(rows, dtype=torch.long, device=device)
            idx_c = torch.tensor(cols, dtype=torch.long, device=device)
            v_t = torch.tensor(vals, dtype=DT_R, device=device)
            M[idx_r, idx_c] = v_t
            M[idx_c, idx_r] = v_t
        return M

    # ---------------------------------------------------------------- sparams (Ω-шкала)
    def sparams(
        self,
        omega,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        method: str = "auto",
        fix_sign: bool = False,
    ) -> torch.Tensor:
        """Комплексная матрица S(Ω) через ядро :func:`solve_sparams`.

        Параметр `omega` — нормированная частота (не Гц!). Для работы в шкале
        частоты f используйте классы `Device`/`Filter`.
        """
        spec = CoreSpec(self.topo.order, self.topo.ports, method, fix_sign)
        return solve_sparams(
            spec,
            self.tensor_M(device=device),
            omega,
            qu=self.qu,
            phase_a=self._phase_to_list(self.phase_a, name="phase_a"),
            phase_b=self._phase_to_list(self.phase_b, name="phase_b"),
            device=device,
        )

    # ---------------------------------------------------------------- to_matrix
    def to_matrix(
        self,
        layout: MatrixLayout = MatrixLayout.TAIL,
        *,
        permutation: Optional[Sequence[int]] = None,
        device: str | torch.device = DEFAULT_DEVICE,
    ) -> torch.Tensor:
        """Возвращает *реальную* квадратную матрицу связи нужного макета.

        Параметры
        ----------
        layout : MatrixLayout
            Макет внешней матрицы (TAIL/SL/CUSTOM).
        permutation : Sequence[int] | None
            Перестановка (0-based) для CUSTOM.
        device : torch.device | str
            Устройство, на которое будет создана матрица.
        """
        order, ports = self.topo.order, self.topo.ports
        device = torch.device(device)

        M_can = self.tensor_M(device=device)
        perm = make_perm(order, ports, layout, permutation)
        P = torch.eye(order + ports, dtype=DT_R, device=device)[perm]
        return P @ M_can @ P.t()

    # ---------------------------------------------------------------- from_matrix
    @classmethod
    def from_matrix(
        cls,
        M_ext,  # (K,K)
        *,
        topo: Optional[Topology] = None,
        layout: MatrixLayout = MatrixLayout.TAIL,
        permutation: Optional[Sequence[int]] = None,
        force_sym: bool = True,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        # сразу прокидываем параметры в конструктор
        qu=None,
        phase_a=None,
        phase_b=None,
        device: str | torch.device = DEFAULT_DEVICE,
    ) -> "CouplingMatrix":
        """Создаёт объект из полной расширенной матрицы связи ``M_ext``.

        *Принимает torch.Tensor/array-like; внутри конвертируется в torch.float32.*

        Если *topo* не указан, порядок и количество портов можно попытаться
        восстановить. Здесь, для простоты, предполагаем, что при отсутствии topo
        ``layout=SL`` и ``ports=2`` (наиболее распространённый случай). В случае
        других макетов лучше явно передать Topology.
        """
        device = torch.device(device)
        M_ext_t = torch.as_tensor(M_ext, dtype=DT_R, device=device)
        if M_ext_t.ndim != 2 or M_ext_t.shape[0] != M_ext_t.shape[1]:
            raise ValueError("M_ext должна быть квадратной матрицей K×K")
        K = M_ext_t.shape[0]

        # Симметричность
        if not torch.allclose(M_ext_t, M_ext_t.t(), atol=atol, rtol=rtol):
            if force_sym:
                M_ext_t = 0.5 * (M_ext_t + M_ext_t.t())
            else:
                raise ValueError("Входная матрица не симметрична")

        # Перестановка в канонический формат (TAIL)
        if layout is MatrixLayout.TAIL:
            perm = list(range(K))
        elif layout is MatrixLayout.SL:
            if permutation is not None:
                raise ValueError("permutation не используется при layout=SL")
            order_guess = K - 2
            perm = make_perm(order_guess, 2, MatrixLayout.SL)
        elif layout is MatrixLayout.CUSTOM:
            if permutation is None:
                raise ValueError("CUSTOM layout требует permutation")
            if len(permutation) != K or sorted(permutation) != list(range(K)):
                raise ValueError("Некорректная permutation")
            perm = list(permutation)
        else:  # защита от будущих Enum-ов
            raise ValueError(f"Неизвестный layout: {layout}")

        P = torch.eye(K, dtype=DT_R, device=device)[perm]
        M_can = P.t() @ M_ext_t @ P  # canonical (TAIL)

        # Преобразуем в словарь M_ij (верхний треугольник включая диагональ)
        mvals: Dict[str, float] = {}
        for i in range(K):
            for j in range(i, K):
                val = float(M_can[i, j].item())
                if abs(val) < atol:
                    continue
                mvals[f"M{i + 1}_{j + 1}"] = val

        # Топология (если не задана)
        if topo is None:
            if layout is MatrixLayout.SL:
                # предположим 2 порта
                order = K - 2
                ports = 2
            else:
                # минималистичная эвристика: портовые индексы идут после резонаторов
                # Попробуем взять ports=2 по умолчанию
                ports = 2
                order = K - ports
                if order <= 0:
                    raise ValueError("Не удалось определить order/ports, задайте topo явно")

            # links = все ненулевые элементы верхнего треугольника без диагонали
            links = []
            for k in mvals:
                i, j = parse_m_key(k)
                if i != j:
                    links.append((i, j))
            topo = Topology(order=order, ports=ports, links=links, name="inferred")

        return cls(topo, mvals, qu=qu, phase_a=phase_a, phase_b=phase_b)

    # ------------------------------------------------------------------ dict I/O
    def to_dict(self) -> Dict[str, float | int | str]:
        """Сериализация в плоский словарь (JSON‑дружелюбный).

        Поля: order, ports, topology, все M… и параметры qu/phase…
        """
        out: Dict[str, float | int | str] = {
            "order": self.topo.order,
            "ports": self.topo.ports,
            "topology": self.topo.name or "",
        }
        out.update({k: float(v) for k, v in self.mvals.items()})

        # qu
        if self.qu is not None:
            if isinstance(self.qu, (list, tuple)):
                for idx, q in enumerate(self.qu, 1):
                    out[f"qu_{idx}"] = float(q)
            else:
                out["qu"] = float(self.qu)

        # phases
        def dump_phase(prefix: str, ph):
            if ph is None:
                return
            if isinstance(ph, Mapping):
                for p, v in ph.items():
                    out[f"{prefix}{int(p)}"] = float(v)
            elif isinstance(ph, (list, tuple)):
                for idx, v in enumerate(ph, 1):
                    out[f"{prefix}{idx}"] = float(v)
            else:  # scalar
                out[prefix] = float(ph)

        dump_phase("phase_a", self.phase_a)
        dump_phase("phase_b", self.phase_b)

        return out

    @classmethod
    def from_dict(cls, topo: Optional[Topology], d: Mapping[str, float | int | str]) -> "CouplingMatrix":
        """Восстановление объекта из plain-словаря (обратное действие to_dict).

        Если topo == None, топология будет построена по mvals (предполагается ports=2).
        """
        # --- 1. M-values ---
        mvals = {k: float(v) for k, v in d.items() if isinstance(k, str) and k.startswith("M")}
        if not mvals:
            raise ValueError("from_dict: нет коэффициентов 'M…'")

        # --- 2. qu ---
        if "qu" in d:
            qu = float(d["qu"])
        else:
            q_keys = sorted((k for k in d if isinstance(k, str) and k.startswith("qu_")),
                            key=lambda s: int(s.split("_")[1]))
            qu = [float(d[k]) for k in q_keys] if q_keys else None

        # --- 3. phases ---
        phase_a = {int(k[7:]): float(v) for k, v in d.items() if isinstance(k, str) and k.startswith("phase_a")} or None
        phase_b = {int(k[7:]): float(v) for k, v in d.items() if isinstance(k, str) and k.startswith("phase_b")} or None

        # --- 4. topo ---
        if topo is None:
            order_in = int(d.get("order", 0))
            ports_in = int(d.get("ports", 0)) or 2
            if order_in and ports_in:
                order, ports = order_in, ports_in
            else:
                # простая эвристика
                max_idx = 0
                for key in mvals:
                    i, j = parse_m_key(key)
                    max_idx = max(max_idx, i, j)
                ports = ports_in or 2
                order = order_in or (max_idx - ports)
                if order <= 0:
                    raise ValueError("Невозможно определить order/ports из словаря — передайте topo")
            links = []
            for k in mvals:
                i, j = parse_m_key(k)
                if i != j:
                    links.append((i, j))
            topo = Topology(order, ports, links=links, name="inferred")

        return cls(topo, mvals, qu=qu, phase_a=phase_a, phase_b=phase_b)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:
        ph_cnt = (len(self._phase_to_list(self.phase_a, name="phase_a") or []) +
                  len(self._phase_to_list(self.phase_b, name="phase_b") or []))
        qu_info = "None" if self.qu is None else (
            f"vec[{len(self.qu)}]" if isinstance(self.qu, (list, tuple)) else f"{float(self.qu):g}"
        )
        return (f"CouplingMatrix(order={self.topo.order}, ports={self.topo.ports}, "
                f"M={len(self.mvals)}, qu={qu_info}, phases={ph_cnt})")


__all__ = [
    "CouplingMatrix",
    "MatrixLayout",
    "make_perm",
    "parse_m_key",
]

