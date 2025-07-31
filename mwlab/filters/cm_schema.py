# mwlab/filters/cm_schema.py
# -*- coding: utf-8 -*-
"""
cm_schema.py — схема параметров расширенной матрицы связи (Coupling Matrix)
============================================================================

Назначение
----------
Единообразно описать и реализовать отображение:

    { "M1_1", "M1_2", ..., "qu", "qu_1", ..., "phase_a1", ... }  ⇄  вектор параметров (torch.Tensor)

а также быстро формировать батч матриц ``M_real`` и вспомогательных векторов
``qu``, ``phase_a``, ``phase_b`` для передачи в ядро ``solve_sparams`` (см. cm_core).

Ключевые идеи
-------------
* **ParamSchema** — единственный «источник истины» по порядку и составу параметров.
* Фиксированный порядок ключей:
  1. Диагональ резонаторов: ``M1_1 … M_order_order``;
  2. Внедиагональные элементы из ``Topology.links`` (i<j), отсортированные;
  3. Блок ``qu`` (по опции: "none" | "scalar" | "vec");
  4. Блоки фаз: ``phase_a1…`` и/или ``phase_b1…`` (векторы длиной ports, если включены).
* В `pack()` допускаются *скалярные* записи фазовых коэффициентов
  (`"phase_a"` / `"phase_b"`) — они автоматически разворачиваются на все порты.
  Сами блоки в схеме, однако, всегда векторные.
* Сборка матрицы ``M`` выполняется низкоуровневой функцией ``build_M`` (torch),
  расположенной в ``cm_core``; здесь есть fallback на случай отсутствия.

Основные методы
---------------
* ``ParamSchema.from_topology(topo, include_qu=..., include_phase=...)`` — конструктор.
* ``pack(params)`` / ``unpack(vec)`` — dict ⇄ vector.
* ``assemble(vec)`` — батч-сборка ``(M_real, qu, phase_a, phase_b)``.
* ``masks()`` — булевы маски по блокам для частичного обучения/регуляризаций.

Зависимости
-----------
* torch — обязательна (векторизация и батчи).
* mwlab.filters.topologies.Topology
* (опционально) mwlab.filters.cm_core.build_M — если отсутствует, используется fallback.

Автор: MWLab, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple, Sequence, Literal, Optional

import torch

try:  # Python 3.8/3.9 совместимость
    from typing import Literal as _Literal  # noqa
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

from .topologies import Topology
from .cm_core import DT_R  # типы ядра
from .cm_core import build_M


# ────────────────────────────────────────────────────────────────────────────
#                                 ParamSchema
# ────────────────────────────────────────────────────────────────────────────
IncludeQU = Literal["none", "scalar", "vec"]


@dataclass(frozen=True)
class ParamSchema:
    """
    Схема параметров расширенной матрицы связи для заданной топологии.

    Атрибуты (основные)
    -------------------
    topo : Topology
        Топология (граф связей).
    include_qu : {"none","scalar","vec"}
        Как включаем добротности:
            * "none"   — не включаем qu в вектор
            * "scalar" — один скаляр qu, который будет размножен на все резонаторы
            * "vec"    — qu_1, ..., qu_order
    include_phase : tuple{"a","b"}
        Какие фазовые множители включать (всегда вектор длиной ports):
            * ()            — ничего
            * ("a",)        — только phase_a
            * ("a","b")     — phase_a и phase_b
            * ("b",)        — (редко, но допустимо)
    keys : tuple[str,...]
        Ключи параметров в ПОЛНОМ порядке, соответствующие позициям в векторе.
    slices : Dict[str, slice]
        Срезы по блокам:
            "M_diag", "M_off", "M", "qu", "phase_a", "phase_b"
        Отсутствующие блоки дают slice(0,0).
    m_rows / m_cols : tuple[int,...]
        Индексы верхнего треугольника (0-based) для блока M (длина = L_M = L_diag + L_off).
    mask_default : torch.BoolTensor
        Маска всех параметров (все True).
    """

    topo: Topology
    include_qu: IncludeQU = "vec"
    include_phase: Tuple[Literal["a", "b"], ...] = field(default_factory=lambda: ("a",))
    keys: Tuple[str, ...] = field(init=False, repr=False)
    slices: Dict[str, slice] = field(init=False, repr=False)
    m_rows: Tuple[int, ...] = field(init=False, repr=False)
    m_cols: Tuple[int, ...] = field(init=False, repr=False)
    mask_default: Tuple[bool, ...] = field(init=False, repr=False)
    K: int = field(init=False, repr=False)

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_topology(cls,
                      topo: Topology,
                      *,
                      include_qu: IncludeQU = "vec",
                      include_phase: Tuple[Literal["a", "b"], ...] = ("a",)) -> "ParamSchema":
        """
        Создание схемы из топологии.

        Порядок параметров:
        1. Диагональ резонаторов: M1_1..M_{n}_{n}
        2. Внедиагональные связи из topo.links (только i<j), отсортированные
        3. qu (в зависимости от include_qu)
        4. phase_a / phase_b (по include_phase), всегда вектор длиной ports
        """
        order, ports = topo.order, topo.ports
        K = topo.size

        # --- 1) Диагонали резонаторов ---
        diag_keys = [f"M{i}_{i}" for i in range(1, order + 1)]
        diag_rows = [i - 1 for i in range(1, order + 1)]
        diag_cols = [i - 1 for i in range(1, order + 1)]

        # --- 2) Off-diag из topo.links (верхний треугольник уже гарантирован) ---
        off_links = sorted(topo.links)  # already 1-based i<j
        off_keys = [f"M{i}_{j}" for (i, j) in off_links]
        off_rows = [i - 1 for (i, j) in off_links]
        off_cols = [j - 1 for (i, j) in off_links]

        # --- 3) qu ---
        qu_keys: Sequence[str]
        if include_qu == "none":
            qu_keys = []
        elif include_qu == "scalar":
            qu_keys = ["qu"]
        elif include_qu == "vec":
            qu_keys = [f"qu_{i}" for i in range(1, order + 1)]
        else:
            raise ValueError("include_qu должен быть 'none' | 'scalar' | 'vec'")

        # --- 4) phases ---
        phase_a_keys: Sequence[str] = []
        phase_b_keys: Sequence[str] = []
        if "a" in include_phase:
            phase_a_keys = [f"phase_a{i}" for i in range(1, ports + 1)]
        if "b" in include_phase:
            phase_b_keys = [f"phase_b{i}" for i in range(1, ports + 1)]

        # --- конкатенация ключей ---
        keys = tuple(diag_keys + off_keys + list(qu_keys) + list(phase_a_keys) + list(phase_b_keys))

        # --- срезы ---
        idx = 0
        sl_diag = slice(idx, idx + len(diag_keys)); idx += len(diag_keys)
        sl_off  = slice(idx, idx + len(off_keys));  idx += len(off_keys)
        sl_M    = slice(sl_diag.start, sl_off.stop)  # полный блок M

        sl_qu = slice(idx, idx + len(qu_keys)); idx += len(qu_keys)

        sl_pa = slice(idx, idx + len(phase_a_keys)); idx += len(phase_a_keys)
        sl_pb = slice(idx, idx + len(phase_b_keys)); idx += len(phase_b_keys)

        slices = {
            "M_diag": sl_diag,
            "M_off":  sl_off,
            "M":      sl_M,
            "qu":     sl_qu,
            "phase_a": sl_pa,
            "phase_b": sl_pb,
        }

        # --- rows / cols ---
        m_rows = tuple(diag_rows + off_rows)
        m_cols = tuple(diag_cols + off_cols)

        mask_default = tuple(True for _ in keys)

        # frozen=True -> используем object.__setattr__
        obj = cls(topo=topo, include_qu=include_qu, include_phase=include_phase)
        object.__setattr__(obj, "keys", keys)
        object.__setattr__(obj, "slices", slices)
        object.__setattr__(obj, "m_rows", m_rows)
        object.__setattr__(obj, "m_cols", m_cols)
        object.__setattr__(obj, "mask_default", mask_default)
        object.__setattr__(obj, "K", K)

        return obj

    # ------------------------------------------------------------------ props
    @property
    def size(self) -> int:
        """Общая длина параметр-вектора."""
        return len(self.keys)

    @property
    def L_M(self) -> int:
        """Число параметров блока M (diag + off)."""
        return self.slices["M"].stop - self.slices["M"].start

    # ------------------------------------------------------------------ dict -> vector

    def pack(self,
             params: Mapping[str, float],
             *,
             device: Optional[torch.device | str] = None,
             dtype: torch.dtype = DT_R,
             strict: bool = True,
             default: float = 0.0) -> torch.Tensor:
        """
        Упаковка словаря параметров в вектор в каноническом порядке.

        params : Mapping[str,float]
            Ключи вида "M1_2", "qu_1", "phase_a2", ... .
            **Упрощение:** допускаются скаляры ``"phase_a"`` и/или ``"phase_b"`` —
            они автоматически разворачиваются на все порты, если соответствующие блоки
            включены в схему.

        strict : bool
            True  → требовать наличие ВСЕХ ключей схемы;
            False → пропущенные заполнять default.
        default : float
            Значение для пропущенных (при strict=False).

        Возвращает
        ----------
        torch.Tensor
            Вектор формы (L,) на указанном device.
        """
        dev = torch.device(device) if device is not None else None

        # Преобразуем к обычному словарю, если понадобится модификация
        mutable = params if isinstance(params, dict) else dict(params)

        # --- sugar для скалярных фаз ---
        P = self.topo.ports
        if "a" in self.include_phase and "phase_a" in mutable:
            val = float(mutable["phase_a"])
            for i in range(1, P + 1):
                mutable.setdefault(f"phase_a{i}", val)
        if "b" in self.include_phase and "phase_b" in mutable:
            val = float(mutable["phase_b"])
            for i in range(1, P + 1):
                mutable.setdefault(f"phase_b{i}", val)

        out = torch.empty(self.size, dtype=dtype, device=dev)
        missing = []
        for i, k in enumerate(self.keys):
            if k in mutable:
                out[i] = float(mutable[k])
            else:
                if strict:
                    missing.append(k)
                out[i] = default

        if strict and missing:
            raise KeyError(f"pack: отсутствуют ключи: {missing}")

        return out

    # ------------------------------------------------------------------ vector -> dict
    def unpack(self, vec: torch.Tensor) -> Dict[str, float]:
        """
        Обратное преобразование для удобства отладки/логов.

        vec : (..., L) поддерживается только 1D
        """
        if vec.ndim != 1 or vec.shape[0] != self.size:
            raise ValueError(f"unpack: ожидается вектор формы ({self.size},), получено {tuple(vec.shape)}")
        return {k: float(vec[i].item()) for i, k in enumerate(self.keys)}

    # ------------------------------------------------------------------ assemble
    def assemble(self,
                 vec: torch.Tensor,
                 *,
                 device: Optional[torch.device | str] = None) -> Tuple[torch.Tensor,
                                                                       Optional[torch.Tensor],
                                                                       Optional[torch.Tensor],
                                                                       Optional[torch.Tensor]]:
        """
        Превращает параметр-вектор в блоки, пригодные для solve_sparams.

        vec : (..., L)
        return:
            M_real  : (..., K, K) float32
            qu      : (..., order) or None
            phase_a : (..., ports) or None
            phase_b : (..., ports) or None
        """
        if vec.shape[-1] != self.size:
            raise ValueError(f"assemble: последний размер vec должен быть {self.size}, получено {vec.shape[-1]}")

        dev = torch.device(device) if device is not None else vec.device

        # --- разбиваем на блоки ---
        M_slice = self.slices["M"]
        m_vals = vec[..., M_slice]  # (..., L_M)

        rows_t = torch.as_tensor(self.m_rows, dtype=torch.long, device=dev)
        cols_t = torch.as_tensor(self.m_cols, dtype=torch.long, device=dev)

        # Если vec не на нужном device — перенесём m_vals
        if m_vals.device != dev:
            m_vals = m_vals.to(dev)

        # Собираем M
        M_real = build_M(rows_t, cols_t, m_vals, self.K)

        # --- qu ---
        qu_slice = self.slices["qu"]
        qu = None
        if qu_slice.stop > qu_slice.start:
            block = vec[..., qu_slice]
            if self.include_qu == "scalar":
                # (...,) -> (..., 1) -> repeat
                val = block.unsqueeze(-1)
                qu = val.repeat_interleave(self.topo.order, dim=-1)
            else:
                qu = block

            if qu.device != dev:
                qu = qu.to(dev)

        # --- phases ---
        pa_slice = self.slices["phase_a"]
        pb_slice = self.slices["phase_b"]
        phase_a = None
        phase_b = None

        if pa_slice.stop > pa_slice.start:
            phase_a = vec[..., pa_slice]
            if phase_a.device != dev:
                phase_a = phase_a.to(dev)

        if pb_slice.stop > pb_slice.start:
            phase_b = vec[..., pb_slice]
            if phase_b.device != dev:
                phase_b = phase_b.to(dev)

        return M_real, qu, phase_a, phase_b

    # ------------------------------------------------------------------ masks
    def masks(self,
              *,
              device: Optional[torch.device | str] = None) -> Dict[str, torch.Tensor]:
        """
        Возвращает булевы маски по блокам.

        keys:
            "all", "M", "M_diag", "M_off", "qu", "phase_a", "phase_b"
        """
        dev = torch.device(device) if device is not None else None

        def _mk(sl: slice) -> torch.Tensor:
            m = torch.zeros(self.size, dtype=torch.bool, device=dev)
            m[sl] = True
            return m

        masks = {
            "all": torch.ones(self.size, dtype=torch.bool, device=dev),
            "M_diag": _mk(self.slices["M_diag"]),
            "M_off": _mk(self.slices["M_off"]),
            "M": _mk(self.slices["M"]),
            "qu": _mk(self.slices["qu"]),
            "phase_a": _mk(self.slices["phase_a"]),
            "phase_b": _mk(self.slices["phase_b"]),
        }
        return masks

    # ------------------------------------------------------------------ helpers
    def index_of(self, key: str) -> int:
        """Возвращает позицию ключа в векторе или бросает ValueError."""
        try:
            return self.keys.index(key)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Ключ {key!r} отсутствует в схеме") from exc

    def block_slice(self, name: str) -> slice:
        """Срез блока по имени ('M_diag','M_off','M','qu','phase_a','phase_b')."""
        try:
            return self.slices[name]
        except KeyError as exc:  # pragma: no cover
            raise KeyError(f"Неизвестный блок {name!r}") from exc

    def __len__(self) -> int:  # удобный алиас
        return self.size

    def __repr__(self) -> str:
        qu_info = self.include_qu
        ph_info = ",".join(self.include_phase) if self.include_phase else "none"
        return (f"ParamSchema(size={self.size}, M={self.L_M}, qu='{qu_info}', "
                f"phase='{ph_info}', order={self.topo.order}, ports={self.topo.ports})")

