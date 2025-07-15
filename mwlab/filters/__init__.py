#mwlab/filters/__init__.py
"""
mwlab.filters
=============
Высокоуровневый API для аналитического расчёта **S-параметров**
СВЧ-фильтров, диплексеров и мультиплексеров на основе
реальной матрицы связи (coupling-matrix).

Основные уровни
---------------
* **Topology** – описание графа связей (узлы-резонаторы + порты);
* **CouplingMatrix** – численные значения *M*ᵢⱼ, добротности *Q*,
  фазовые коэффициенты линий;
* **Filter / Device** – конкретное устройство (LP/HP/BP/BR и др.),
  которое переводит нормированную частоту Ω в физическую *f*.

Быстрый старт
-------------
```python
import numpy as np
from mwlab.filters import get_topology, CouplingMatrix, Filter

# 1) топология «folded» на 4 резонатора
topo = get_topology("folded", order=4)

# 2) задаём ненулевые M и добротности
cm = CouplingMatrix(
    topo,
    M_vals={"M1_2": 1.05, "M2_3": 0.90, "M3_4": 1.05, "M1_4": 0.25},
    Q=7000,
)

# 3) создаём полосовой фильтр (f0=2 ГГц, BW=200 МГц)
flt = Filter.bp(cm, f0=2.0, bw=0.2, unit="GHz")

# 4) расчёт S-параметров
f = np.linspace(1.6e9, 2.4e9, 801)          # частотная сетка, Гц
S = flt.sparams(f)                          # shape = (801, 2, 2)
```
"""

# ── Topology registry ──────────────────────────────────────────────────
from .topologies import (
    Topology,
    get_topology,
    list_topologies,
    register_topology,
)

# ── Coupling-matrix layer ──────────────────────────────────────────────
from .cm import (
    cm_sparams,
    CouplingMatrix,
    MatrixLayout,
)

# ── Device layer (лёгкие классы) ───────────────────────────────────────
from .devices import (
    Device,
    Filter,
    # Multiplexer   # ← пока не экспортируем: класс черновой
)

__all__ = [
    # topologies
    "Topology",
    "get_topology",
    "list_topologies",
    "register_topology",
    # coupling matrix
    "cm_sparams",
    "CouplingMatrix",
    "MatrixLayout",
    # devices
    "Device",
    "Filter",
]

