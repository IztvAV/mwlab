"""
mwlab.filters
=============

Подпакет для расчёта S‑параметров СВЧ‑фильтров и мультиплексеров по
расширенной матрице связи (coupling matrix).

Структура
---------
- topologies.py  — графы связей (Topology) и реестр шаблонов.
- cm_core.py     — вычислительное ядро (Torch‑only) расчёта S(Ω).
- cm.py          — контейнер CouplingMatrix + преобразования макетов/сериализация.
- cm_io.py       — импорт/экспорт полных матриц (ASCII/JSON).
- devices.py     — высокоуровневые классы устройств (Filter, Device, …).

Основные точки входа
--------------------
- Topology / get_topology / register_topology — описание топологий.
- CouplingMatrix — контейнер параметров матрицы связи.
- solve_sparams  — прямой вызов ядра (sparams_core).
- Filter         — готовый 2‑портовый фильтр (LP/HP/BP/BR).
"""

from .topologies import (
    Topology,
    TopologyError,
    register_topology,
    get_topology,
    list_topologies,
)

from .cm import (
    CouplingMatrix,
    MatrixLayout,
    make_perm,
    parse_m_key,
)

from .cm_core import (
    solve_sparams,
    CoreSpec,
    CMError,
    DEFAULT_DEVICE,
)

from .cm_io import (
    write_matrix,
    read_matrix,
)

from .devices import (
    Device,
    Filter,
    Multiplexer,
)

__all__ = [
    # topologies
    "Topology",
    "TopologyError",
    "register_topology",
    "get_topology",
    "list_topologies",
    # coupling matrix
    "CouplingMatrix",
    "MatrixLayout",
    "make_perm",
    "parse_m_key",
    # core
    "solve_sparams",
    "CoreSpec",
    "CMError",
    "DEFAULT_DEVICE",
    # io
    "write_matrix",
    "read_matrix",
    # devices
    "Device",
    "Filter",
    "Multiplexer",
]

