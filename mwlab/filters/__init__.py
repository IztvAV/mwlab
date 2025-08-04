# mwlab/filters/__init__.py
# -*- coding: utf-8 -*-
"""
mwlab.filters
=============

Подпакет для работы с **S‑параметрами СВЧ‑фильтров и мультиплексеров**
на основе расширенной матрицы связи (*coupling matrix*).

Содержимое
----------

* **topologies.py**
    Неизменяемые графы связей (:class:`Topology`) + реестр шаблонов.

* **cm_core.py**
    Torch‑ядро прямого расчёта S(Ω) (:func:`solve_sparams`).

* **cm_schema.py**
    :class:`ParamSchema` — описание вектора параметров (M, qu, фазы)
    и утилиты pack / unpack / assemble.

* **cm.py**
    :class:`CouplingMatrix` — контейнер значений матрицы связи,
    импорты/экспорты, визуализация.

* **cm_io.py**
    ASCII / JSON‑файлы полной матрицы связи.

* **devices.py**
    Высокоуровневые классы :class:`Filter`, :class:`Device`.

* **cm_nn.py**
    :class:`CMLayer` — слой *torch.nn.Module* для аналитического
    расчёта S‑параметров в составе нейросети.

* **cm_gen.py**
    Утилиты генерации данных:
    `schema_to_space`, `space_to_schema`, :class:`CMDataset`.

Основные точки входа
--------------------
* **Функции/классы структуры**
    – :class:`Topology`, :func:`get_topology`,
      :class:`ParamSchema`, :class:`CouplingMatrix`.

* **Вычислительное ядро**
    – :func:`solve_sparams`, :class:`CoreSpec`.

* **Устройства**
    – :class:`Filter` (LP/HP/BP/BR).

* **Интеграция с PyTorch**
    – :class:`CMLayer`, :class:`CMDataset`.
"""

# ────────────────────────────────────────────────────────────────────────────
#                                  imports
# ────────────────────────────────────────────────────────────────────────────
from .topologies import (
    Topology,
    TopologyError,
    register_topology,
    get_topology,
    list_topologies,
)

from .cm_schema import (
    ParamSchema,
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
    clear_core_cache,
    core_cache_info,
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

# ────────────────────────────────────────────────────────────────────────────
#                                   __all__
# ────────────────────────────────────────────────────────────────────────────
__all__ = [
    # topologies
    "Topology",
    "TopologyError",
    "register_topology",
    "get_topology",
    "list_topologies",
    # schema
    "ParamSchema",
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
    "clear_core_cache",
    "core_cache_info",
    # io
    "write_matrix",
    "read_matrix",
    # devices
    "Device",
    "Filter",
    "Multiplexer",
]
