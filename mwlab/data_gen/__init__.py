# mwlab/data_gen/__init__.py
"""Корневая точка входа под‑библиотеки **mwlab.data_gen**.

Здесь аккумулируются наиболее востребованные классы/функции для удобного
импорта пользователя:

```python
from mwlab.data_gen import (
    # инфраструктура
    ParamSource, Writer, DataGenerator, GenRunner, run_pipeline,
    # готовые источники
    ListSource, CsvSource, DesignSpaceSource, FolderSource,
    # стандартные Writer‑ы
    ListWriter, TouchstoneDirWriter, HDF5Writer, RAMWriter, TensorWriter,
    # синтетические генераторы на основе Coupling‑Matrix
    CMGenerator, DeviceCMGenerator,
)
```

Полная иерархия модулей остаётся доступной, поэтому при необходимости
можно производить «глубокий» импорт (`from mwlab.data_gen.sources.list import …`).
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
#                                базовый слой
# ────────────────────────────────────────────────────────────────────────────
from .base import (
    ParamSource,
    Writer,
    DataGenerator,
)

# ────────────────────────────────────────────────────────────────────────────
#                        высокоуровневый раннер/пайплайн
# ────────────────────────────────────────────────────────────────────────────
from .runner import run_pipeline, GenRunner

# ────────────────────────────────────────────────────────────────────────────
#                                   Sources
# ────────────────────────────────────────────────────────────────────────────
from .sources import (
    ListSource,
    CsvSource,
    DesignSpaceSource,
    FolderSource,
)

# ────────────────────────────────────────────────────────────────────────────
#                                   Writers
# ────────────────────────────────────────────────────────────────────────────
from .writers import (
    ListWriter,
    TouchstoneDirWriter,
    HDF5Writer,
    RAMWriter,
    TensorWriter,
)

# ────────────────────────────────────────────────────────────────────────────
#                   Coupling‑Matrix синтетические генераторы
# ────────────────────────────────────────────────────────────────────────────
from .cm_generators import CMGenerator, DeviceCMGenerator

# ────────────────────────────────────────────────────────────────────────────
#                                   __all__
# ────────────────────────────────────────────────────────────────────────────
__all__ = [
    # базовый слой
    "ParamSource",
    "Writer",
    "DataGenerator",
    # раннеры
    "run_pipeline",
    "GenRunner",
    # источники
    "ListSource",
    "CsvSource",
    "DesignSpaceSource",
    "FolderSource",
    # writer‑ы
    "ListWriter",
    "TouchstoneDirWriter",
    "HDF5Writer",
    "RAMWriter",
    "TensorWriter",
    # coupling‑matrix генераторы
    "CMGenerator",
    "DeviceCMGenerator",
]
