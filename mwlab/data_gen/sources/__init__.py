# mwlab/data_gen/sources/__init__.py
"""
Подпакет **mwlab.data_gen.sources**
==================================
Содержит готовые реализации :class:`mwlab.data_gen.base.ParamSource`:

* :class:`~mwlab.data_gen.sources.list.ListSource`          – in‑memory список;
* :class:`~mwlab.data_gen.sources.csv.CsvSource`            – CSV/TSV‑таблица;
* :class:`~mwlab.data_gen.sources.design_space.DesignSpaceSource` – DOE на лету;
* :class:`~mwlab.data_gen.sources.folder.FolderSource`      – каталог *.json*/yaml.

При необходимости пользователь может легко добавить собственный Source,
просто реализовав абстрактный контракт и импортировав класс здесь для
автодоступности через `mwlab.data_gen.sources`.
"""
from typing import TYPE_CHECKING

__all__ = ["ListSource", "CsvSource", "DesignSpaceSource", "FolderSource"]

def __getattr__(name):
    if name == "ListSource":
        from .list import ListSource; return ListSource
    if name == "CsvSource":
        from .csv import CsvSource; return CsvSource
    if name == "DesignSpaceSource":
        from .design_space import DesignSpaceSource; return DesignSpaceSource
    if name == "FolderSource":
        from .folder import FolderSource; return FolderSource
    raise AttributeError(name)

if TYPE_CHECKING:  # только для подсветки/автодополнения
    from .list import ListSource as ListSource
    from .csv import CsvSource as CsvSource
    from .design_space import DesignSpaceSource as DesignSpaceSource
    from .folder import FolderSource as FolderSource

