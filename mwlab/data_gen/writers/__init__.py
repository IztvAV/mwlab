# mwlab/data_gen/writers/__init__.py
"""
Подпакет **mwlab.data_gen.writers** – готовые приёмники (Writer‑ы)
==================================================================
Реэкспортирует следующие классы:

* :class:`~mwlab.data_gen.writers.list.ListWriter`
* :class:`~mwlab.data_gen.writers.touchstone_dir.TouchstoneDirWriter`
* :class:`~mwlab.data_gen.writers.hdf5.HDF5Writer`
* :class:`~mwlab.data_gen.writers.ram.RAMWriter`
* :class:`~mwlab.data_gen.writers.tensor.TensorWriter`
"""
def __getattr__(name):
    if name == "ListWriter":
        from .list import ListWriter; return ListWriter
    if name == "TouchstoneDirWriter":
        from .touchstone_dir import TouchstoneDirWriter; return TouchstoneDirWriter
    if name == "HDF5Writer":
        from .hdf5 import HDF5Writer; return HDF5Writer
    if name == "RAMWriter":
        from .ram import RAMWriter; return RAMWriter
    if name == "TensorWriter":
        from .tensor import TensorWriter; return TensorWriter
    raise AttributeError(name)
