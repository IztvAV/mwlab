#tests/test_hdf5_backend.py
"""
HDF5Backend: запись → чтение, контекстный менеджер, close().
"""

from __future__ import annotations

import pathlib
import numpy as np
import h5py

from mwlab.io.backends.hdf5_backend import HDF5Backend
from mwlab.io.backends.file_backend import FileBackend


def test_write_and_read(tmp_dir, sample_dir):
    """Записываем 10 записей в файл, затем читаем и сравниваем."""
    h5_path = pathlib.Path(tmp_dir) / "data.h5"

    file_backend = FileBackend(sample_dir)
    with HDF5Backend(h5_path, mode="w") as writer:
        for idx in range(10):
            writer.append(file_backend.read(idx))

    # файл должен быть закрыт
    assert not writer.h5.id.valid

    # ------- reopen на чтение
    reader = HDF5Backend(h5_path, mode="r")
    assert len(reader) == 10

    # сравним первую запись
    ts0_file = file_backend.read(0)
    ts0_h5 = reader.read(0)

    np.testing.assert_allclose(ts0_file.network.s, ts0_h5.network.s)
    np.testing.assert_allclose(ts0_file.network.f, ts0_h5.network.f)
    for k in ts0_file.params:
        assert k in ts0_h5.params
        if isinstance(ts0_file.params[k], float):
            assert np.isclose(ts0_file.params[k], ts0_h5.params[k])
        else:
            assert ts0_file.params[k] == ts0_h5.params[k]

    reader.close()

    # файл должен быть закрыт
    assert not reader.h5.id.valid
