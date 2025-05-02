#tests/test_hdf5_backend.py
"""
tests/test_hdf5_backend.py
--------------------------
Проверяем корректность работы HDF5Backend:

* запись → чтение (данные, метаданные, параметры)
* контекст‑менеджер, close()
* опциональное обновление длины через refresh() при SWMR‑чтении
"""

from __future__ import annotations

import pathlib

import numpy as np
import h5py
import pytest

from mwlab.io.backends.hdf5_backend import HDF5Backend
from mwlab.io.backends.file_backend import FileBackend


# ---------------------------------------------------------------------------
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

    # S‑матрица и частоты
    np.testing.assert_allclose(ts0_file.network.s, ts0_h5.network.s)
    np.testing.assert_allclose(ts0_file.network.f, ts0_h5.network.f)

    # параметры пользователя
    for k in ts0_file.params:
        assert k in ts0_h5.params
        if isinstance(ts0_file.params[k], float):
            assert np.isclose(ts0_file.params[k], ts0_h5.params[k])
        else:
            assert ts0_file.params[k] == ts0_h5.params[k]

    # системные метаданные
    assert ts0_file.network.frequency.unit == ts0_h5.network.frequency.unit
    assert ts0_file.network.s_def == ts0_h5.network.s_def
    np.testing.assert_allclose(ts0_file.network.z0, ts0_h5.network.z0)
    assert [c.strip() for c in ts0_file.network.comments] == [
        c.strip() for c in ts0_h5.network.comments
    ]

    # repr не должен падать
    assert isinstance(repr(ts0_h5), str)

    reader.close()
    # файл должен быть закрыт
    assert not reader.h5.id.valid


# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not hasattr(h5py.File, "refresh"),
    reason="refresh() отсутствует в данной версии h5py",
)
def test_len_updates_after_append(tmp_dir, sample_dir):
    """
    Проверяем, что reader, открытый ДО записи, видит новые данные
    после len() благодаря File.refresh().
    """
    h5_path = pathlib.Path(tmp_dir) / "live.h5"

    writer = HDF5Backend(h5_path, mode="w")
    reader = HDF5Backend(h5_path, mode="r")  # открываем до записи

    assert len(reader) == 0

    file_backend = FileBackend(sample_dir)
    writer.append(file_backend.read(0))      # добавляем запись
    writer.close()

    # SWMR‑читатель должен увидеть 1 запись
    assert len(reader) == 1

    reader.close()
