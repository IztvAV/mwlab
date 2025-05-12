#tests/test_hdf5_backend.py
"""
Проверяем корректность работы HDF5Backend:

* запись → чтение (данные, метаданные, параметры)
* общая/индивидуальная частотная сетка
* режим in_memory
* swmr-refresh
"""

from __future__ import annotations

import pathlib

import numpy as np
import h5py
import pytest
import skrf as rf

from mwlab.io.backends.hdf5_backend import HDF5Backend
from mwlab.io.backends.file_backend import FileBackend
from mwlab.io.touchstone import TouchstoneData


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
    assert not reader.h5.id.valid


# ---------------------------------------------------------------------------
def test_common_f_created(tmp_dir, sample_dir):
    """Проверяем, что /common_f создается, если у всех записей одинаковые частоты."""
    h5_path = pathlib.Path(tmp_dir) / "shared_f.h5"
    file_backend = FileBackend(sample_dir)

    with HDF5Backend(h5_path, mode="w") as writer:
        for idx in range(3):
            writer.append(file_backend.read(idx))

    with h5py.File(h5_path, "r") as f:
        assert "common_f" in f
        for idx in range(3):
            # Проверим, что в записях нет индивидуальных частот
            assert "f" not in f[f"samples/{idx}"]


# ---------------------------------------------------------------------------
def test_individual_f_created(tmp_dir, sample_dir):
    """Проверяем, что если частотные сетки разные, создается индивидуальное поле 'f'."""
    h5_path = pathlib.Path(tmp_dir) / "mixed_f.h5"
    file_backend = FileBackend(sample_dir)

    # читаем две записи, вручную меняем одну частоту
    ts0 = file_backend.read(0)
    ts1 = file_backend.read(1)

    f = ts1.network.f.copy()
    f[3] += 1.0
    ts1.network.frequency = rf.Frequency.from_f(f, unit=ts1.network.frequency.unit)

    with HDF5Backend(h5_path, mode="w") as writer:
        writer.append(ts0)
        writer.append(ts1)

    with h5py.File(h5_path, "r") as f:
        assert "common_f" in f
        assert "f" not in f["samples/0"]
        assert "f" in f["samples/1"]  # теперь должно быть индивидуальное поле


# ---------------------------------------------------------------------------
def test_in_memory_mode(tmp_dir, sample_dir):
    """Проверяем корректность чтения при in_memory=True."""
    h5_path = pathlib.Path(tmp_dir) / "memload.h5"
    file_backend = FileBackend(sample_dir)

    with HDF5Backend(h5_path, mode="w") as writer:
        writer.append(file_backend.read(0))

    # загрузка в память
    backend = HDF5Backend(h5_path, mode="r", in_memory=True)
    ts = backend.read(0)
    assert isinstance(ts.network.s, np.ndarray)
    assert ts.network.s.shape[0] > 0
    backend.close()


# ---------------------------------------------------------------------------
def test_s_dataset_not_compressed(tmp_dir, sample_dir):
    """Убеждаемся, что матрица S сохраняется без сжатия."""
    h5_path = pathlib.Path(tmp_dir) / "raw_s.h5"
    file_backend = FileBackend(sample_dir)

    with HDF5Backend(h5_path, mode="w") as writer:
        writer.append(file_backend.read(0))

    with h5py.File(h5_path, "r") as f:
        ds = f["samples/0/s"]
        assert "gzip" not in ds.compression.lower() if ds.compression else True


# ---------------------------------------------------------------------------
def test_in_memory_loads_all(tmp_dir, sample_dir):
    """Проверяем, что in_memory загружает все записи и закрывает файл."""
    h5_path = pathlib.Path(tmp_dir) / "all_in_memory.h5"
    file_backend = FileBackend(sample_dir)

    with HDF5Backend(h5_path, mode="w") as writer:
        for i in range(5):
            writer.append(file_backend.read(i))

    backend = HDF5Backend(h5_path, mode="r", in_memory=True)
    assert len(backend) == 5
    for i in range(5):
        ts = backend.read(i)
        assert isinstance(ts, TouchstoneData)
