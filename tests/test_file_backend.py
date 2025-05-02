#tests/test_file_backend.py
"""
Тесты FileBackend (старое поведение «каталог с .sNp»).
"""

from __future__ import annotations

import numpy as np
import pytest
from mwlab.io.backends.file_backend import FileBackend
from mwlab.io.touchstone import TouchstoneData

def test_empty_directory_raises(tmp_dir):
    with pytest.raises(FileNotFoundError, match="нет файлов"):
        FileBackend(tmp_dir)

def test_len_and_read(sample_dir):
    backend = FileBackend(sample_dir)
    assert len(backend) == 200  # известно из условия
    sample = backend.read(0)
    assert isinstance(sample, TouchstoneData)


def test_append(tmp_dir, sample_file):
    """
    Проверяем, что append():
      • увеличивает длину,
      • действительно создает новый файл в директории.
    """
    ts = TouchstoneData.load(sample_file)
    # сначала сохраняем руками
    ts.save(tmp_dir / "init.s2p")

    backend = FileBackend(tmp_dir)
    n0 = len(backend)

    backend.append(ts)

    assert len(backend) == n0 + 1
    assert any(tmp_dir.glob("sample_*.s2p"))

def test_append_content(tmp_dir, sample_file):
    """
    Проверяем, что append() не только записывает файл,
    но и делает это правильно: round-trip проверка
    """
    ts = TouchstoneData.load(sample_file)

    # сначала руками кладём один .s2p, чтобы FileBackend не упал
    ts.save(tmp_dir / "init.s2p")

    backend = FileBackend(tmp_dir)
    backend.append(ts)

    # Последний добавленный элемент — в конце списка
    new_ts = backend.read(len(backend) - 1)

    assert new_ts.network.number_of_ports == ts.network.number_of_ports
    np.testing.assert_allclose(new_ts.network.s, ts.network.s)


def test_append_filename_collision(tmp_dir, sample_file):
    """
    Тест на конфликт имен
    """
    # Сначала вручную положим sample_000000.s2p
    ts = TouchstoneData.load(sample_file)
    first_path = tmp_dir / "sample_000000.s2p"
    ts.save(first_path)

    backend = FileBackend(tmp_dir)
    backend.append(ts)

    # Убедимся, что новый файл – sample_000001.s2p
    assert (tmp_dir / "sample_000001.s2p").exists()


