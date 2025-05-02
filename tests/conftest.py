#tests/conftest.py
"""
Общие фикстуры, доступные во всех тестах.
"""

from __future__ import annotations

import pathlib

import pytest
import skrf as rf
import numpy as np


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    """Корень репозитория = два уровня выше этого файла."""
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_file(repo_root: pathlib.Path) -> pathlib.Path:
    """
    `Sprms.s2p` лежит рядом с тестами (см. условие задачи).
    Проверяем, что файл существует, иначе помечаем тест как xfail.
    """
    fp = repo_root / "tests" / "Sprms.s2p"
    if not fp.exists():
        pytest.xfail("Файл Sprms.s2p не найден")
    return fp


@pytest.fixture(scope="session")
def sample_dir(repo_root: pathlib.Path) -> pathlib.Path:
    """
    Каталог с ~200 .s2p для более «тяжёлых» тестов.
    """
    folder = repo_root / "Data" / "Filter12"
    if not folder.exists():
        pytest.xfail("Каталог Data/Filter12 не найден")
    return folder


@pytest.fixture()
def tmp_dir(tmp_path_factory) -> pathlib.Path:
    """
    Временная директория, автоматически удаляется после теста.
    """
    return tmp_path_factory.mktemp("mwlab_test")


@pytest.fixture()
def dummy_network() -> rf.Network:
    """
    Создаёт простую 2-портовую сеть с нулевыми S-параметрами
    на сетке из 5 точек (1…5 ГГц) — годится для round-trip тестов.
    """
    f = rf.Frequency(1, 5, 5, "GHz")
    s = np.zeros((len(f), 2, 2), dtype=np.complex64)
    return rf.Network(frequency=f, s=s)
