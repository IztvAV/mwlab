# tests/test_cm_io.py
"""pytest-тесты для mwlab.filters.cm_io
======================================

Покрываем:
* write_matrix / read_matrix для ASCII и JSON
* layout=TAIL/SL/CUSTOM и layout="auto" (эвристика)
* precision / delimiter
* round-trip CouplingMatrix сохранение/загрузка
* Ошибочные параметры (fmt, permutation и т.п.)

Запуск:  pytest -q tests/test_cm_io.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from mwlab.filters.topologies import get_topology
from mwlab.filters.cm import CouplingMatrix, MatrixLayout
from mwlab.filters import cm_io


# ----------------------------------------------------------------------------
# Общие фикстуры
# ----------------------------------------------------------------------------

@pytest.fixture()
def topo_folded():
    return get_topology("folded", order=4)


@pytest.fixture()
def cm_example(topo_folded):
    mvals = {
        "M1_2": 1.05, "M2_3": 0.90, "M3_4": 1.05, "M1_4": 0.25,
        "M1_5": 0.60, "M4_6": 0.80,
    }
    return CouplingMatrix(topo_folded, mvals, qu=700.0)


# ----------------------------------------------------------------------------
# write/read: ASCII (SL), JSON (TAIL)
# ----------------------------------------------------------------------------

def _assert_close_mvals(cm1: CouplingMatrix, cm2: CouplingMatrix, tol=1e-6):
    # сравниваем только ненулевые
    nz1 = {k: v for k, v in cm1.mvals.items() if abs(v) > 1e-12}
    nz2 = {k: v for k, v in cm2.mvals.items() if abs(v) > 1e-12}
    assert nz1.keys() == nz2.keys()
    for k in nz1:
        assert nz1[k] == pytest.approx(nz2[k], rel=tol, abs=tol)


def test_write_read_ascii_sl(tmp_path: Path, cm_example: CouplingMatrix):
    path = tmp_path / "test_sl_ascii.cm"
    cm_io.write_matrix(cm_example, path, layout=MatrixLayout.SL, fmt="ascii", precision=10, delimiter="\t")

    # проверим, что файл существует и начинается с хедера
    text = path.read_text().splitlines()
    assert text[0].startswith("# mwlab-cm")
    assert "layout=SL" in text[0]

    cm_loaded = cm_io.read_matrix(path, topo=cm_example.topo, layout="auto")
    _assert_close_mvals(cm_example, cm_loaded)
    assert cm_loaded.qu == cm_example.qu


def test_write_read_json_tail(tmp_path: Path, cm_example: CouplingMatrix):
    path = tmp_path / "test_tail.json"
    cm_io.write_matrix(cm_example, path, layout=MatrixLayout.TAIL, fmt="json")

    blob = json.loads(path.read_text())
    assert blob["layout"] == "TAIL"
    assert blob["order"] == cm_example.topo.order

    cm_loaded = cm_io.read_matrix(path, topo=cm_example.topo, layout="auto")
    _assert_close_mvals(cm_example, cm_loaded)


# ----------------------------------------------------------------------------
# CUSTOM permutation
# ----------------------------------------------------------------------------

def test_custom_permutation_roundtrip(tmp_path: Path, cm_example: CouplingMatrix):
    K = cm_example.topo.size
    perm = list(range(K))[::-1]

    path = tmp_path / "custom_ascii.cm"
    cm_io.write_matrix(cm_example, path, layout=MatrixLayout.CUSTOM,
                       fmt="ascii", delimiter=",",
                       precision=6, permutation=perm,)
    # читаем обратно с тем же perm
    with pytest.raises(ValueError):
        # layout=CUSTOM, но permutation не указан
        cm_io.read_matrix(path, topo=cm_example.topo, layout=MatrixLayout.CUSTOM)

    cm_loaded = cm_io.read_matrix(path, topo=cm_example.topo, layout=MatrixLayout.CUSTOM,
                                  delimiter=",", permutation=perm,)

    _assert_close_mvals(cm_example, cm_loaded)


# ----------------------------------------------------------------------------
# layout="auto" эвристика: создаём SL и TAIL
# ----------------------------------------------------------------------------

def test_layout_auto_detection(tmp_path: Path, cm_example: CouplingMatrix):
    # SL-файл
    p_sl = tmp_path / "auto_sl.cm"
    cm_io.write_matrix(cm_example, p_sl, layout=MatrixLayout.SL, fmt="ascii")
    cm_sl = cm_io.read_matrix(p_sl, topo=cm_example.topo, layout="auto")
    _assert_close_mvals(cm_example, cm_sl)

    # TAIL-файл
    p_tail = tmp_path / "auto_tail.cm"
    cm_io.write_matrix(cm_example, p_tail, layout=MatrixLayout.TAIL, fmt="ascii")
    cm_tail = cm_io.read_matrix(p_tail, topo=cm_example.topo, layout="auto")
    _assert_close_mvals(cm_example, cm_tail)


# ----------------------------------------------------------------------------
# precision / delimiter check
# ----------------------------------------------------------------------------

def test_precision_and_delimiter(tmp_path: Path, cm_example: CouplingMatrix):
    path = tmp_path / "prec_delim.cm"
    cm_io.write_matrix(cm_example, path, layout=MatrixLayout.SL, fmt="ascii", precision=4, delimiter=", ")
    txt = path.read_text()
    # проверяем, что разделитель реально ", " (хотя бы в первой строке данных)
    data_line = [ln for ln in txt.splitlines() if not ln.startswith("#")][0]
    assert ", " in data_line


# ----------------------------------------------------------------------------
# Ошибочные случаи
# ----------------------------------------------------------------------------

def test_bad_fmt_raises(tmp_path: Path, cm_example: CouplingMatrix):
    path = tmp_path / "bad.xxx"
    with pytest.raises(ValueError):
        cm_io.write_matrix(cm_example, path, fmt="binary")


def test_read_matrix_non_square(tmp_path: Path, cm_example: CouplingMatrix):
    # создаём кривой json
    bad = {
        "layout": "TAIL",
        "order": cm_example.topo.order,
        "ports": cm_example.topo.ports,
        "M": [[0, 1], [2, 3], [4, 5]],  # не квадрат
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        cm_io.read_matrix(path, topo=cm_example.topo)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])  # для ручного запуска
