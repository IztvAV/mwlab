"""tests/data_gen/test_sources.py
================================================
Юнит-тесты для стандартных источников параметров (ParamSource).

Покрываем:
* ListSource
* CsvSource
* DesignSpaceSource (режим stream и full-plan)
* FolderSource

Каждый тест использует фикстуру `tmp_path`; внешних зависимостей (CUDA,
pyDOE2, …) нет — всё работает на чистом CPU.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import pytest

from mwlab.data_gen.sources import (
    ListSource,
    CsvSource,
    DesignSpaceSource,
    FolderSource,
)
from mwlab.opt.design.space import DesignSpace, ContinuousVar

# ---------------------------------------------------------------------------
# 1. ListSource
# ---------------------------------------------------------------------------

def test_listsource_basic():
    data = [{"idx": i} for i in range(5)]
    src = ListSource(data, shuffle=False, copy=True)

    # len / iteration
    assert len(src) == 5
    assert [d["idx"] for d in src] == list(range(5))

    # shuffle=True должен изменить порядок
    src_shuf = ListSource(data, shuffle=True, copy=True)
    order = [d["idx"] for d in src_shuf]
    assert sorted(order) == list(range(5)) and order != list(range(5))


# ---------------------------------------------------------------------------
# 2. CsvSource  (reserve / mark_done / mark_failed)
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict[str, str]]):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_csvsource_flow(tmp_path: Path):
    rows = [
        {"a": "1", "b": "x"},
        {"a": "2", "b": "y"},
    ]
    csv_file = tmp_path / "params.csv"
    _write_csv(csv_file, rows)

    with CsvSource(csv_file) as src:
        it = iter(src)

        # Берём первую строку
        first = next(it)
        # Проверяем полезные поля (нормализуем к строкам для стабильности)
        assert str(first["a"]) == "1" and str(first["b"]) == "x"

        # Резервируем и помечаем как done по __id
        src.reserve([first["__id"]])
        src.mark_done([first["__id"]])

        # Вторая строка остаётся pending → помечаем failed
        second = next(it)
        src.mark_failed([second["__id"]], RuntimeError("boom"))

    # файл обновлён
    import pandas as pd

    df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
    assert list(df["__status"]) == ["done", "failed"]
    assert "boom" in df.loc[1, "__error"]


# ---------------------------------------------------------------------------
# 3. DesignSpaceSource (stream + full-plan)
# ---------------------------------------------------------------------------

def _make_space():
    return DesignSpace({
        "x": ContinuousVar(-1.0, 1.0),
        "y": ContinuousVar(0.0, 10.0),
    })


@pytest.mark.parametrize("sampler_name, n_total", [("normal", 8), ("lhs", 16)])
def test_designspacesource(tmp_path: Path, sampler_name: str, n_total: int):
    space = _make_space()
    # reserve_n маленькое, чтобы проверить инкрементальную дозагрузку очереди
    src = DesignSpaceSource(space, sampler=sampler_name, n_total=n_total, reserve_n=4)

    pts = list(src)
    assert len(pts) == n_total

    # нет повторений по *содержимому параметров* (игнорируем служебный __id)
    seen = {
        tuple(sorted((k, v) for k, v in p.items() if k != "__id"))
        for p in pts
    }
    assert len(seen) == n_total


# ---------------------------------------------------------------------------
# 4. FolderSource (JSON / YAML)
# ---------------------------------------------------------------------------

def test_foldersource(tmp_path: Path):
    root = tmp_path / "inbox"
    root.mkdir()

    # создаём два JSON файла
    files = []
    for i in range(2):
        p = root / f"point_{i}.json"
        p.write_text(json.dumps({"idx": i}))
        files.append(p)

    src = FolderSource(root, move_done=True, move_failed=True)

    with src:
        it = iter(src)
        p0 = next(it)
        p1 = next(it)

        # mark first as done, second as fail (по __id)
        src.mark_done([p0["__id"]])
        src.mark_failed([p1["__id"]], RuntimeError("bad"))

    # файлы перемещены
    done_dir = root / "done"
    failed_dir = root / "failed"

    assert (done_dir / files[0].name).exists()
    assert (failed_dir / files[1].name).exists()
