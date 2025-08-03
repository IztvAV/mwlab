# tests/data_gen/test_api_basic.py
# -*- coding: utf-8 -*-
"""
Базовые smoke-тесты для публичного API модуля `mwlab.data_gen`.

Цели файла:
1. Убедиться, что ключевые символы доступны для импорта через корневой пакет.
2. Проверить, что `run_pipeline` и тонкая обёртка `GenRunner` корректно
   соединяют Source → Generator → Writer и возвращают статистику.
3. Минимизировать зависимость от тяжёлых компонентов (без файлового I/O, CPU-режим).

Все тесты быстрые и подходят для любого CI.
"""
from __future__ import annotations

from typing import Mapping, Sequence, List, Dict, Any

import pytest

from mwlab.data_gen import (
    # публичные объекты, которые должны быть экспортированы в __all__
    ListSource,
    ListWriter,
    DataGenerator,
    run_pipeline,
    GenRunner,
)

# ---------------------------------------------------------------------------
# 1. Проверяем, что публичные символы действительно экспортируются.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol_name",
    [
        "ListSource",
        "ListWriter",
        "DataGenerator",
        "run_pipeline",
        "GenRunner",
    ],
)
def test_public_exports(symbol_name):
    import mwlab.data_gen as mg

    assert hasattr(mg, symbol_name), f"mwlab.data_gen missing {symbol_name}"


# ---------------------------------------------------------------------------
# 2. Минимальный рабочий pipeline: ListSource → DummyGenerator → ListWriter
# ---------------------------------------------------------------------------


class _DummyGenerator(DataGenerator):
    """Очень простой генератор: возвращает входные parameters как outputs.

    * outputs — те же dict-ы (echo), что пришли (копии для чистоты);
    * meta    — список пустых словарей такой же длины.
    """

    def generate(self, params_batch: Sequence[Mapping[str, Any]]):
        outputs: List[Dict[str, Any]] = [dict(p) for p in params_batch]
        meta = [{} for _ in params_batch]
        return outputs, meta


def _make_simple_pipeline(batch_size: int = 2):
    data = [
        {"idx": 0, "x": 1},
        {"idx": 1, "x": 2},
        {"idx": 2, "x": 3},
    ]
    # ListSource сам проставит уникальные "__id"
    source = ListSource(data, shuffle=False, copy=True)
    generator = _DummyGenerator()
    writer = ListWriter()
    return source, generator, writer, len(data)


def test_run_pipeline_basic():
    """Smoke-тест прямого вызова run_pipeline."""
    source, gen, writer, n_expected = _make_simple_pipeline()

    stats = run_pipeline(source, gen, writer, batch_size=2, progress=False)

    # Проверяем статистику
    assert isinstance(stats, dict)
    assert stats.get("processed") == n_expected
    assert stats.get("failed") == 0

    # Проверяем, что Writer получил данные
    result = writer.result()
    assert isinstance(result, dict)
    assert "data" in result and "meta" in result and "params" in result

    xs = [row["x"] for row in result["data"]]
    assert xs == [1, 2, 3]
    assert result["meta"] == [{}] * n_expected
    # В params должны быть исходные dict-ы + добавленный "__id"
    assert all("__id" in p for p in result["params"])
    # Выходные data — echo входных params (с "__id")
    assert all("__id" in d for d in result["data"])


def test_genrunner_wrapper():
    """Проверяем, что GenRunner вызывает тот же пайплайн, что и run_pipeline."""
    source, gen, writer1, n_expected = _make_simple_pipeline(batch_size=1)
    writer2 = ListWriter()

    # GenRunner хранит конфигурацию и вызывает run_pipeline под капотом
    runner = GenRunner(batch_size=1, progress=False)
    stats1 = runner(source, gen, writer1)

    # Сравним с прямым run_pipeline
    stats2 = run_pipeline(source, gen, writer2, batch_size=1, progress=False)

    for stats in (stats1, stats2):
        assert stats.get("processed") == n_expected
        assert stats.get("failed") == 0

    # Сравним и полезную нагрузку
    res1 = writer1.result()
    res2 = writer2.result()
    assert [r["x"] for r in res1["data"]] == [1, 2, 3]
    assert [r["x"] for r in res2["data"]] == [1, 2, 3]
    assert res1["meta"] == [{}] * n_expected
    assert res2["meta"] == [{}] * n_expected


# ---------------------------------------------------------------------------
# 3. Негативный сценарий: генератор бросает исключение → mark_failed у Source
# ---------------------------------------------------------------------------


class _FailingGenerator(DataGenerator):
    def generate(self, params_batch):  # pragma: no cover (поведение очевидно)
        raise RuntimeError("intentional failure")


class _SpySource(ListSource):
    """ListSource с перехватом mark_failed для проверки ошибок."""

    def __init__(self, data, **kw):
        super().__init__(data, **kw)
        self.failed_calls: List[Dict[str, Any]] = []

    def mark_failed(self, ids, exc):  # noqa: D401
        self.failed_calls.append({"ids": list(ids), "error": str(exc)})


def test_error_handling():
    src = _SpySource([{"x": 1}], copy=True)
    gen = _FailingGenerator()
    writer = ListWriter()

    # run_pipeline не должен поднимать исключение наружу;
    # ошибка фиксируется в Source.mark_failed, а Writer ничего не получает
    stats = run_pipeline(src, gen, writer, batch_size=1, progress=False)

    assert stats.get("processed") == 0
    assert stats.get("failed") == 1

    res = writer.result()
    assert res["data"] == [] and res["meta"] == [] and res["params"] == []

    assert src.failed_calls, "mark_failed не был вызван"
    assert "intentional failure" in src.failed_calls[0]["error"]
