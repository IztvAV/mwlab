"""tests/data_gen/test_runner.py
================================================
Тесты высокого уровня для `mwlab.data_gen.runner` — проверяем работу
`run_pipeline` и thin-wrapper-класса `GenRunner`.
"""
from __future__ import annotations

from typing import Any, List, Mapping, MutableMapping, Sequence, Tuple

import pytest

from mwlab.data_gen.base import DataGenerator, ParamSource, Batch
from mwlab.data_gen.sources import ListSource
from mwlab.data_gen.writers import ListWriter
from mwlab.data_gen.runner import run_pipeline, GenRunner


# ---------------------------------------------------------------------------
# Вспомогательные генераторы для тестов
# ---------------------------------------------------------------------------

class EchoGenerator(DataGenerator):
    """Пасс-тру генератор: outputs == params; meta — список dict'ов той же длины."""

    # ВАЖНО: у базового DataGenerator нет __init__, поэтому super().__init__ не вызываем
    def generate(
        self,
        params_batch: Sequence[Mapping[str, Any]],
    ) -> Tuple[Sequence[Any], Sequence[Mapping[str, Any]]]:
        # Возвращаем список params как outputs и список meta той же длины
        return list(params_batch), [{} for _ in params_batch]


class WithPreprocessGenerator(DataGenerator):
    """Генератор, проверяющий вызовы preprocess() и preprocess_batch()."""

    def __init__(self):
        # не вызываем super().__init__
        self.pre_calls: int = 0
        self.pre_batch_calls: int = 0

    # помечаем каждую точку
    def preprocess(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self.pre_calls += 1
        d = dict(params)
        d["pre"] = True
        return d

    # и весь батч разом
    def preprocess_batch(self, params_batch: Batch) -> Batch:
        self.pre_batch_calls += 1
        out: List[MutableMapping[str, Any]] = [dict(p) for p in params_batch]
        for p in out:
            p["pre_b"] = True
        return out  # тот же размер и порядок

    def generate(
        self, params_batch: Sequence[Mapping[str, Any]]
    ) -> Tuple[Sequence[Any], Sequence[Mapping[str, Any]]]:
        # возвращаем уже «подготовленные» параметры (после pre*/pre*_batch)
        return list(params_batch), [{} for _ in params_batch]


class BoomOnFlagGenerator(DataGenerator):
    """Бросает исключение, если в точке стоит флаг boom=True."""

    def generate(self, params_batch: Sequence[Mapping[str, Any]]):
        p = params_batch[0]
        if p.get("boom"):
            raise RuntimeError("boom")
        return [dict(p)], [{}]


# ---------------------------------------------------------------------------
# Вспомогательный Source с дубликатами __id (обходит проверки ListSource)
# ---------------------------------------------------------------------------

class DuplicateIdSource(ParamSource):
    """Источником служит переданный список dict-ов как есть (без доп. проверок)."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self._rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield from self._rows


# ---------------------------------------------------------------------------
# 1) Базовый happy-path
# ---------------------------------------------------------------------------

def test_run_pipeline_happy_path():
    points = [{"v": i} for i in range(5)]  # без __id — ListSource добавит
    src = ListSource(points, shuffle=False, copy=True)
    gen = EchoGenerator()
    wr = ListWriter()

    # размер батча задаём параметром run_pipeline
    run_pipeline(src, gen, wr, batch_size=2, progress=False)

    res = wr.result()
    # outputs могли получить служебные поля (__id), поэтому сравниваем полезные
    assert [d["v"] for d in res["data"]] == [p["v"] for p in points]
    # meta — список dict'ов длиной == числу точек
    assert isinstance(res["meta"], list)
    assert len(res["meta"]) == len(points)


# ---------------------------------------------------------------------------
# 2) preprocess() + preprocess_batch() вызываются и модификации доходят до outputs
# ---------------------------------------------------------------------------

def test_preprocess_hooks_are_called_and_applied():
    pts = [{"x": 1}, {"x": 2}, {"x": 3}]
    src = ListSource(pts, shuffle=False, copy=True)
    gen = WithPreprocessGenerator()
    wr = ListWriter()

    run_pipeline(src, gen, wr, batch_size=2, progress=False)

    res = wr.result()
    assert len(res["data"]) == len(pts)
    # каждая точка получила метки из preprocess() и preprocess_batch()
    for p in res["data"]:
        assert p.get("pre") is True
        assert p.get("pre_b") is True

    # счётчики вызовов: preprocess — по числу точек; preprocess_batch — по числу батчей (2+1 → 2 вызова)
    assert gen.pre_calls == len(pts)
    assert gen.pre_batch_calls == 2


# ---------------------------------------------------------------------------
# 3) Ошибки генератора не прерывают пайплайн (точки с boom=True пропускаются)
# ---------------------------------------------------------------------------

def test_generator_error_is_caught_and_pipeline_continues():
    pts = [{"x": 1}, {"x": 2, "boom": True}, {"x": 3}]
    src = ListSource(pts, shuffle=False, copy=True)
    gen = BoomOnFlagGenerator()
    wr = ListWriter()

    # не должно бросать наружу
    stats = run_pipeline(src, gen, wr, batch_size=1, progress=False)

    data = wr.result()["data"]
    assert [p["x"] for p in data] == [1, 3]  # «выпавшая» точка пропущена
    # meta также той же длины
    assert len(wr.result()["meta"]) == 2
    # статистика отражает 1 fail
    assert stats["processed"] == 2 and stats["failed"] == 1


# ---------------------------------------------------------------------------
# 4) Дубликаты __id внутри одного батча → раннер помечает fail (исключение не утекает)
# ---------------------------------------------------------------------------

def test_duplicate_ids_in_same_batch_are_marked_failed():
    # Подготовим два элемента с одинаковым __id, чтобы они попали в один батч
    rows = [{"__id": "same", "x": 1}, {"__id": "same", "x": 2}]
    src = DuplicateIdSource(rows)
    gen = EchoGenerator()
    wr = ListWriter()

    # раннер не должен бросать наружу
    stats = run_pipeline(src, gen, wr, batch_size=2, progress=False)

    # ничего не записано, обе точки упали
    assert wr.result()["data"] == []
    assert stats["processed"] == 0 and stats["failed"] == 2


# ---------------------------------------------------------------------------
# 5) on_batch_end callback вызывается с корректными размерами батчей
# ---------------------------------------------------------------------------

def test_on_batch_end_is_called_with_expected_batches():
    pts = [{"x": i} for i in range(5)]  # батчи: 2, 2, 1
    src = ListSource(pts, shuffle=False, copy=True)
    gen = EchoGenerator()
    wr = ListWriter()

    calls: List[dict] = []

    def on_batch_end(payload: Mapping[str, Any]):
        # проверим минимальный состав payload и собираем размеры коммитов
        assert set(payload).issuperset({"ok_delta", "fail_delta", "processed_total", "failed_total", "ids", "exception"})
        calls.append(dict(payload))

    run_pipeline(src, gen, wr, batch_size=2, on_batch_end=on_batch_end, progress=False)

    # ожидаем три вызова (2,2,1) — читаем из ok_delta (на успехе fail_delta=0)
    sizes = [c["ok_delta"] + c["fail_delta"] for c in calls]
    assert sizes == [2, 2, 1]
    # processed_total должен монотонно расти и закончиться размером датасета
    assert calls[-1]["processed_total"] == len(pts)
    assert all(c["exception"] is None for c in calls)  # ошибок не было


# ---------------------------------------------------------------------------
# 6) GenRunner thin-wrapper: вызывается как функция (__call__)
# ---------------------------------------------------------------------------

def test_genrunner_wrapper_call_only():
    pts = [{"v": i} for i in range(3)]
    src = ListSource(pts, shuffle=False, copy=True)
    gen = EchoGenerator()
    wr = ListWriter()

    runner = GenRunner(batch_size=1, progress=False)
    stats = runner(src, gen, wr)  # __call__

    assert wr.result()["data"] and [d["v"] for d in wr.result()["data"]] == [0, 1, 2]
    assert stats["processed"] == 3 and stats["failed"] == 0
