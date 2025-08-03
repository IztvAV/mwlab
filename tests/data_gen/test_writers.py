"""tests/data_gen/test_writers.py
================================================
Юнит-тесты для стандартных приёмников данных (Writer-ов):

* ListWriter
* TensorWriter
* TouchstoneDirWriter
* HDF5Writer
* RAMWriter

Тесты не используют runner; мы вызываем .write(...) напрямую,
чтобы изолированно проверить поведение/валидацию. Зависимости
( torch, scikit-rf, HDF5/RAM backends ) помечены importorskip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Sequence

import pytest

from mwlab.data_gen.writers import (
    ListWriter,
    TensorWriter,
    TouchstoneDirWriter,
    HDF5Writer,
    RAMWriter,
)

# ---------------------------------------------------------------------------
# Вспомогательные генераторы данных
# ---------------------------------------------------------------------------

def _make_params(n: int) -> List[Mapping[str, Any]]:
    """Сервис: список params с обязательным '__id'."""
    return [{"__id": f"p{i}", "i": i} for i in range(n)]


# --- TouchstoneData конструктор (через scikit-rf) ---------------------------

def _make_touchstone(n_ports: int = 2, n_freq: int = 5):
    """Создать простой TouchstoneData с синтетическим S, используя scikit-rf."""
    rf = pytest.importorskip("skrf")
    from mwlab.io.touchstone import TouchstoneData  # noqa: WPS433 (runtime import)

    # Частоты в Гц
    freq = rf.Frequency.from_f([1e9 + k * 1e6 for k in range(n_freq)], unit="Hz")

    # Простейшая S-матрица: диагональные элементы ~ -10 dB, остальное ~ -20 dB
    import numpy as np

    s = np.zeros((n_freq, n_ports, n_ports), dtype=complex)
    for f in range(n_freq):
        for i in range(n_ports):
            for j in range(n_ports):
                if i == j:
                    s[f, i, j] = 10 ** (-10 / 20) * np.exp(1j * 0.01 * f)
                else:
                    s[f, i, j] = 10 ** (-20 / 20) * np.exp(1j * 0.02 * f)

    net = rf.Network(frequency=freq, s=s)
    return TouchstoneData(net, params={"kind": f"{n_ports}p"})


# ---------------------------------------------------------------------------
# 1) ListWriter
# ---------------------------------------------------------------------------

def test_listwriter_basic():
    W = ListWriter()

    outputs = [{"y": 1}, {"y": 2}]
    meta    = [{}, {}]
    params  = _make_params(2)

    W.write(outputs, meta, params)
    res = W.result()

    assert res["data"] == outputs
    assert res["meta"] == meta
    assert res["params"] == params


def test_listwriter_len_mismatch_raises():
    W = ListWriter()
    with pytest.raises(ValueError):
        W.write([1, 2], [{}], _make_params(2))  # meta короче


# ---------------------------------------------------------------------------
# 2) TensorWriter
# ---------------------------------------------------------------------------

def test_tensorwriter_stack_same_shapes():
    torch = pytest.importorskip("torch")
    W = TensorWriter(stack=True)

    outs = [torch.zeros(3), torch.ones(3)]
    meta = [{}, {}]
    params = _make_params(2)

    W.write(outs, meta, params)
    res = W.result()

    # Должен произойти stack → (N, *shape)
    assert hasattr(res["data"], "shape")
    assert tuple(res["data"].shape) == (2, 3)
    assert res["meta"] == meta and res["params"] == params


def test_tensorwriter_no_stack_mixed_shapes():
    torch = pytest.importorskip("torch")
    W = TensorWriter(stack=True)  # включен, но формы будут разные → список

    outs = [torch.zeros(3), torch.zeros(4)]
    meta = [{}, {}]
    params = _make_params(2)

    W.write(outs, meta, params)
    res = W.result()

    # Из-за разных форм получаем список
    assert isinstance(res["data"], list) and len(res["data"]) == 2


def test_tensorwriter_stack_disabled():
    torch = pytest.importorskip("torch")
    W = TensorWriter(stack=False)

    outs = [torch.zeros(2), torch.zeros(2)]
    meta = [{}, {}]
    params = _make_params(2)

    W.write(outs, meta, params)
    res = W.result()
    assert isinstance(res["data"], list)


# ---------------------------------------------------------------------------
# 3) TouchstoneDirWriter
# ---------------------------------------------------------------------------

def test_touchstonedirwriter_saves_files(tmp_path: Path):
    ts1 = _make_touchstone(n_ports=2, n_freq=4)
    ts2 = _make_touchstone(n_ports=3, n_freq=4)

    W = TouchstoneDirWriter(tmp_path, stem_format="sample_{idx:03d}", overwrite=False)

    outputs = [ts1, ts2]
    meta = [{"extra": 1}, {"note": "ok"}]
    params = _make_params(2)

    W.write(outputs, meta, params)

    # Файлы появляются с ожидаемыми расширениями
    files = sorted(p.name for p in tmp_path.iterdir())
    # sample_000.s2p, sample_001.s3p
    assert any(name.endswith(".s2p") for name in files)
    assert any(name.endswith(".s3p") for name in files)

    # Метаданные были in-place добавлены в TouchstoneData.params
    assert "extra" in outputs[0].params and outputs[0].params["extra"] == 1
    assert "note" in outputs[1].params and outputs[1].params["note"] == "ok"


def test_touchstonedirwriter_overwrite_flag(tmp_path: Path):
    ts = _make_touchstone(n_ports=2, n_freq=3)

    # 1-й проход: создаём файл sample_000.s2p
    W1 = TouchstoneDirWriter(tmp_path, stem_format="sample_{idx:03d}", overwrite=False)
    W1.write([ts], [{}], _make_params(1))

    # 2-й writer с тем же stem_format/счётчиком начнёт с того же имени → FileExistsError
    W2 = TouchstoneDirWriter(tmp_path, stem_format="sample_{idx:03d}", overwrite=False)
    with pytest.raises(FileExistsError):
        W2.write([_make_touchstone(2, 3)], [{}], _make_params(1))

    # А с overwrite=True — разрешено
    W3 = TouchstoneDirWriter(tmp_path, stem_format="sample_{idx:03d}", overwrite=True)
    W3.write([_make_touchstone(2, 3)], [{}], _make_params(1))


def test_touchstonedirwriter_type_checks(tmp_path: Path):
    W = TouchstoneDirWriter(tmp_path)
    # Неверный тип outputs
    with pytest.raises(TypeError):
        W.write([object()], [{}], _make_params(1))  # type: ignore[list-item]
    # Неверный meta (не Mapping)
    ts = _make_touchstone(2, 3)
    with pytest.raises(TypeError):
        W.write([ts], [42], _make_params(1))  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 4) HDF5Writer  (пропускаем, если backend отсутствует)
# ---------------------------------------------------------------------------

def test_hdf5writer_basic(tmp_path: Path):
    # Импортируем бэкенд — если его нет, тест пропускается
    pytest.importorskip("mwlab.io.backends.hdf5_backend")

    path = tmp_path / "data.h5"
    ts1 = _make_touchstone(2, 8)
    ts2 = _make_touchstone(2, 8)

    # Проверяем overwrite=True (если файл будет существовать)
    with HDF5Writer(path, compression=None, overwrite=True) as W:
        W.write([ts1, ts2], [{}, {}], _make_params(2))
        W.flush()

    # Файл создан и непустой
    assert path.exists() and path.stat().st_size > 0


def test_hdf5writer_type_checks(tmp_path: Path):
    pytest.importorskip("mwlab.io.backends.hdf5_backend")
    path = tmp_path / "bad.h5"

    with HDF5Writer(path, overwrite=True) as W:
        with pytest.raises(TypeError):
            W.write([object()], [{}], _make_params(1))  # type: ignore[list-item]
        ts = _make_touchstone(2, 3)
        with pytest.raises(TypeError):
            W.write([ts], [123], _make_params(1))  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 5) RAMWriter  (пропускаем, если backend отсутствует)
# ---------------------------------------------------------------------------

def test_ramwriter_append_and_backend_type():
    # Если RAMBackend отсутствует — пропускаем
    RAMBackend = pytest.importorskip("mwlab.io.backends.in_memory").RAMBackend  # noqa: N816

    W = RAMWriter()
    ts1 = _make_touchstone(2, 5)
    ts2 = _make_touchstone(2, 5)

    W.write([ts1, ts2], [{}, {}], _make_params(2))

    be = W.backend()
    # Как минимум тип backend — верный
    assert isinstance(be, RAMBackend)

    # (Опционально) пробуем, что append сработал — многие in-memory контейнеры
    # реализуют __len__(). Если нет — просто не проверяем длину.
    try:
        assert len(be) >= 2  # type: ignore[arg-type]
    except TypeError:
        pass
