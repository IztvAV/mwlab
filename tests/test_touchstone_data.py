#tests/test_touchstone_data.py
"""
Тесты модуля mwlab.io.touchstone.TouchstoneData
"""

from __future__ import annotations

import numpy as np
import pathlib
import skrf as rf

from mwlab.io.touchstone import TouchstoneData


def test_load_and_parse_params(sample_file):
    """Убеждаемся, что .load() возвращает объект и корректно парсит параметры."""
    ts = TouchstoneData.load(sample_file)
    assert isinstance(ts.network, rf.Network)
    assert isinstance(ts.params, dict)
    # В Sprms.s2p заведомо должна быть хотя бы одна пара "ключ=значение"
    assert ts.params, "Словарь параметров пуст!"


def test_save_roundtrip(tmp_dir, dummy_network):
    """Сохраняем во временный файл → загружаем обратно → сравниваем."""
    params = {"w": 1.23, "gap": 0.1, "label": "sample1"}

    ts = TouchstoneData(dummy_network, params)
    dst = pathlib.Path(tmp_dir) / "roundtrip.s2p"
    ts.save(dst)

    ts2 = TouchstoneData.load(dst)

    # --- частоты и S-матрица совпадают
    np.testing.assert_allclose(ts.network.s, ts2.network.s)
    np.testing.assert_allclose(ts.network.f, ts2.network.f)

    # --- параметры
    assert ts2.params == params


def test_numpy_serialisation(dummy_network):
    """to_numpy() ↔ from_numpy() не теряет информацию."""
    params = {"k": 42.0, "note": "abc"}
    ts1 = TouchstoneData(dummy_network, params)
    bundle = ts1.to_numpy()
    ts2 = TouchstoneData.from_numpy(bundle)

    # Проверяем S-матрицу
    np.testing.assert_allclose(ts1.network.s, ts2.network.s)
    # Проверяем параметры
    assert ts2.params == params

