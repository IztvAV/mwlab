"""
PyTest‑набор для проверки основных сценариев работы класса
mwlab.io.touchstone.TouchstoneData.

Запуск из корня проекта:
    pytest -q
"""
import numpy as np
import pytest
import skrf as rf

from mwlab import TouchstoneData


# ---------- вспомогательное -------------------------------------------------
HERE = rf.os.path.dirname(__file__)
SRC  = rf.os.path.join(HERE, "Sprms.s2p")     # тестовый файл из CST


# ---------- тесты -----------------------------------------------------------
def test_load_params():
    """Файл загружается, параметры читаются в dict."""
    ts = TouchstoneData.load(SRC)
    assert isinstance(ts.params, dict)
    # в Touchstone‑файле из CST параметры точно есть
    assert len(ts.params) > 0


def test_save_roundtrip(tmp_path):
    """
    • Загружаем исходный файл
    • Меняем один параметр
    • Сохраняем -> перечитываем
    • Проверяем, что S‑матрица не изменилась,
      а параметр в header обновился
    """
    ts = TouchstoneData.load(SRC)
    old_s = ts.network.s.copy()

    # меняем (или добавляем) параметр w
    ts.params["w"] = ts.params.get("w", 0.0) + 0.123
    out = tmp_path / "roundtrip.s2p"
    ts.save(out)

    ts2 = TouchstoneData.load(out)
    assert np.allclose(ts2.network.s, old_s)
    assert pytest.approx(ts2.params["w"]) == ts.params["w"]


def test_init_from_network(tmp_path):
    """
    Создаём TouchstoneData «с нуля» из Network + params,
    сохраняем и перечитываем.
    """
    # простая синтетическая сеть 2×2 на 101 точке
    freq = rf.Frequency(1, 3, 101, unit="GHz")
    s = np.zeros((101, 2, 2), dtype=np.complex64)
    net = rf.Network(frequency=freq, s=s, z0=50)

    td = TouchstoneData(net, params={"a": 1.0, "b": 2.0})
    out = tmp_path / "gen.s2p"
    td.save(out)

    td2 = TouchstoneData.load(out)
    assert td2.params == {"a": 1.0, "b": 2.0}


def test_port_limit():
    """Network с 10‑ю портами должен выбросить ValueError."""
    freq = rf.Frequency(1, 1, 1, unit="GHz")
    s = np.zeros((1, 10, 10), dtype=np.complex64)
    net = rf.Network(frequency=freq, s=s, z0=50)

    with pytest.raises(ValueError):
        TouchstoneData(net)
