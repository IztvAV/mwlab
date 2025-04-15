import numpy as np
import tempfile
import pytest
from mwlab.fileio import save_touchstone, load_touchstone
from mwlab.touchstone import TouchstoneFile

def test_mismatched_lengths_should_raise():
    f = np.array([1e9, 2e9])
    s = np.random.rand(1, 2, 2) + 1j * np.random.rand(1, 2, 2)
    with tempfile.NamedTemporaryFile(suffix=".s2p", delete=False) as tmp:
        with pytest.raises(ValueError, match="Длины массива частот и S-параметров не совпадают."):
            save_touchstone(tmp.name, f, s)

def test_invalid_shape_should_raise():
    f = np.array([1e9])
    s = np.array([[[1+1j, 2+2j]]])  # неправильная форма (не квадрат)
    with tempfile.NamedTemporaryFile(suffix=".s2p", delete=False) as tmp:
        with pytest.raises(ValueError):
            save_touchstone(tmp.name, f, s)

def test_invalid_format_should_raise():
    f = np.array([1e9])
    s = np.array([[[1+1j]]])
    with tempfile.NamedTemporaryFile(suffix=".s1p", delete=False) as tmp:
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            save_touchstone(tmp.name, f, s, format="XX")

def test_invalid_unit_should_raise():
    f = np.array([1e9])
    s = np.array([[[1+1j]]])
    with tempfile.NamedTemporaryFile(suffix=".s1p", delete=False) as tmp:
        with pytest.raises(ValueError, match="Поддерживаются только единицы"):
            save_touchstone(tmp.name, f, s, unit="GZH")

def test_invalid_touchstone_data_format():
    with tempfile.NamedTemporaryFile(suffix=".s2p", delete=False, mode="w", encoding="ISO-8859-1") as tmp:
        tmp.write("# GHZ S XX R 50\n")  # 'XX' — неподдерживаемый формат данных
        tmp.write("1.0 0.1 0.2 0.3 0.4\n")
        tmp.flush()
        with pytest.raises(ValueError, match="Поддерживаются только форматы"):
            load_touchstone(tmp.name)

def test_to_format_error_non_RI():
    f = np.array([1e9])
    s = np.array([[[1+1j]]])
    ts = TouchstoneFile.from_data(f, s, format="MA")
    with pytest.raises(ValueError, match="Конвертация возможна только из формата 'RI'"):
        ts.to_format("DB")

def test_plot_invalid_element_name():
    f = np.array([1e9])
    s = np.array([[[1+1j]]])
    ts = TouchstoneFile.from_data(f, s)
    with pytest.raises(ValueError):
        ts.plot_s_db("Q22")
