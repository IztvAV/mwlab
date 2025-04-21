import pytest
import torch
import numpy as np
import pathlib
import skrf as rf

from mwlab import TouchstoneData
from mwlab import TouchstoneDataset
from mwlab.transforms import x_transforms, s_transforms
from mwlab.transforms import TComposite


# ---------- Фикстуры -------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset_dir():
    """Путь к директории с тестовыми *.sNp файлами."""
    return pathlib.Path(__file__).parent.parent / "Data" / "Filter12"


@pytest.fixture(scope="module")
def sample_info(dataset_dir):
    file = next(dataset_dir.glob("*.s?p"))
    ts = TouchstoneData.load(file)

    # Параметры
    param_keys = list(ts.params.keys())
    selected_keys = param_keys[:3] if len(param_keys) >= 3 else param_keys

    # Частоты в Гц
    fmin_hz, fmax_hz = ts.network.f[0], ts.network.f[-1]
    f_crop_min = fmin_hz + 0.25 * (fmax_hz - fmin_hz)
    f_crop_max = fmax_hz - 0.25 * (fmax_hz - fmin_hz)

    # Единица частоты (для S_Crop)
    unit = ts.network.frequency.unit  # например, 'GHz'
    multiplier = rf.Frequency.multiplier_dict[unit.lower()]
    f_crop_min_scaled = f_crop_min / multiplier
    f_crop_max_scaled = f_crop_max / multiplier

    # Создаём объект Frequency для S_Resample (в Гц)
    n_points = 50
    f_interp = rf.Frequency.from_f(f=np.linspace(f_crop_min, f_crop_max, n_points), unit='Hz')

    return {
        "selected_keys": selected_keys,
        "f_crop_min": f_crop_min_scaled,  # в native unit для crop
        "f_crop_max": f_crop_max_scaled,
        "f_interp": f_interp,
        "ports": ts.network.number_of_ports,
        "unit": unit
    }


# ---------- Базовые тесты -------------------------------------------------------

def test_dataset_len_and_paths(dataset_dir):
    ds = TouchstoneDataset(dataset_dir)
    assert len(ds) > 0
    for path in ds.paths:
        assert path.suffix.lower() in {".s1p", ".s2p", ".s3p", ".s4p"}


def test_dataset_getitem_raw(dataset_dir):
    ds = TouchstoneDataset(dataset_dir)
    x, s = ds[0]
    assert isinstance(x, dict)
    assert isinstance(s, rf.Network)
    assert s.s.shape[0] > 0  # F
    assert s.s.ndim == 3     # (F, P, P)


# ---------- Трансформы (на основе реальных данных) ------------------------------
def test_dataset_with_transforms(dataset_dir, sample_info):
    x_tf = TComposite([
        x_transforms.X_SelectKeys(sample_info["selected_keys"]),
    ])
    s_tf = TComposite([
        s_transforms.S_Crop(f_start=sample_info["f_crop_min"],
                            f_stop=sample_info["f_crop_max"]),
        s_transforms.S_Resample(freq_or_n=sample_info["f_interp"]),
    ])

    ds = TouchstoneDataset(dataset_dir,
                           x_keys=sample_info["selected_keys"],
                           x_tf=x_tf,
                           s_tf=s_tf)

    x, s = ds[0]

    # Проверка параметров (X)
    assert isinstance(x, dict), "x должен быть словарём после X_SelectKeys"
    assert set(x.keys()) == set(sample_info["selected_keys"]), "Неверный набор ключей"
    assert all(isinstance(v, float) or np.isnan(v) for v in x.values()), "Значения должны быть числами или NaN"

    # Проверка S‑матрицы (rf.Network)
    assert isinstance(s, rf.Network)
    assert s.s.ndim == 3  # (F, P, P)
    assert s.s.shape[0] == 50                    # F
    assert s.s.shape[1] == sample_info["ports"]  # P
    assert s.s.shape[2] == sample_info["ports"]  # P

# ---------- Обработка отсутствующих параметров -----------------------------------

def test_missing_params_are_nan(dataset_dir):
    ds = TouchstoneDataset(dataset_dir, x_keys=["nonexistent_parameter"])
    x, _ = ds[0]
    assert isinstance(x, dict)
    assert "nonexistent_parameter" in x
    assert np.isnan(x["nonexistent_parameter"])


# ---------- Параметры без трансформов (fallback-поведение) -----------------------

def test_fallback_behavior(dataset_dir):
    ds = TouchstoneDataset(dataset_dir)
    x, s = ds[0]

    # x: словарь параметров
    assert isinstance(x, dict)
    assert all(isinstance(k, str) for k in x)
    assert all(isinstance(v, float | int | str | type(np.nan)) for v in x.values())

    # s: skrf.Network
    assert isinstance(s, rf.Network)
    assert s.s.ndim == 3  # (F, P, P)

