# mwlab/tests/test_scalers.py

import torch
import pytest

from mwlab.nn.scalers import StdScaler, MinMaxScaler, RobustScaler

# ───────────────────────── StdScaler ──────────────────────────
def test_std_scaler_inverse():
    data = torch.randn(100, 20)
    scaler = StdScaler(dim=0).fit(data)
    scaled = scaler(data)
    restored = scaler.inverse(scaled)
    assert torch.allclose(restored, data, atol=1e-5)


def test_std_scaler_mean_std():
    data = torch.randn(50, 10)
    scaler = StdScaler(dim=0).fit(data)
    assert scaler.mean.shape == (1, 10)
    assert scaler.std.min() > 0


@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_std_scaler_dims(dim):
    data = torch.randn(32, 16)
    scaler = StdScaler(dim=dim).fit(data)
    scaled = scaler(data)
    restored = scaler.inverse(scaled)
    assert torch.allclose(restored, data, atol=1e-5)


# ───────────────────────── MinMaxScaler ───────────────────────
def test_minmax_scaler_forward_inverse_default_range():
    data = torch.randn(100, 5)
    scaler = MinMaxScaler(dim=0).fit(data)
    scaled = scaler(data)
    restored = scaler.inverse(scaled)
    assert torch.allclose(restored, data, atol=1e-5)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


def test_minmax_scaler_custom_range():
    data = torch.randn(100, 5)
    scaler = MinMaxScaler(dim=0, feature_range=(-1, 1)).fit(data)
    scaled = scaler(data)
    restored = scaler.inverse(scaled)
    assert torch.allclose(restored, data, atol=1e-5)
    assert scaled.min() >= -1.0
    assert scaled.max() <= 1.0


def test_minmax_scaler_invalid_range():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 1))
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(2, 1))
    with pytest.raises(TypeError):
        MinMaxScaler(feature_range=("a", "b"))


# ───────────────────────── RobustScaler ───────────────────────
def test_robust_scaler_inverse():
    # ввод с сильными выбросами
    data = torch.randn(200, 10)
    data[::10] *= 50.0           # редкие экстремальные точки
    scaler = RobustScaler(dim=0).fit(data)
    scaled = scaler(data)
    restored = scaler.inverse(scaled)
    # допускаем чуть более свободный атол из-за float-арифметики
    assert torch.allclose(restored, data, atol=1e-5)


def test_robust_scaler_center_scale():
    data = torch.randn(80, 12)
    scaler = RobustScaler(dim=0, quantile_range=(15, 85)).fit(data)
    assert scaler.center.shape == (1, 12)
    assert torch.all(scaler.scale > 0)


@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_robust_scaler_dims(dim):
    data = torch.randn(64, 8)
    data[::7] += 7.0             # выбросы
    scaler = RobustScaler(dim=dim).fit(data)
    scaled = scaler(data)
    restored = scaler.inverse(scaled)
    assert torch.allclose(restored, data, atol=1e-5)


def test_robust_scaler_custom_quantiles():
    data = torch.randn(50, 4)
    scaler = RobustScaler(dim=0, quantile_range=(5, 95)).fit(data)
    scaled = scaler(data)
    # убедимся, что медиана стала ~0
    med = torch.median(scaled, dim=0).values
    assert torch.allclose(med, torch.zeros_like(med), atol=1e-6)


def test_robust_scaler_invalid_quantiles():
    with pytest.raises(ValueError):
        RobustScaler(quantile_range=(50, 50))
    with pytest.raises(ValueError):
        RobustScaler(quantile_range=(90, 10))
    with pytest.raises(TypeError):
        RobustScaler(quantile_range=("low", "high"))

