# mwlab/tests/test_scalers.py

import torch
import pytest
from mwlab.nn.scalers import StdScaler, MinMaxScaler


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
