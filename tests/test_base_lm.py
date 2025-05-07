# tests/test_base_lm.py
import numpy as np
import torch
from torch import nn
import skrf as rf

from mwlab.lightning.base_lm import BaseLModule
from mwlab.nn.scalers import StdScaler, MinMaxScaler
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.io.touchstone import TouchstoneData

# -------------------------  helpers ---------------------------
class Dummy(nn.Module):
    """Linear (B, D) -> (B, C)."""
    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c, bias=False)
        torch.manual_seed(0)
        nn.init.constant_(self.fc.weight, 0.1)

    def forward(self, x):  # type: ignore[override]
        return self.fc(x)


class DummyCF(nn.Module):
    """Linear (B, D) -> (B, C, F=1)."""
    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c, bias=False)
        torch.manual_seed(0)
        nn.init.constant_(self.fc.weight, 0.1)

    def forward(self, x):  # type: ignore[override]
        return self.fc(x).unsqueeze(-1)          # (B,C,1)


def mini_codec() -> TouchstoneCodec:
    return TouchstoneCodec(
        x_keys=["a"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=np.array([1.0e9]),              # F = 1
    )


def sample_batch(codec: TouchstoneCodec):
    net = rf.Network(f=[1e9], s=[[[0.1 + 0.2j]]], f_unit="Hz")
    ts = TouchstoneData(net, params={"a": 0.5})
    x, y, meta = codec.encode(ts)
    return x.unsqueeze(0), y.unsqueeze(0), meta        # (1,D), (1,C,F)


# 1. базовые тесты (без codec) --------------------------------------------
def test_forward_with_scaler():
    module = BaseLModule(model=Dummy(4, 2),
                         scaler_in=StdScaler(dim=0).fit(torch.randn(10, 4)))
    assert module(torch.randn(3, 4)).shape == (3, 2)


def test_training_step_with_scaler_out():
    module = BaseLModule(model=Dummy(4, 2),
                         scaler_out=MinMaxScaler(dim=0).fit(torch.randn(10, 2)))
    loss = module.training_step((torch.randn(5, 4), torch.randn(5, 2)), 0)
    assert loss.requires_grad and loss.ndim == 0


def test_predict_step_inverse_scaler():
    module = BaseLModule(model=Dummy(4, 2),
                         scaler_out=MinMaxScaler(dim=0).fit(torch.randn(10, 2)))
    p = module.predict_step((torch.randn(5, 4), torch.zeros(5, 2)), 0)
    assert p.shape == (5, 2)


def test_configure_optimizers():
    mod = BaseLModule(model=Dummy(4, 2),
                      optimizer_cfg={"name": "Adam", "lr": 1e-3},
                      scheduler_cfg={"name": "StepLR", "step_size": 5})
    cfg = mod.configure_optimizers()
    assert {"optimizer", "lr_scheduler"} <= cfg.keys()


def test_state_dict_contains_scalers_and_codec():
    codec = mini_codec()
    mod = BaseLModule(model=Dummy(4, 2),
                      scaler_in=StdScaler(dim=0).fit(torch.randn(10, 4)),
                      scaler_out=MinMaxScaler(dim=0).fit(torch.randn(10, 2)),
                      codec=codec)
    new = BaseLModule(model=Dummy(4, 2))
    new.scaler_in = StdScaler(dim=0).fit(torch.randn(5, 4))
    new.scaler_out = MinMaxScaler(dim=0).fit(torch.randn(5, 2))
    new.load_state_dict(mod.state_dict(), strict=False)
    assert new.codec is not None and hasattr(new.scaler_in, "mean")


# 2. сценарии с codec --------------------------------------------
def test_predict_tensor_autodecode_false():
    codec = mini_codec()
    mod = BaseLModule(model=DummyCF(1, 2),
                      codec=codec, auto_decode=False)
    out = mod.predict_step(sample_batch(codec), 0)
    assert isinstance(out, torch.Tensor) and out.shape == (1, 2, 1)


def test_predict_touchstone_autodecode_true():
    codec = mini_codec()
    mod = BaseLModule(model=DummyCF(1, 2),
                      codec=codec, auto_decode=True)
    x, y, meta = sample_batch(codec)
    ts_out = mod.predict_step((x, y, [meta]), 0)
    assert isinstance(ts_out, list)
    assert isinstance(ts_out[0], TouchstoneData)
    s = ts_out[0].network.s[0, 0, 0]
    assert np.isclose(s.real, 0.05, atol=1e-6)
    assert np.isclose(s.imag, 0.05, atol=1e-6)

def test_predict_inverse_returns_dict():
    codec = mini_codec()
    mod = BaseLModule(model=Dummy(2, 1), codec=codec, swap_xy=True)
    _, y, meta = sample_batch(codec)
    out = mod.predict_step((y.squeeze(-1), torch.zeros(1, 1), meta), 0)
    assert isinstance(out, list) and isinstance(out[0], dict)
    assert "a" in out[0]
