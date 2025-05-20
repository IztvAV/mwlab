# tests/test_surrogates.py
"""
Проверяем слой mwlab.opt.surrogates:

1.  Registry / factory      → get_surrogate
2.  NNSurrogate.predict     → rf.Network (direct) / Dict (inverse)
3.  batch_predict           → List[…]
4.  save / load             → state переносится
5.  Поведение при swap_xy=True
"""

import numpy as np
import torch
import skrf as rf
import pytest

from mwlab.opt.design.space import DesignSpace, ContinuousVar
from mwlab.opt.surrogates import get_surrogate, NNSurrogate
from mwlab.lightning.base_lm import BaseLModule
from mwlab.codecs.touchstone_codec import TouchstoneCodec


# ───────────────────────── helpers ─────────────────────────
class TinyNet(torch.nn.Module):
    """Простейшая линейная сеть (D=1) → (C=2,F=1)."""
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 2, bias=False)
        torch.nn.init.constant_(self.fc.weight, 0.1)

    def forward(self, x):               # type: ignore[override]
        return self.fc(x).unsqueeze(-1)  # (B,2,1)


def make_codec() -> TouchstoneCodec:
    return TouchstoneCodec(
        x_keys=["a"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=np.array([1.0e9]),
    )


def trained_module(swap_xy: bool = False) -> BaseLModule:
    codec = make_codec()
    lm = BaseLModule(model=TinyNet(), codec=codec, swap_xy=swap_xy)
    # вес уже инициализирован константой, обучение не нужно
    return lm.eval()


# ───────────────────────── tests ─────────────────────────
def test_registry_returns_nn():
    sur = get_surrogate("nn", pl_module=trained_module(), design_space=None)
    assert isinstance(sur, NNSurrogate)



def test_predict_returns_network():
    sur = NNSurrogate(pl_module=trained_module(), design_space=None)
    net = sur.predict({"a": 0.5})
    assert isinstance(net, rf.Network)
    assert net.nports == 1
    assert len(net.f) == 1


def test_batch_predict_length():
    sur = NNSurrogate(pl_module=trained_module(), design_space=None)
    X = [{"a": 0.1 * k} for k in range(5)]
    out = sur.batch_predict(X)
    assert isinstance(out, list) and len(out) == 5


def test_swap_xy_rejected():
    with pytest.raises(ValueError):
        NNSurrogate(pl_module=trained_module(swap_xy=True))


def test_factory_with_designspace():
    ds = DesignSpace({"a": ContinuousVar(-1, 1)})
    sur = get_surrogate("pytorch", pl_module=trained_module(), design_space=ds)
    assert sur.design_space is ds


def test_batch_predict_with_uncertainty():
    sur = NNSurrogate(pl_module=trained_module(), design_space=None)
    X = [{"a": 0.1 * k} for k in range(3)]
    with pytest.raises(NotImplementedError):
        sur.batch_predict(X, return_std=True)


def test_factory_keyerror():
    with pytest.raises(KeyError):
        get_surrogate("unknown_model")


def test_gp_not_implemented():
    try:
        from mwlab.opt.surrogates import GPSurrogate
        with pytest.raises(NotImplementedError):
            GPSurrogate()
    except ImportError:
        pass  # ок, если зависимость не установлена