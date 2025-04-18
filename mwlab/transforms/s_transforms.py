# transforms/s_transforms.py
import torch, numpy as np

class S_Crop:
    """Вырезает диапазон [fmin, fmax] (в GHz, MHz, …)."""
    def __init__(self, fmin, fmax, unit="GHz"):
        self.fmin, self.fmax, self.unit = fmin, fmax, unit
    def __call__(self, net):
        return net[self.fmin:self.fmax, self.unit].s          # ndarray (F', P, P)

class S_Resize:
    """Интерполирует S‑матрицу на новую частотную сетку."""
    def __init__(self, new_freqs, unit="GHz", kind="linear"):
        self.new_freqs, self.unit, self.kind = np.asarray(new_freqs), unit, kind
    def __call__(self, net):
        return net.interpolate(self.new_freqs, unit=self.unit, kind=self.kind).s

class S_ToTensor:
    def __call__(self, arr):
        # arr — ndarray complex64 (F, P, P) → 2×F×P×P (Re/Im) float32
        re, im = arr.real.astype(np.float32), arr.imag.astype(np.float32)
        return torch.from_numpy(np.stack([re, im], axis=0))
