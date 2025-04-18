# transforms/x_transforms.py
import torch, numpy as np

class X_SelectKeys:
    def __init__(self, keys): self.keys = keys
    def __call__(self, xdict):
        return np.array([xdict[k] for k in self.keys], dtype=np.float32)

class X_ToTensor:
    def __call__(self, arr): return torch.from_numpy(arr)
