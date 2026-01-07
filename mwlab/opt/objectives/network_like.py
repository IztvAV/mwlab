# mwlab/opt/objectives/network_like.py
"""
Протоколы "duck-typed" сетей и частотной оси для objectives.
Лёгкие view-контейнеры для частоты и S-параметров с ленивым доступом к NumPy.

Нужны, чтобы компоненты objectives могли принимать объекты, совместимые со
`skrf.Network`, без жёсткой зависимости от scikit-rf.
"""

from __future__ import annotations

import importlib.util
from typing import Protocol, Optional, Union

import numpy as np

_TORCH_SPEC = importlib.util.find_spec("torch")
if _TORCH_SPEC is not None:
    import torch
else:
    torch = None

ArrayLike = Union[np.ndarray, "torch.Tensor"]

# =============================================================================
# Протоколы "duck-typed" сетей и частотной оси для objectives.
# =============================================================================

class FrequencyLike(Protocol):
    """Минимальный интерфейс частотной оси, совместимый с rf.Frequency."""

    f: np.ndarray


class NetworkLike(Protocol):
    """Минимальный интерфейс сети, совместимый с rf.Network."""

    frequency: FrequencyLike
    s: np.ndarray
    s_mag: np.ndarray
    s_db: np.ndarray


# =============================================================================
# Лёгкие view-контейнеры
# =============================================================================
def _as_numpy(arr: ArrayLike, *, name: str) -> np.ndarray:
    """Преобразовать входной массив в NumPy.

    Если вход — CPU-contiguous torch.Tensor, возвращается zero-copy view.
    Иначе возвращается NumPy-массив, полученный из копии CPU-contiguous тензора.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if torch is not None and isinstance(arr, torch.Tensor):
        if arr.device.type == "cpu" and arr.is_contiguous():
            return arr.numpy()
        t = arr
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        return t.numpy()
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor")


class FrequencyView:
    """Ленивый view частот с опциональным кешированием.

    Примечания по zero-copy:
        * CPU-contiguous torch.Tensor конвертируется в NumPy без копии.
        * Неконтинуальные или CUDA-тензоры копируются в CPU-contiguous буфер.
    """

    def __init__(self, f_src: ArrayLike, *, cache: bool = True) -> None:
        self.f_src = f_src
        self.cache = cache
        self._f_np: Optional[np.ndarray] = None

    @property
    def f(self) -> np.ndarray:
        if self.cache:
            if self._f_np is None:
                self._f_np = _as_numpy(self.f_src, name="f")
            return self._f_np
        return _as_numpy(self.f_src, name="f")


class SParamView:
    """Ленивый view для S-параметров и производных mag/db массивов.

    Примечания по zero-copy:
        * CPU-contiguous torch.Tensor конвертируется в NumPy без копии.
        * Неконтинуальные или CUDA-тензоры копируются в CPU-contiguous буфер.
    """

    def __init__(self, s: ArrayLike, freq: ArrayLike, *, cache: bool = True) -> None:
        self._s_src = s
        self._f_src = freq
        self.cache = cache
        self._s_np: Optional[np.ndarray] = None
        self._f_np: Optional[np.ndarray] = None
        self._s_mag_np: Optional[np.ndarray] = None
        self._s_db_np: Optional[np.ndarray] = None

    @property
    def freq(self) -> np.ndarray:
        if self.cache:
            if self._f_np is None:
                self._f_np = _as_numpy(self._f_src, name="freq")
            return self._f_np
        return _as_numpy(self._f_src, name="freq")

    @property
    def s(self) -> np.ndarray:
        if self.cache:
            if self._s_np is None:
                self._s_np = _as_numpy(self._s_src, name="s")
            return self._s_np
        return _as_numpy(self._s_src, name="s")

    @property
    def s_mag(self) -> np.ndarray:
        if self.cache:
            if self._s_mag_np is None:
                self._s_mag_np = np.abs(self.s)
            return self._s_mag_np
        return np.abs(self.s)

    @property
    def s_db(self) -> np.ndarray:
        if self.cache:
            if self._s_db_np is None:
                self._s_db_np = 20 * np.log10(self.s_mag)
            return self._s_db_np
        return 20 * np.log10(self.s_mag)