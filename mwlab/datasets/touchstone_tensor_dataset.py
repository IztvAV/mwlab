# mwlab/datasets/touchstone_tensor_dataset.py
"""
`TouchstoneTensorDataset` — тонкая надстройка над `TouchstoneDataset`,
которая возвращает из `__getitem__` пару PyTorch‑тензоров `(X, Y)` (и
при желании `meta`) с помощью заданного `TouchstoneCodec`.

Ключевые возможности
--------------------
* **LRU‑кэш** готовых примеров (наследуется от `_CachedDataset`);
* опциональная **перестановка признаков и целей** (`swap_xy`);
* опциональный возврат словаря **meta**
 &nbsp; &nbsp;(`{"params": …,"unit": …, "orig_path": …}`);
* автоматически принимает **любой источник** данных:
  каталог *.sNp*, монолитный HDF5, LMDB — главное, чтобы существовал
  `StorageBackend`.

Пример использования
--------------------
```python
# 1) создаем «сырой» датасет с трансформами
src      = "Data/Filter12"                        # каталог или .h5‑файл
base_ds  = TouchstoneDataset(
              src,
              s_tf=S_Crop(1e9, 10e9) >> S_Resample(256),   # цепочка S‑TF
              x_keys=["w", "gap"],
          )

# 2) строим Codec автоматически по содержимому base_ds
codec = TouchstoneCodec.from_dataset(base_ds, components=("real", "imag"))

# 3) Tensor‑датасет для PyTorch‑модели
tensor_ds = TouchstoneTensorDataset(
                source=src,          # можно передать тот же путь
                codec=codec,
                swap_xy=False,
                return_meta=True,
                cache_size=512,      # LRU на 512 примеров
            )

x_t, y_t, meta = tensor_ds[0]       # x_t: (Dx,), y_t: (Cy, F)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Any, Tuple

from torch.utils.data import get_worker_info
import torch

from mwlab.io.backends import StorageBackend, FileBackend
from mwlab.io.touchstone import TouchstoneData
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.datasets._cached_dataset import _CachedDataset


# ────────────────────────────────────────────────────────────────────────────
#                           TouchstoneTensorDataset
# ────────────────────────────────────────────────────────────────────────────
class TouchstoneTensorDataset(_CachedDataset):
    r"""
    Датасет, который возвращает **тензорные** представления Touchstone‑файлов.

    Parameters
    ----------
    source : str | Path | StorageBackend
        ─ каталог с *.sNp* **или** путь к бинарному файлу (HDF5, …)
        ─ **или** уже созданный backend.
    codec : TouchstoneCodec
        Кодек, отвечающий за преобразование TouchstoneData ↔ тензоры.
    swap_xy : bool, default=False
        Переставлять ли X и Y местами в `__getitem__`.
    return_meta : bool, default=False
        Возвращать ли `meta` (dict) третьим элементом.
    cache_size : int | None, default=0
        Размер LRU‑кеша; `None` – неограниченный, `0` – кеш выключен.
    base_kwargs : dict, optional
        Доп. аргументы для базового `TouchstoneDataset`
        (например `{"s_tf": S_Crop(...), "x_keys": [...]}`).
    """

    def __init__(
        self,
        source: Union[str, Path, StorageBackend],
        codec: TouchstoneCodec,
        *,
        swap_xy: bool = False,
        return_meta: bool = False,
        cache_size: Optional[int] = 0,
        base_kwargs: Optional[dict] = None,
    ):
        super().__init__(cache_size=cache_size, swap_xy=swap_xy, return_meta=return_meta)

        # --- инициализируем «сырой» TouchstoneDataset -------------------
        self._base = TouchstoneDataset(source, **(base_kwargs or {}))
        self.codec = codec

    # -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._base)

    # _CachedDataset будет дёргать именно этот метод
    def _get_item_raw(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        1. Берем элемент из базового датасета (`dict`, `rf.Network`).
        2. Формируем из этих данных временный `TouchstoneData`.
        3. Прогоняем через `codec.encode`.
        """

        x_part, net_part = self._base[idx]              # x_tf / s_tf уже применены

        # x_part – dict параметров (возможно подсет); безопасно дополняем NaN'ами
        ts = TouchstoneData(net_part, params=x_part)

        return self.codec.encode(ts)

    # -------------------------------------------------------------------
    def __getitem__(self, idx: int):
        # IDE‑friendly подпись, но делегируем всю логику в _CachedDataset
        return super().__getitem__(idx)

    # -------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        cache_state = (
            "disabled"
            if not self._cache_enabled
            else f"{len(self._cache)}/{'∞' if self._cache_limit < 0 else self._cache_limit}"
        )
        return (
            f"{self.__class__.__name__}("
            f"samples={len(self)}, "
            f"swap_xy={self._swap}, return_meta={self._retm}, cache={cache_state}, "
            f"codec={self.codec})"
        )

    __str__ = __repr__




