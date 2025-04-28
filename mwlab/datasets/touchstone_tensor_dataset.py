# mwlab/datasets/touchstone_tensor_dataset.py

from pathlib import Path
from typing import Optional, Union

from torch.utils.data import Dataset
from mwlab.io.touchstone import TouchstoneData
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from ._cached_dataset import _CachedDataset
from .touchstone_dataset import TouchstoneDataset

# ────────────────────────────────────────────────────────────────────────────
#                           TouchstoneTensorDataset
# ────────────────────────────────────────────────────────────────────────────
class TouchstoneTensorDataset(_CachedDataset):
    """
    Датасет для конвертации TouchstoneData → (X, Y, meta) через заданный TouchstoneCodec.

    Поддержка:
    - кэширования элементов;
    - перестановки X ↔ Y;
    - возврата метаинформации.

    Параметры
    ----------
    root : str | Path
        Путь к директории с Touchstone-файлами.
    codec : TouchstoneCodec
        Кодек для преобразования данных.
    swap_xy : bool
        Переставлять ли X и Y местами при выдаче.
    return_meta : bool
        Возвращать ли meta вместе с (x, y).
    cache_size : int | None
        Размер кэша. 0 — без кэширования. None — неограниченный кэш.
    base_kwargs : dict, optional
        Дополнительные параметры для базового TouchstoneDataset.
    """

    def __init__(
        self,
        root: Union[str, Path],
        codec: TouchstoneCodec,
        *,
        swap_xy: bool = False,
        return_meta: bool = False,
        cache_size: Optional[int] = 0,
        base_kwargs: Optional[dict] = None,
    ):
        super().__init__(cache_size=cache_size, swap_xy=swap_xy, return_meta=return_meta)

        self.codec = codec
        self._base = TouchstoneDataset(root, **(base_kwargs or {}))

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        """Нужен для подсказок IDE. Делегируется в _CachedDataset."""
        return super().__getitem__(idx)

    # основной «сырой» вызов
    def _get_item_raw(self, idx: int):
        ts = TouchstoneData.load(self._base.paths[idx])
        return self.codec.encode(ts)

    # для вывода основной информации о датасете -> print(dataset)
    def __repr__(self) -> str:
        cache = (
            "disabled"
            if not self._cache_enabled
            else f"{len(self._cache)}/{'∞' if self._cache_limit < 0 else self._cache_limit}"
        )
        return (
            f"{self.__class__.__name__}(samples={len(self)}, "
            f"swap_xy={self._swap}, return_meta={self._retm}, cache={cache})"
        )

    __str__ = __repr__


