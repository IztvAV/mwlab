# mwlab/datasets/_cached_dataset.py
"""
Mixin: LRU‑кэширование + поддержка перестановки X↔Y + возврат meta.

* Можно использовать для базового слоя всех кастомных датасетов.
"""

import collections
from torch.utils.data import Dataset, get_worker_info
from typing import Optional, Tuple, Any


class _CachedDataset(Dataset):
    """
    Базовый класс с поддержкой:
    - LRU-кэширования элементов по индексу;
    - Перестановки X ↔︎ Y при выдаче;
    - Опционального возврата дополнительной информации (meta).

    Дочерние классы обязаны реализовать метод:
        _get_item_raw(idx) -> (x, y, meta)
    """

    def __init__(
        self,
        cache_size: Optional[int] = 0,
        swap_xy: bool = False,
        return_meta: bool = False,
    ):
        """
        Параметры
        ----------
        cache_size : int | None
            Максимальный размер кэша. 0 — без кэширования. None — неограниченный кэш.
        swap_xy : bool
            Менять ли местами X и Y при возврате __getitem__().
        return_meta : bool
            Возвращать ли третий элемент meta (иначе возвращается только (x, y)).
        """
        super().__init__()
        self._cache_enabled = bool(cache_size) or cache_size is None
        self._cache_limit = -1 if cache_size is None else int(cache_size)
        self._cache: "collections.OrderedDict[int, Tuple[Any, Any, Any]]" = collections.OrderedDict()
        self._swap = bool(swap_xy)
        self._retm = bool(return_meta)

    def _get_item_raw(self, idx: int) -> Tuple[Any, Any, Any]:
        """Метод, который должен быть переопределён в дочерних классах."""
        raise NotImplementedError("Дочерний класс обязан реализовать _get_item_raw")

    def __getitem__(self, idx: int):
        # Отключаем кэш в режиме многопоточности
        if get_worker_info() is not None:
            self._cache_enabled = False
            self._cache.clear()

        if self._cache_enabled and idx in self._cache:
            self._cache.move_to_end(idx)
            x, y, meta = self._cache[idx]
        else:
            x, y, meta = self._get_item_raw(idx)
            if self._cache_enabled:
                self._cache[idx] = (x, y, meta)
                self._cache.move_to_end(idx)
                if self._cache_limit >= 0 and len(self._cache) > self._cache_limit:
                    self._cache.popitem(last=False)

        if self._swap:
            x, y = y, x

        return (x, y, meta) if self._retm else (x, y)
