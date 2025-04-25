# touchstone_tensor_dataset.py
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, List, Tuple
import collections

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

import skrf as rf
from mwlab.datasets import TouchstoneDataset  # «сырой» датасет

# --------------------------------------------------------------------------- #
#                                helpers                                      #
# --------------------------------------------------------------------------- #


def _parse_channel(tag: str) -> Tuple[int, int, str]:
    """Парсинг строки вида ``Sij.part`` → (i‑1, j‑1, part), где part ∈ {real,imag,db,mag,deg}.
       Вызывает ``ValueError`` при некорректном формате или неизвестной компоненте.
    """
    try:
        s_idx, comp = tag.split(".")
        if not (s_idx.startswith("S") and len(s_idx) == 3):
            raise ValueError
        i, j = int(s_idx[1]) - 1, int(s_idx[2]) - 1
        comp = comp.lower()
    except Exception:
        raise ValueError(f"Неверный формат канала: {tag!r}") from None
    if comp not in {"real", "imag", "db", "mag", "deg"}:
        raise ValueError(f"Неизвестная компонента: {comp!r}")
    return i, j, comp


def _convert_component(s: np.ndarray, comp: str, *, eps: float = 1e-12) -> np.ndarray:
    """Преобразует одномерный комплексный вектор *s* в заданную скалярную компоненту."""
    comp = comp.lower()
    if comp == "real":
        return s.real
    if comp == "imag":
        return s.imag
    if comp == "db":
        return 20 * np.log10(np.abs(s) + eps)
    if comp == "mag":
        return np.abs(s)
    if comp == "deg":
        return np.unwrap(np.angle(s), axis=-1) * 180 / np.pi
    raise ValueError(f"Неизвестная компонента: {comp!r}")


# --------------------------------------------------------------------------- #
#                        TouchstoneTensorDataset                              #
# --------------------------------------------------------------------------- #

class TouchstoneTensorDataset(Dataset):
    r"""PyTorch‑совместимый датасет, преобразующий ``TouchstoneDataset``.

    * **X**  – вектор: ``torch.float32``  «shape» = ``(D,)``
    * **Y**  – тензор: ``torch.float32``  «shape» = ``(C, F)``

    **Настройка каналов Y**
    -----------------------
    *Если* указан параметр ``y_channels`` – используется он (список тегов
    ``Sij.part``). В противном случае каналы формируются автоматически по
    всем парам портов и выбранным компонентам ``components``.
    """

    # ----------------------------- init ------------------------------------ #

    def __init__(
        self,
        root: str | Path,
        *,
        x_tf: Optional[Callable[[Mapping[str, float]], Mapping[str, float]]] = None,
        s_tf: Optional[Callable[[rf.Network], rf.Network]] = None,
        y_channels: Optional[Sequence[str]] = None,
        components: Sequence[str] = ("real", "imag"),
        cache_size: int | None = 0,  # 0 → без кэша, None → не ограничивать
        eps_db: float = 1e-12,
    ) -> None:
        super().__init__()

        # «Сырой» датасет --------------------------------------------------- #
        self._base = TouchstoneDataset(root, x_tf=x_tf, s_tf=s_tf)
        if len(self._base) == 0:
            raise ValueError("Пустой TouchstoneDataset – нечего обрабатывать")

        # ---- X‑признаки: объединение ключей ------------------------------ #
        keys_union: set[str] = set()
        first_net: rf.Network | None = None
        for idx in range(len(self._base)):
            x_i, net_i = self._base[idx]
            keys_union.update(x_i.keys())
            if first_net is None:
                first_net = net_i
        if not keys_union:
            raise ValueError("Не удалось сформировать список X‑признаков – пусто")
        self.x_keys: List[str] = sorted(keys_union)
        self._x_dim: int = len(self.x_keys)

        # ---- Network‑зависимые параметры ---------------------------------- #
        assert first_net is not None
        self._n_ports: int = first_net.number_of_ports
        self._freq_len: int = first_net.s.shape[0]
        self._eps_db: float = eps_db

        # ---- Y‑каналы ----------------------------------------------------- #
        if y_channels is not None:
            # Пользовательский список – валидируем
            self.y_channels: List[str] = [str(ch) for ch in y_channels]
            for tag in self.y_channels:
                _parse_channel(tag)  # валидируем формат и компоненту
            # выведем список компонентов, если понадобится _y_from_network()
            self._components = list({_parse_channel(t)[2] for t in self.y_channels})
        else:
            # Авто‑генерация по components
            self._components: List[str] = [c.lower() for c in components]
            self.y_channels = [
                f"S{i + 1}{j + 1}.{comp}"
                for comp in self._components
                for i in range(self._n_ports)
                for j in range(self._n_ports)
            ]
        self._num_channels: int = len(self.y_channels)

        # ---- LRU‑кэш ------------------------------------------------------ #
        self._cache_enabled: bool = bool(cache_size) or cache_size is None
        self._cache_limit: int = -1 if cache_size is None else int(cache_size)
        self._cache: "collections.OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]" = (
            collections.OrderedDict()
        )

    # --------------------------- privates ---------------------------------- #

    def _x_to_tensor(self, x_dict: Mapping[str, float]) -> torch.Tensor:
        vals: List[float] = []
        for k in self.x_keys:
            v = x_dict.get(k, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                raise ValueError(f"Parameter '{k}' is undefined or NaN")
            vals.append(float(v))
        return torch.tensor(vals, dtype=torch.float32)

    def _y_from_network(self, net: rf.Network) -> torch.Tensor:
        """Формирует Y‑тензор согласно ``self.y_channels``."""
        s = net.s  # (F, P, P)
        chans: List[np.ndarray] = []
        for tag in self.y_channels:
            i, j, comp = _parse_channel(tag)
            chans.append(_convert_component(s[:, i, j], comp, eps=self._eps_db))
        y_np = np.stack(chans, axis=0)  # (C, F)
        return torch.from_numpy(y_np).float()

    # ----------------------- Dataset interface ----------------------------- #

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        # Отключаем кэш в воркере (если включен - случай многопотоковой загрузки num_workers > 0)
        worker_info = get_worker_info()
        if worker_info is not None and self._cache_enabled:
            # Мы внутри воркера — кэш отключается
            self._cache_enabled = False
            self._cache.clear()  # опционально: сбрасываем кэш, если был

        if self._cache_enabled and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        x_dict, net = self._base[idx]
        x_t = self._x_to_tensor(x_dict)
        y_t = self._y_from_network(net)

        if self._cache_enabled:
            self._cache[idx] = (x_t, y_t)
            self._cache.move_to_end(idx)
            if self._cache_limit >= 0 and len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return x_t, y_t

    # ----------------------- repr / str ------------------------------------ #

    def __repr__(self) -> str:  # noqa: D401
        cache_state = (
            "disabled"
            if not self._cache_enabled
            else f"{len(self._cache)}/{'∞' if self._cache_limit < 0 else self._cache_limit}"
        )
        return (
            f"{self.__class__.__name__}(samples={len(self)}, "
            f"x_dim={self._x_dim}, "
            f"y_shape=({self._num_channels}, {self._freq_len}), "
            f"cache={cache_state})"
        )

    __str__ = __repr__

