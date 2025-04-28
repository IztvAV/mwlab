# mwlab/codecs/touchstone_codec.py

"""
Кодек TouchstoneData ↔ (X, Y, meta).

* encode()  — преобразование объекта TouchstoneData в тензоры + метаинформацию;
* decode()  — восстановление TouchstoneData по предсказанию модели;
* dumps()/loads() — сериализация для хранения в checkpoint'ах.
"""

import math
import pickle
import numpy as np
import torch
import skrf as rf
from typing import Sequence, Tuple, List, Dict, Optional, Any, Mapping

from mwlab.io.touchstone import TouchstoneData


# ─────────────────────────────────────────────────────────────────────────────
#                                 TouchstoneCodec
# ─────────────────────────────────────────────────────────────────────────────

class TouchstoneCodec:
    """
    Универсальный кодек TouchstoneData ↔︎ (x_t, y_t, meta).

    Публичный API:
    ---------------
    encode(ts: TouchstoneData) → (x_t, y_t, meta)
    decode(y_t, meta)          → TouchstoneData
    dumps() / loads()          → (де)сериализация через pickle
    """

    VERSION: int = 1  # Версия сериализации (на будущее)

    # ────────────────────────────────────────────────────────────────────────
    # init
    # ────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        x_keys: Sequence[str],
        y_channels: Sequence[str],
        freq_hz: np.ndarray,
        eps_db: float = 1e-12,
        force_resample: bool = True,
        nan_fill: complex | float = np.nan + 1j * np.nan,
    ):
        self.x_keys: List[str] = list(x_keys)
        self.y_channels: List[str] = list(y_channels)
        self.freq_hz: np.ndarray = np.asarray(freq_hz, dtype=float)
        self.eps_db = float(eps_db)
        self.force_resample = bool(force_resample)
        self.nan_fill = nan_fill

        # Портность сети (нужна для decode)
        pairs = [self._parse_channel(tag)[:2] for tag in self.y_channels]
        self.n_ports = int(max(max(i, j) for i, j in pairs) + 1)

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────────────

    def encode(
        self, ts: TouchstoneData
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Преобразует TouchstoneData → (x_t, y_t, meta).

        Возвращает
        ----------
        x_t : torch.Tensor
            Вектор признаков.
        y_t : torch.Tensor
            Тензор выходных данных (C, F).
        meta : dict
            Сопутствующая информация.
        """
        # --- X ---
        x_vec = torch.tensor([ts.params[k] for k in self.x_keys], dtype=torch.float32)

        # --- Y ---
        net = ts.network
        if self.force_resample and not np.allclose(net.f, self.freq_hz):
            net = net.copy()
            net.resample(self.freq_hz, inplace=True)

        chans = [
            self._convert(net.s[:, i, j], part)
            for (i, j, part) in map(self._parse_channel, self.y_channels)
        ]
        y_arr = np.stack(chans, axis=0)  # (C, F)
        y_t = torch.from_numpy(y_arr).float()

        # --- Meta ---
        meta = {
            "params": ts.params,
            "orig_path": str(ts.path) if ts.path else None,
        }
        return x_vec, y_t, meta

    @torch.no_grad()
    def decode(
        self, y_pred: torch.Tensor, meta: Optional[Mapping[str, Any]] = None
    ) -> TouchstoneData:
        """
        Восстанавливает TouchstoneData по предсказанию модели.

        Параметры
        ----------
        y_pred : Tensor
            Предсказание модели (C,F) или (B,C,F).
        meta : dict, optional
            Метаинформация, полученная при encode().
        """
        # --- нормализуем размерность ---
        if y_pred.dim() == 3:
            y_pred = y_pred[0]
        y_np = y_pred.cpu().float().numpy()

        # --- первый проход: собираем компоненты ---
        RE, IM, AMP, PH = {}, {}, {}, {}
        for k, tag in enumerate(self.y_channels):
            i, j, part = self._parse_channel(tag)
            kind, arr = self._reverse(y_np[k], part)
            if kind == "re":
                RE[(i, j)] = arr
            elif kind == "im":
                IM[(i, j)] = arr
            elif kind == "amp":
                AMP[(i, j)] = arr
            elif kind == "phase":
                PH[(i, j)] = arr

        # --- второй проход: восстанавливаем комплексную матрицу ---
        F, P = self.freq_hz.size, self.n_ports
        s = np.full((F, P, P), self.nan_fill, dtype=complex)

        for i in range(P):
            for j in range(P):
                key = (i, j)
                if key in RE and key in IM:
                    s[:, i, j] = RE[key] + 1j * IM[key]
                elif key in AMP and key in PH:
                    s[:, i, j] = AMP[key] * np.exp(1j * PH[key])
                elif key in RE:
                    s[:, i, j] = RE[key] + 1j * np.zeros_like(RE[key])
                elif key in IM:
                    s[:, i, j] = 1j * IM[key]
                elif key in AMP:
                    s[:, i, j] = AMP[key]  # фаза = 0

        net = rf.Network(f=self.freq_hz, s=s, f_unit="Hz")
        return TouchstoneData(net, params=dict(meta.get("params", {})) if meta else {})

    def dumps(self) -> bytes:
        """Сериализация → pickle-байты."""
        return pickle.dumps({"ver": self.VERSION, **self.__dict__})

    @classmethod
    def loads(cls, data: bytes) -> "TouchstoneCodec":
        """Десериализация из pickle-байтов."""
        obj = cls.__new__(cls)
        obj.__dict__.update(pickle.loads(data))
        return obj

    # ────────────────────────────────────────────────────────────────────────
    # PRIVATE UTILITIES
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_channel(tag: str) -> Tuple[int, int, str]:
        """
        Парсинг строки 'Sij.part' → (i, j, part).
        """
        try:
            idx, part = tag.split(".")
            if not idx.startswith("S") or len(idx[1:]) != 2:
                raise ValueError
            i, j = int(idx[1]) - 1, int(idx[2]) - 1
        except Exception:
            raise ValueError(f"Invalid channel tag: {tag!r}") from None

        part = part.lower()
        if part not in {"real", "imag", "db", "mag", "deg"}:
            raise ValueError(f"Unknown component: {part!r}")
        return i, j, part

    def _convert(self, s: np.ndarray, part: str) -> np.ndarray:
        """
        Выбор требуемой компоненты из комплексного вектора.
        """
        p = part.lower()
        if p == "real":
            return s.real
        if p == "imag":
            return s.imag
        if p == "db":
            return 20 * np.log10(np.abs(s) + self.eps_db)
        if p == "mag":
            return np.abs(s)
        if p == "deg":
            return np.unwrap(np.angle(s)) * 180 / math.pi
        raise RuntimeError(f"Unexpected part {part!r} in _convert")

    def _reverse(self, arr: np.ndarray, part: str) -> Tuple[str, np.ndarray]:
        """
        Инверсия компоненты:
        - real → re
        - imag → im
        - mag → amp
        - db   → amp
        - deg  → phase
        """
        p = part.lower()
        if p == "real":
            return "re", arr
        if p == "imag":
            return "im", arr
        if p == "mag":
            return "amp", arr
        if p == "db":
            return "amp", 10 ** (arr / 20)
        if p == "deg":
            return "phase", np.deg2rad(arr)
        raise RuntimeError(f"Unexpected part {part!r} in _reverse")
