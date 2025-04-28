# mwlab/codecs/touchstone_codec.py
"""
Кодек TouchstoneData ↔︎ (X-тензор, Y-тензор, meta).

Функциональность
----------------
- encode()        : TouchstoneData → (x_t, y_t, meta);
- decode()        : (y_pred, meta) → TouchstoneData;
- dumps()/loads() : pickle-сериализация для checkpoint-ов;
- from_dataset()  : метод для автоматической генерации Codec для директории *.sNp.
"""

from __future__ import annotations
import math
import pickle
import numpy as np
import torch
import skrf as rf
from typing import Sequence, Tuple, List, Dict, Any, Mapping, Optional

from mwlab.io.touchstone import TouchstoneData


# ──────────────────────────────────────────────────────────────────────────
#                               TouchstoneCodec
# ──────────────────────────────────────────────────────────────────────────
class TouchstoneCodec:
    """
    Преобразование Touchstone-файлов (TouchstoneData) в тензоры PyTorch и обратно.

    Публичные методы
    ----------------
    encode(ts: TouchstoneData)           → (x_t, y_t, meta)
    decode(y_pred, meta)                 → TouchstoneData
    dumps() / loads()                    → bytes  /  TouchstoneCodec
    from_dataset(TouchstoneDataset, …)   → TouchstoneCodec (фабричный метод)

    Параметры конструктора
    ----------------------
    x_keys : list[str]
        Список признаков модели (X-пространство).
    y_channels : list[str]
        Список каналов сети вида 'Sij.part' (Y-пространство).
    freq_hz : np.ndarray
        Частотная сетка для S-параметров (Hz).
    eps_db : float, optional
        Малое значение для защиты при логарифмировании (по умолчанию 1e-12).
    force_resample : bool, optional
        Принудительный ресемплинг к freq_hz (по умолчанию True).
    nan_fill : complex, optional
        Значение-заполнитель для пропущенных элементов матрицы S.
    """

    VERSION: int = 1  # Номер схемы сериализации (сохранения класса в checkpoint)

    # ------------------------------------------------------------------ init
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

        # определяем количество портов устройства по y_channels
        pairs = [self._parse_channel(tag)[:2] for tag in self.y_channels]
        self.n_ports = int(max(max(i, j) for i, j in pairs) + 1)

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_dataset(
        cls,
        ds,                    # TouchstoneDataset
        *,
        components: Sequence[str] = ("real", "imag"),
        eps_db: float = 1e-12,
        force_resample: bool = True,
        nan_fill: complex | float = np.nan + 1j * np.nan,
    ) -> "TouchstoneCodec":                # ← строковая аннотация
        """
        Автоматически формирует Codec по «сырому» (исходному) TouchstoneDataset.
        """
        # ленивый импорт, чтобы не провоцировать циклические связи
        from mwlab.datasets.touchstone_dataset import TouchstoneDataset

        if not isinstance(ds, TouchstoneDataset):
            raise TypeError("from_dataset() ожидает TouchstoneDataset")

        if len(ds) == 0:
            raise ValueError("Dataset пуст – нечего анализировать")

        # собираем полный набор параметров
        key_union: set[str] = set()
        first_ts: TouchstoneData | None = None
        for p in ds.paths:
            ts = TouchstoneData.load(p)
            key_union.update(ts.params.keys())
            if first_ts is None:
                first_ts = ts

        if first_ts is None:
            raise RuntimeError("Не удалось прочитать ни одного Touchstone-файла")

        # генерируем y_channels
        n_ports = first_ts.network.number_of_ports
        comp_lower = [c.lower() for c in components]
        y_ch = [
            f"S{i + 1}{j + 1}.{comp}"
            for comp in comp_lower
            for i in range(n_ports)
            for j in range(n_ports)
        ]

        return cls(
            x_keys=sorted(key_union),
            y_channels=y_ch,
            freq_hz=first_ts.network.f,
            eps_db=eps_db,
            force_resample=force_resample,
            nan_fill=nan_fill,
        )

    # ───────────────────────────────────────────────────────────── encode
    def encode(
        self, ts: TouchstoneData
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        TouchstoneData → (x_t, y_t, meta)

        x_t : (D,)       — float32
        y_t : (C, F)     — float32
        meta : dict      — {'params': …, 'orig_path': …}
        """
        # -------- X ----------------------------------------------------
        x_vec = torch.tensor([ts.params[k] for k in self.x_keys], dtype=torch.float32)

        # -------- Y (с ресемплингом при необходимости) -----------------
        net = ts.network
        need_resample = (
            len(net.f) != len(self.freq_hz) or not np.allclose(net.f, self.freq_hz)
        )
        if self.force_resample and need_resample:
            net = net.copy()
            net.resample(self.freq_hz)

        chans = [
            self._convert(net.s[:, i, j], part)
            for (i, j, part) in map(self._parse_channel, self.y_channels)
        ]
        y_t = torch.from_numpy(np.stack(chans, axis=0)).float()  # (C, F)

        meta = {
            "params": ts.params,
            "orig_path": str(ts.path) if ts.path else None,
        }
        return x_vec, y_t, meta

    # ───────────────────────────────────────────────────────────── decode
    @torch.no_grad()
    def decode(
        self, y_pred: torch.Tensor, meta: Optional[Mapping[str, Any]] = None
    ) -> TouchstoneData:
        """
        (y_pred, meta) → TouchstoneData.

        y_pred : (C,F)  *или* (B,C,F) — в случае батча берётся [0].
        """

        if y_pred.dim() == 3:
            y_pred = y_pred[0]
        y_np = y_pred.cpu().float().numpy()

        # --- первый проход ---
        RE, IM, AMP, PH = {}, {}, {}, {}
        for k, tag in enumerate(self.y_channels):
            i, j, part = self._parse_channel(tag)
            kind, arr = self._reverse(y_np[k], part)
            {"re": RE, "im": IM, "amp": AMP, "phase": PH}[kind][(i, j)] = arr

        # --- второй проход: сборка матрицы S ---
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
                    s[:, i, j] = RE[key] + 1j * 0.0
                elif key in IM:
                    s[:, i, j] = 1j * IM[key]
                elif key in AMP:
                    s[:, i, j] = AMP[key]

        net = rf.Network(f=self.freq_hz, s=s, f_unit="Hz")
        return TouchstoneData(net, params=dict(meta.get("params", {})) if meta else {})

    # ─────────────────────────────────────────────────── (de)serialization
    def dumps(self) -> bytes:
        """→ pickle-bytes (для сохранения в checkpoint)."""
        return pickle.dumps({"ver": self.VERSION, **self.__dict__})

    @classmethod
    def loads(cls, data: bytes) -> "TouchstoneCodec":
        """← из pickle-bytes."""
        obj = cls.__new__(cls)
        obj.__dict__.update(pickle.loads(data))
        return obj

    # ───────────────────────────────────────────────────── utilities
    @staticmethod
    def _parse_channel(tag: str) -> Tuple[int, int, str]:
        """Парсинг строки 'Sij.part' → (i, j, part)."""
        try:
            idx, part = tag.split(".")
            if not (idx.startswith("S") and len(idx) == 3):
                raise ValueError
            i, j = int(idx[1]) - 1, int(idx[2]) - 1
        except Exception:
            raise ValueError(f"Invalid channel tag: {tag!r}") from None

        part = part.lower()
        if part not in {"real", "imag", "db", "mag", "deg"}:
            raise ValueError(f"Unknown component: {part!r}")
        return i, j, part

    # forward component
    def _convert(self, s: np.ndarray, part: str) -> np.ndarray:
        """Комплексный вектор → скалярная компонента."""
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
        raise RuntimeError

    # reverse component
    def _reverse(self, arr: np.ndarray, part: str) -> Tuple[str, np.ndarray]:
        """Инверсия преобразования для восстановления комплексных величин."""
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
        raise RuntimeError

