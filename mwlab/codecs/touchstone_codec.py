# mwlab/codecs/touchstone_codec.py
"""
`TouchstoneCodec` превращает объект `TouchstoneData` в тензоры PyTorch
и обратно.  Поддерживает сериализацию (pickle) и авто‑конфигурацию
по готовому `TouchstoneDataset`.

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import re
import torch
import skrf as rf

from mwlab.io.touchstone import TouchstoneData
from mwlab.datasets.touchstone_dataset import TouchstoneDataset

# ─────────────────────────────────────────────────────────────────────────────
#                                   TouchstoneCodec
# ─────────────────────────────────────────────────────────────────────────────


class TouchstoneCodec:
    """
    Преобразует Touchstone‑файлы в тензоры PyTorch и обратно.

    Конструктор (ручной) .....................................................
    Parameters
    ----------
    x_keys       : list[str]             – имена скалярных параметров (X‑пространство)
    y_channels   : list[str]             – каналы вида ``"Sij.part"``
                                            part ∈ {real, imag, db, mag, deg}
    freq_hz      : ndarray (F,)          – целевая частотная сетка (Гц)
    eps_db       : float, default 1e‑12  – защита при логарифмировании (dB)
    force_resample : bool, default True  – если сетка не совпадает → resample
    nan_fill     : complex, default NaN+1jNaN – заполнитель при неполной матрице
    """

    VERSION = 1  # Номер схемы сериализации (сохранения класса в checkpoint)

    # --------------------------------------------------------------------- init
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

        # максимум индексов портов → число портов
        pairs = [self._parse_channel(tag)[:2] for tag in self.y_channels]
        self.n_ports: int = int(max(max(i, j) for i, j in pairs) + 1)

    # ----------------------------------------------------------------- factory
    @classmethod
    def from_dataset(
        cls,
        ds: TouchstoneDataset,
        *,
        components: Sequence[str] = ("real", "imag"),
        eps_db: float = 1e-12,
        force_resample: bool = True,
        nan_fill: complex | float = np.nan + 1j * np.nan,
    ) -> "TouchstoneCodec":
        """
        Автоматически формирует Codec на основе `TouchstoneDataset`.

        • Собирает объединение ключей параметров → `x_keys`.
        • Берет первую сеть → определяет `n_ports` и `freq_hz`.
        • Формирует `y_channels` = Sij.компонента по всем портам.
        """

        if not isinstance(ds, TouchstoneDataset):
            raise TypeError("from_dataset() ожидает TouchstoneDataset")

        if len(ds) == 0:
            raise ValueError("Dataset пуст – нечего анализировать")

        key_union: set[str] = set()
        first_net: Optional[rf.Network] = None

        for idx in range(len(ds)):
            x_dict, net = ds[idx]  # TouchstoneDataset гарантирует такую структуру
            if not isinstance(x_dict, dict) or not hasattr(net, "s"):
                raise TypeError(
                    "from_dataset() ожидает, что элемент датасета содержит "
                    "dict и skrf.Network"
                )
            key_union.update(x_dict.keys())
            if first_net is None:
                first_net = net

        assert first_net is not None
        n_ports = first_net.number_of_ports
        freq_hz = first_net.f  # уже после s_tf

        comps = [c.lower() for c in components]
        y_channels = [
            f"S{i + 1}_{j + 1}.{comp}"
            for comp in comps
            for i in range(n_ports)
            for j in range(n_ports)
        ]

        return cls(
            x_keys=sorted(key_union),
            y_channels=y_channels,
            freq_hz=freq_hz,
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

        x_t : (Dx,)   float32
        y_t : (C,F)  float32   (C = len(y_channels))
        meta: словарь, необходимый для полного восстановления сети
        """
        # ----------- X‑вектор -----------------------------------------
        x_vec = torch.tensor(
            [ts.params.get(k, float("nan")) for k in self.x_keys],
            dtype=torch.float32,
        )

        # ----------- S‑матрица (ресемплинг) ---------------------------
        net = ts.network

        net_ports = int(net.number_of_ports)
        if net_ports > self.n_ports:
            self.n_ports = net_ports

        if self.force_resample and (
            len(net.f) != len(self.freq_hz) or not np.allclose(net.f, self.freq_hz)
        ):
            net = net.copy()
            freq_target = rf.Frequency.from_f(self.freq_hz, unit="Hz")
            net.resample(freq_target)
        else:
            if not np.allclose(net.f, self.freq_hz):
                raise ValueError("Несоответствие сетки частот и force_resample=False")

        chans = [
            self._convert(net.s[:, i, j], part)
            for (i, j, part) in map(self._parse_channel, self.y_channels)
        ]
        y_t = torch.from_numpy(np.stack(chans, axis=0)).float()

        covered_pairs = {self._parse_channel(tag)[:2] for tag in self.y_channels}
        full_pairs = {(i, j) for i in range(net_ports) for j in range(net_ports)}
        need_backup = covered_pairs != full_pairs

        # ----------- meta (вся системная инфа + user params) ----------
        meta: Dict[str, Any] = {
            "params": ts.params,
            "unit": net.frequency.unit,
            "s_def": net.s_def,
            "z0": net.z0[0].copy(),          # (P,)
            "n_ports": net_ports,
            "comments": net.comments,
            "orig_path": str(ts.path) if ts.path else None,
        }
        if need_backup:
            meta["s_backup"] = net.s.astype(np.complex64)
        return x_vec, y_t, meta

    # ───────────────────────────────────────────────────────────── decode
    @torch.no_grad()
    def decode(
        self, y_pred: torch.Tensor, meta: Optional[Mapping[str, Any]] = None
    ) -> TouchstoneData:
        """
        (y_pred, meta) → TouchstoneData

        • `y_pred` формы (C,F) или (B,C,F) → используется первый элемент.
        • Если `meta` отсутствует — восстанавливаем базово (unit='Hz', z0=50Ω).
        """

        if y_pred.dim() == 3:
            y_pred = y_pred[0]
        y_np = y_pred.cpu().float().numpy()

        meta = meta or {}
        P = int(meta.get("n_ports", self.n_ports))
        F = len(self.freq_hz)

        # --- разбор по компонентам -----------------------------------
        RE, IM, AMP, PH = {}, {}, {}, {}
        for k, tag in enumerate(self.y_channels):
            i, j, part = self._parse_channel(tag)
            kind, arr = self._reverse(y_np[k], part)
            {"re": RE, "im": IM, "amp": AMP, "phase": PH}[kind][(i, j)] = arr

        # --- сборка комплексной S‑матрицы ----------------------------
        if "s_backup" in meta:
            s = np.asarray(meta["s_backup"]).astype(complex).copy()
            F, P = s.shape[0], s.shape[1]
        else:
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

        # заменяем NaN → 0
        nan_mask = np.isnan(s.real) | np.isnan(s.imag)
        s[nan_mask] = 0.0 + 0.0j

        # --- восстанавливаем метаданные ------------------------------
        meta = meta or {}
        # всегда считаем, что self.freq_hz в Гц
        freq = rf.Frequency.from_f(self.freq_hz, unit="Hz")
        # но сохраним оригинальную единицу для отображения графиков
        orig_unit = meta.get("unit", "Hz")
        freq.unit = orig_unit

        z0_meta = meta.get("z0", 50)  # scalar или (P,)
        if np.ndim(z0_meta) == 0:
            z0_full = np.full((F, P), z0_meta, dtype=complex)
        else:
            z0_vec = np.asarray(z0_meta, dtype=complex).reshape(1, -1)  # (1,P)
            if z0_vec.shape[1] != P:
                raise ValueError("Количество значений meta['z0'] не соответствует n_ports")
            z0_full = np.broadcast_to(z0_vec, (F, P))

        net = rf.Network(frequency=freq, s=s, z0=z0_full)
        net.s_def = meta.get("s_def", None)
        net.comments = meta.get("comments", [])

        return TouchstoneData(net, params=dict(meta.get("params", {})))

    # ─────────────────────────────────────────────────── (de)serialization
    def dumps(self) -> bytes:
        """Сериализация → bytes (pickle)."""
        payload = {"ver": self.VERSION, **self.__dict__}
        return pickle.dumps(payload)

    @classmethod
    def loads(cls, data: bytes) -> "TouchstoneCodec":
        """Десериализация ← bytes (pickle)."""
        state = pickle.loads(data)
        ver = state.pop("ver", 0)
        if ver != cls.VERSION:
            raise ValueError(
                f"Incompatible TouchstoneCodec version: file {ver}, code {cls.VERSION}"
            )
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    # ───────────────────────────────────────────────────── utilities
    @staticmethod
    def _parse_channel(tag: str) -> Tuple[int, int, str]:
        """
        'S<i>_<j>.<part>' → (i, j, part)  (нумерация портов с 0).
        """
        try:
            m = re.fullmatch(r"S(\d+)_(\d+)\.(\w+)", tag, re.IGNORECASE)
            if not m:
                raise ValueError(f"Invalid channel tag: {tag!r}")

            i, j, part = int(m.group(1)) - 1, int(m.group(2)) - 1, m.group(3).lower()
            if part not in {"real", "imag", "db", "mag", "deg"}:
                raise ValueError(f"Unknown component: {part!r}")
            return i, j, part
        except Exception:
            raise ValueError(f"Invalid channel tag: {tag!r}") from None

        part = part.lower()
        if part not in {"real", "imag", "db", "mag", "deg"}:
            raise ValueError(f"Unknown component: {part!r}")
        return i, j, part

    # forward component
    def _convert(self, s: np.ndarray, part: str) -> np.ndarray:
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

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # noqa: D401
        comps = {self._parse_channel(t)[2] for t in self.y_channels}
        return (
            f"{self.__class__.__name__}("
            f"x_keys={len(self.x_keys)}, "
            f"y_channels={len(self.y_channels)}[{','.join(sorted(comps))}], "
            f"freq_pts={len(self.freq_hz)}, "
            f"ports={self.n_ports})"
        )

    __str__ = __repr__


