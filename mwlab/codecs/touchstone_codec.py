# mwlab/codecs/touchstone_codec.py
"""
TouchstoneCodec
===============

Расширенная версия кодека MWLab с поддержкой:

* гибкого режима резервного сохранения S‑матрицы (`backup_mode` = NONE | MISSING | FULL);
* специализированных методов для инференса, когда на входе присутствует
  **только** X‑параметры или **только** S‑параметры:
    - `encode_x` / `decode_x`
    - `encode_s` / `decode_s`

Большая часть исходной логики (encode / decode полных TouchstoneData,
(d)e‑сериализация, utilities) сохранена — добавлены лишь новые
ветки кода.
"""
from __future__ import annotations

import enum
import math
import pickle
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import skrf as rf
import torch

from mwlab.io.touchstone import TouchstoneData
from mwlab.datasets.touchstone_dataset import TouchstoneDataset

__all__ = ["TouchstoneCodec", "BackupMode"]


# -----------------------------------------------------------------------------
#                               Backup Mode
# -----------------------------------------------------------------------------
class BackupMode(enum.Enum):
    """Определяет, что сохраняем в meta при неполной S‑матрице.

    NONE     – ничего не сохраняем (отсутствующие компоненты будут NaN/0).
    MISSING  – сохраняем *только* отсутствующие пары Sij в meta['s_missing'].
    FULL     – сохраняем всю матрицу в meta['s_backup'].
    """

    NONE = "none"
    MISSING = "missing"
    FULL = "full"

    def __str__(self) -> str:  # полезно для печати / сериализации
        return self.value


# -----------------------------------------------------------------------------
#                               TouchstoneCodec
# -----------------------------------------------------------------------------
class TouchstoneCodec:
    """Преобразует Touchstone‑данные ⇄ тензоры PyTorch.

    Основные методы:
    ----------------
    * **encode** / **decode** – работают с полноценным ``TouchstoneData``.
    * **encode_x** / **decode_x** – только с X‑вектором (прямая модель X→S).
    * **encode_s** / **decode_s** – только с S‑матрицей (обратная модель S→X).

    Параметры конструктора (новые\*):
    ---------------------------------
    backup_mode : BackupMode | str, default="missing"
        Стратегия сохранения непокрытых компонент S‑матрицы.
    """

    VERSION = 1

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
        backup_mode: BackupMode | str = BackupMode.FULL,
    ) -> None:
        self.x_keys: List[str] = list(x_keys)
        self.y_channels: List[str] = list(y_channels)
        self.freq_hz: np.ndarray = np.asarray(freq_hz, dtype=float)
        self.eps_db = float(eps_db)
        self.force_resample = bool(force_resample)
        self.nan_fill = nan_fill
        self.backup_mode = BackupMode(backup_mode)

        # количество портов определяется максимумом индексов в y_channels
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
        backup_mode: BackupMode | str = BackupMode.FULL,
    ) -> "TouchstoneCodec":
        """Автоматическое построение кодека по «сырому» TouchstoneDataset."""
        if not isinstance(ds, TouchstoneDataset):
            raise TypeError("from_dataset() ожидает TouchstoneDataset")
        if len(ds) == 0:
            raise ValueError("Dataset пуст — нечего анализировать")

        key_union: set[str] = set()
        first_net: Optional[rf.Network] = None
        for idx in range(len(ds)):
            x_dict, net = ds[idx]
            if first_net is None:
                first_net = net
            key_union.update(x_dict.keys())
        assert first_net is not None

        n_ports = first_net.number_of_ports
        freq_hz = first_net.f  # после всех s_tf

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
            backup_mode=backup_mode,
        )

    # =====================================================================
    #                          X‑ЧАСТЬ  (params)
    # =====================================================================
    def encode_x(self, params: Mapping[str, float], *, strict: bool = False) -> torch.Tensor:
        """Преобразует dict → тензор (Dx,).

        strict=True — если отсутствует какой‑то ключ из ``self.x_keys`` → бросаем KeyError.
        """
        vec: List[float] = []
        for k in self.x_keys:
            if k in params:
                vec.append(float(params[k]))
            else:
                if strict:
                    raise KeyError(f"encode_x: параметр '{k}' отсутствует")
                vec.append(float("nan"))
        return torch.tensor(vec, dtype=torch.float32)

    def decode_x(self, x_tensor: torch.Tensor) -> Dict[str, float]:
        """Тензор → словарь параметров с python float."""
        if x_tensor.dim():
            x_tensor = x_tensor.view(-1)
        return {k: float(v) for k, v in zip(self.x_keys, x_tensor.tolist())}

    # =====================================================================
    #                          S‑ЧАСТЬ  (Network)
    # =====================================================================
    def encode_s(
        self,
        net: rf.Network,
        *,
        strict: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Готовит вход для **обратной** модели S→X.

        Возвращает пару ``(y_tensor, meta)`` точно в том же формате, который
        ждут методы ``decode / decode_s``.
        """
        # 1) ресэмплируем, если нужно
        if self.force_resample and (
            len(net.f) != len(self.freq_hz) or not np.allclose(net.f, self.freq_hz)
        ):
            net = net.copy()
            freq_target = rf.Frequency.from_f(self.freq_hz, unit="Hz")
            net.resample(freq_target)
        else:
            if strict and not np.allclose(net.f, self.freq_hz):
                raise ValueError("encode_s: несоответствие частотной сетки")

        # 2) y‑тензор (C,F)
        chans = [
            self._convert(net.s[:, i, j], part)
            for (i, j, part) in map(self._parse_channel, self.y_channels)
        ]
        y_t = torch.from_numpy(np.stack(chans, axis=0)).float()

        # 3) meta (минимальный набор)
        meta: Dict[str, Any] = {
            "unit": net.frequency.unit,
            "s_def": net.s_def,
            "z0": net.z0[0].copy(),
            "n_ports": int(net.number_of_ports),
            "comments": net.comments,
        }
        # резервные данные – логика совпадает с encode()
        self._maybe_add_backup(meta, net)
        return y_t, meta

    def decode_s(
        self,
        y_tensor: torch.Tensor,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> rf.Network | TouchstoneData:
        """Обратное преобразование S‑тензора → ``rf.Network`` / ``TouchstoneData``.

        Если в meta есть пользовательские ``params`` → возвращается `TouchstoneData`,
        иначе только `rf.Network`.
        """
        ts = self.decode(y_tensor, meta)
        if meta and "params" in meta:
            return ts
        return ts.network

    # =====================================================================
    #                       ПОЛНЫЙ encode / decode
    # =====================================================================
    def encode(
        self, ts: TouchstoneData
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Полная упаковка ``TouchstoneData`` → ``(x_t, y_t, meta)``."""
        # -------- X‑вектор -------------------------------------------------
        x_t = self.encode_x(ts.params)

        # -------- S‑часть --------------------------------------------------
        y_t, meta = self.encode_s(ts.network, strict=False)

        # meta + пользовательские параметры + путь к исходнику
        meta["params"] = ts.params
        meta["orig_path"] = str(ts.path) if ts.path else None
        return x_t, y_t, meta

    @torch.no_grad()
    def decode(
        self, y_pred: torch.Tensor, meta: Optional[Mapping[str, Any]] = None
    ) -> TouchstoneData:
        """Обратное преобразование ``(y_pred, meta)`` → ``TouchstoneData``."""
        if y_pred.dim() == 3:
            y_pred = y_pred[0]
        y_np = y_pred.cpu().float().numpy()
        meta = meta or {}

        P = int(meta.get("n_ports", self.n_ports))
        F = len(self.freq_hz)

        # ---------- распределяем по компонентам ---------------------------
        RE: Dict[Tuple[int, int], np.ndarray] = {}
        IM: Dict[Tuple[int, int], np.ndarray] = {}
        AMP: Dict[Tuple[int, int], np.ndarray] = {}
        PH: Dict[Tuple[int, int], np.ndarray] = {}
        for k, tag in enumerate(self.y_channels):
            i, j, part = self._parse_channel(tag)
            kind, arr = self._reverse(y_np[k], part)
            {"re": RE, "im": IM, "amp": AMP, "phase": PH}[kind][(i, j)] = arr

        # ---------- исходный контейнер S[FxPxP] ---------------------------
        s = np.full((F, P, P), self.nan_fill, dtype=complex)

        # резерв full
        if "s_backup" in meta:
            s[:, :, :] = np.asarray(meta["s_backup"], dtype=complex)
        # резерв missing
        if "s_missing" in meta:
            for key, arr in meta["s_missing"].items():
                i, j = map(int, key.split("_"))
                s[:, i, j] = arr

        # ---------- заполняем тем, что предсказано ------------------------
        for i in range(P):
            for j in range(P):
                key = (i, j)
                if key in RE and key in IM:
                    s[:, i, j] = RE[key] + 1j * IM[key]
                elif key in AMP and key in PH:
                    s[:, i, j] = AMP[key] * np.exp(1j * PH[key])
                elif key in PH:
                    amp = np.abs(s[:, i, j]) if not np.isnan(s[:, i, j]).all() else 1.0
                    s[:, i, j] = amp * np.exp(1j * PH[key])
                elif key in RE:
                    s[:, i, j] = RE[key] + 1j * 0.0
                elif key in IM:
                    s[:, i, j] = 1j * IM[key]
                elif key in AMP:
                    s[:, i, j] = AMP[key]

        # NaN → 0 (безопасно для dB)
        nan_mask = np.isnan(s.real) | np.isnan(s.imag)
        s[nan_mask] = 0.0 + 0.0j

        # ---------- восстанавливаем Network --------------------------------
        freq = rf.Frequency.from_f(self.freq_hz, unit="Hz")
        orig_unit = meta.get("unit", "Hz")
        freq.unit = orig_unit

        z0_meta = meta.get("z0", 50)
        if np.ndim(z0_meta) == 0:
            z0_full = np.full((F, P), z0_meta, dtype=complex)
        else:
            z0_vec = np.asarray(z0_meta, dtype=complex).reshape(1, -1)
            z0_full = np.broadcast_to(z0_vec, (F, P))

        net = rf.Network(frequency=freq, s=s, z0=z0_full)
        net.s_def = meta.get("s_def", None)
        net.comments = meta.get("comments", [])

        return TouchstoneData(net, params=dict(meta.get("params", {})))

    # ---------------------------------------------------------------------
    #                           helpers (backup)
    # ---------------------------------------------------------------------
    def _maybe_add_backup(self, meta: Dict[str, Any], net: rf.Network) -> None:
        """Добавляет резервные данные S‑матрицы в meta по self.backup_mode."""
        # определяем, все ли пары Sij покрыты y_channels
        net_ports = int(net.number_of_ports)
        full_pairs = {(i, j) for i in range(net_ports) for j in range(net_ports)}
        covered_pairs = {self._parse_channel(tag)[:2] for tag in self.y_channels}
        need_backup = covered_pairs != full_pairs
        if not need_backup:
            return

        if self.backup_mode is BackupMode.FULL:
            meta["s_backup"] = net.s.astype(np.complex64)
        elif self.backup_mode is BackupMode.MISSING:
            meta["s_missing"] = {
                f"{i}_{j}": net.s[:, i, j].astype(np.complex64)
                for (i, j) in full_pairs - covered_pairs
            }
        # BackupMode.NONE → ничего не пишем

    # ---------------------------------------------------------------------
    #                         (de)‑serialization
    # ---------------------------------------------------------------------
    def dumps(self) -> bytes:
        payload = {"ver": self.VERSION, **self.__dict__}
        # Enum не пикуется красиво → сохраняем строку
        payload["backup_mode"] = str(self.backup_mode)
        return pickle.dumps(payload)

    @classmethod
    def loads(cls, data: bytes) -> "TouchstoneCodec":
        state = pickle.loads(data)
        ver = state.pop("ver", 0)
        if ver != cls.VERSION:
            raise ValueError(
                f"Incompatible TouchstoneCodec version: file {ver}, code {cls.VERSION}"
            )
        # строку → Enum
        state["backup_mode"] = BackupMode(state["backup_mode"])
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    # ---------------------------------------------------------------------
    #                               utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _parse_channel(tag: str) -> Tuple[int, int, str]:
        m = re.fullmatch(r"S(\d+)_(\d+)\.(\w+)", tag, re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid channel tag: {tag!r}")
        i, j, part = int(m.group(1)) - 1, int(m.group(2)) - 1, m.group(3).lower()
        if part not in {"real", "imag", "db", "mag", "deg"}:
            raise ValueError(f"Unknown component: {part!r}")
        return i, j, part

    # forward conversion ---------------------------------------------------
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

    # reverse conversion ---------------------------------------------------
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
            f"{self.__class__.__name__}(x_keys={len(self.x_keys)}, "
            f"y_channels={len(self.y_channels)}[{','.join(sorted(comps))}], "
            f"freq_pts={len(self.freq_hz)}, ports={self.n_ports}, "
            f"backup={self.backup_mode})"
        )

    __str__ = __repr__
