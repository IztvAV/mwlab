#mwlab/filters/devices.py
"""
mwlab.filters.devices
=====================
Высокоуровневый слой «устройств» (Device-level) поверх матрицы связи.

Что изменилось по сравнению с прежней версией:
---------------------------------------------
* Backend NumPy/Torch заменён на **Torch-only**. Параметр `backend` удалён.
* Добавлен параметр `device` (по умолчанию авто: CUDA если доступна, иначе CPU).
* Все расчёты S-параметров выполняются на Torch; для Touchstone/`skrf` данные
  конвертируются в NumPy (`.detach().cpu().numpy()`).
* Логика и API классов сохранены концептуально, но сигнатуры обновлены.

Компоненты
----------
* **Device**  – абстрактное устройство (N-порт, любое число звеньев)
* **Filter**  – 2-портовый одно-полосный фильтр (LP / HP / BP / BR)
* **Multiplexer** – заготовка N-канального мультиплексора (не завершён)

Основные возможности
--------------------
* Прямой расчёт S-матрицы устройства в шкале **f (Гц)**.
* Преобразования сеток Ω ↔ f с учётом типа фильтра.
* Круглая триада «CouplingMatrix ↔ Filter ↔ Touchstone».
* Фабрики-шорткаты: `Filter.lp / hp / bp / br / bp_edges / br_edges`.

Зависимости
-----------
* torch — вычисления S
* numpy — вспомогательные операции, создание массивов частот
* scikit-rf (skrf) — Touchstone I/O

Автор: (c) MWLab, 2025
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Mapping, Sequence, Tuple, Optional

import numpy as np
import torch
import skrf as rf

from mwlab.filters.cm import CouplingMatrix
from mwlab.filters.cm_core import DEFAULT_DEVICE
from mwlab.io.touchstone import TouchstoneData

# ────────────────────────────────────────────────────────────────────────────
#                               helpers
# ────────────────────────────────────────────────────────────────────────────

def _as_frequency(f, *, unit: str = "Hz") -> Tuple[np.ndarray, Optional[rf.Frequency]]:
    """Приводит *f* к ``ndarray`` в Гц.

    Если передан объект ``rf.Frequency`` – возвращает ``(f_numpy, rf.Frequency)``
    для сохранения исходных единиц при экспортe в Touchstone.

    Параметр *unit* используется только когда на входе массив чисел (для
    согласованности с прежним API, хотя здесь мы предполагаем, что массив уже в Гц).
    """
    if isinstance(f, rf.Frequency):
        return f.f.copy(), f
    arr = np.asarray(f, dtype=float)
    if arr.ndim == 0:  # скаляр → (1,)
        arr = arr.reshape(1)
    return arr, None


def _to_hz(val: float, unit: str) -> float:
    """Переводит число *val* из unit ('kHz'/'MHz'/'GHz'/…) в Гц."""
    mdict = rf.frequency.Frequency.multiplier_dict
    try:
        mult = mdict[unit.lower()]
    except KeyError as err:
        valid = ", ".join(sorted(mdict.keys()))
        raise ValueError(
            f"Неизвестная единица частоты: {unit!r}. Допустимые: {valid}"
        ) from err
    return float(val) * mult


# ────────────────────────────────────────────────────────────────────────────
#                                   Device
# ────────────────────────────────────────────────────────────────────────────

class Device(ABC):
    """Абстрактное устройство, описываемое расширенной матрицей связи.

    Атрибуты
    --------
    cm : CouplingMatrix
        Контейнер с топологией и параметрами матрицы связи.
    name : str | None
        Человекочитаемое имя устройства.
    """

    __slots__ = ("cm", "name")

    def __init__(self, cm: CouplingMatrix, *, name: str | None = None):
        self.cm = cm
        self.name = str(name) if name is not None else None

    # ---------------------------------------------------------------- props
    @property
    def order(self) -> int:
        """Порядок (число резонаторов)."""
        return self.cm.topo.order

    @property
    def ports(self) -> int:
        """Число портов."""
        return self.cm.topo.ports

    @property
    def Q(self):
        """Физическая добротность резонаторов (скаляр или вектор) или None."""
        if self.cm.qu is None:
            return None
        return np.asarray(self.cm.qu) / self._qu_scale()

    # ----------------------------------------------------- abstract methods
    @abstractmethod
    def _omega(self, f_hz: np.ndarray) -> np.ndarray:
        """Преобразование **f(Гц)** → **Ω** для данного устройства."""
        ...

    @abstractmethod
    def _qu_scale(self) -> float | np.ndarray:
        """Отношение qu / Q для устройства.

        LP/HP  → 1
        BP/BR  → FBW
        """
        ...

    # ---------- установка Q со стороны пользователя ----------
    def set_Q(self, Q):
        """Задать *физические* добротности Q и пересчитать нормированные qu."""
        if Q is None:
            self.cm.qu = None
        else:
            arr = np.asarray(Q, dtype=float)
            if arr.ndim == 0:  # скаляр
                arr = np.full(self.order, float(arr), dtype=float)
            if arr.size != self.order:
                raise ValueError(f"Q: ожидалось {self.order} значений, получено {arr.size}")
            self.cm.qu = arr * self._qu_scale()

    # ---------------------------------------------------------------- API – S-параметры
    def sparams(
        self,
        f_hz,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        method: str = "auto",
        fix_sign: bool = False,
    ) -> torch.Tensor:
        """Комплексная матрица **S(f)** в реальной шкале частот.

        Parameters
        ----------
        f_hz : array-like | rf.Frequency
            Сетка частот в Гц или объект `skrf.Frequency`.
        device : torch.device | str
            Устройство вычислений (по умолчанию авто).
        method : {"auto","inv","solve"}
            Алгоритм обращения матрицы (см. `cm_core.solve_sparams`).
        fix_sign : bool
            Инвертировать ли знак S12/S21 для 2-портового случая.

        Returns
        -------
        torch.Tensor
            Тензор формы (F, P, P) complex64 (или с batch-осями, если поданы соответствующие батчи).
        """
        f_arr, _ = _as_frequency(f_hz)
        omega = self._omega(f_arr)
        return self.cm.sparams(omega, device=device, method=method, fix_sign=fix_sign)

    # -------------------------- Touchstone экспорт -----------------------
    def to_touchstone(
        self,
        f_hz,
        *,
        unit: str = "Hz",
        device: str | torch.device = DEFAULT_DEVICE,
        method: str = "auto",
        fix_sign: bool = False,
    ) -> TouchstoneData:
        """Возвращает `TouchstoneData` с готовой сетью `skrf.Network`.

        Все метаданные (параметры фильтра/матрицы связи) кладутся в `params`.
        """
        f_arr, freq_obj = _as_frequency(f_hz)
        S_t = self.sparams(f_arr, device=device, method=method, fix_sign=fix_sign)
        S_np = S_t.detach().cpu().numpy()

        freq = freq_obj or rf.Frequency.from_f(f_arr, unit=unit)
        net = rf.Network(frequency=freq, s=S_np)

        params = {
            **self.cm.to_dict(),
            **self._device_params(),
            "order": self.order,
            "ports": self.ports,
        }
        return TouchstoneData(net, params=params)

    # -------------------------- Touchstone импорт ------------------------
    @classmethod
    def from_touchstone(
        cls,
        ts: TouchstoneData,
        *,
        kind: str | None = None,
        name: str | None = None,
    ):
        """Восстанавливает объект-потомок `Device` из Touchstone-структуры."""
        dev = str(ts.params.get("device", "")).lower()
        if dev and dev != "filter":
            raise ValueError(
                f"from_touchstone: ожидалось устройство 'Filter', получено {dev!r}"
            )

        cm = CouplingMatrix.from_dict(None, ts.params)  # topology inference
        return Filter._from_touchstone(ts.params, cm, kind=kind, name=name)

    # ---------------------------------------------------------------- params
    def _device_params(self) -> Dict[str, float | str]:
        """Параметры устройства для записи в Touchstone (переопределяется)."""
        return {"device": type(self).__name__}

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:
        extra = f", name={self.name!r}" if self.name else ""
        return f"{type(self).__name__}(order={self.order}, ports={self.ports}{extra})"


# ────────────────────────────────────────────────────────────────────────────
#                                  Filter
# ────────────────────────────────────────────────────────────────────────────

class Filter(Device):
    """Универсальный **2-портовый** одно-полосный фильтр.

    Поддерживаемые типы (`kind`):
        * LP — low-pass
        * HP — high-pass
        * BP — band-pass
        * BR — band-reject

    Задание полосы:
    --------------
    | kind  | допустимые параметры                 |
    |-------|--------------------------------------|
    | LP/HP | `f_cut`                              |
    | BP/BR | `f_edges=(f_l,f_u)` **или** `f0,bw` **или** `f0,fbw` |

    Фабрики-шорткаты: `Filter.lp`, `hp`, `bp`, `br`, `bp_edges`, `br_edges`.
    """

    __slots__ = ("kind", "f_edges", "f0", "bw", "fbw", "_spec")

    # ------------------------------------------------------- internal utils
    @staticmethod
    def _resolve_edges(
        kind: str,
        f_edges,
        f0,
        bw,
        fbw,
    ) -> Tuple[str, float, Optional[float], Optional[float], Tuple[Optional[float], Optional[float]]]:
        """Унифицирует входные частотные аргументы.

        Возвращает кортеж ``(spec, f0, bw, fbw, (f_l, f_u))``, где *spec* фиксирует,
        каким образом была задана полоса ("cut" | "bw" | "fbw" | "edges").
        """
        kind = kind.upper()

        # ---------- LP / HP ------------------------------------------------
        if kind in {"LP", "HP"}:
            if f_edges is not None:
                if len(f_edges) != 2:
                    raise ValueError("f_edges должен быть кортежем (low, high)")
                low, high = f_edges
                f_cut = (low if low is not None else high)
            else:
                f_cut = f0
            if f_cut is None:
                raise ValueError("LP/HP: необходимо указать f_cut (f0 или f_edges)")
            edges = (f_cut, None) if kind == "LP" else (None, f_cut)
            return "cut", float(f_cut), None, None, edges

        # ---------- BP / BR ------------------------------------------------
        spec_cnt = sum(bool(x) for x in (
            f_edges is not None,
            (f0 is not None and bw is not None),
            (f0 is not None and fbw is not None),
        ))
        if spec_cnt != 1:
            raise ValueError("BP/BR: задайте ровно одну комбинацию (f_edges) | (f0+bw) | (f0+fbw)")

        # (a) Границы полосы
        if f_edges is not None:
            f_l, f_u = map(float, f_edges)
            if f_l >= f_u:
                raise ValueError("f_edges: f_l должно быть < f_u")
            f0_ = math.sqrt(f_l * f_u)
            bw_ = f_u - f_l
            fbw_ = bw_ / f0_
            return "edges", f0_, bw_, fbw_, (f_l, f_u)

        # (b) f0 + bw
        if bw is not None:
            if f0 is None:
                raise ValueError("требуется указать f0 вместе с bw")
            f0_ = float(f0)
            bw_ = float(bw)
            if bw_ <= 0:
                raise ValueError("bw должно быть > 0")
            fbw_ = bw_ / f0_
            f_l, f_u = f0_ - bw_ / 2, f0_ + bw_ / 2
            return "bw", f0_, bw_, fbw_, (f_l, f_u)

        # (c) f0 + fbw
        if f0 is None:
            raise ValueError("требуется указать f0 вместе с fbw")
        f0_ = float(f0)
        fbw_ = float(fbw)
        if fbw_ <= 0:
            raise ValueError("fbw должно быть > 0")
        bw_ = fbw_ * f0_
        f_l, f_u = f0_ - bw_ / 2, f0_ + bw_ / 2
        return "fbw", f0_, bw_, fbw_, (f_l, f_u)

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        cm: CouplingMatrix,
        *,
        kind: str,
        f_edges: Tuple[Optional[float], Optional[float]] | None = None,
        f0: Optional[float] = None,
        bw: Optional[float] = None,
        fbw: Optional[float] = None,
        name: Optional[str] = None,
    ):
        if cm.topo.ports != 2:
            raise ValueError("Filter поддерживает только 2-портовые матрицы (ports == 2)")
        super().__init__(cm, name=name)

        self.kind = kind = kind.upper()
        if kind not in {"LP", "HP", "BP", "BR"}:
            raise ValueError("kind должен быть LP / HP / BP / BR")

        spec, self.f0, self.bw, self.fbw, self.f_edges = self._resolve_edges(kind, f_edges, f0, bw, fbw)
        self._spec = spec

    # ------------------------------------------------------- factory helpers
    # LP / HP
    @classmethod
    def lp(cls, cm, f_cut, *, unit: str = "Hz", **kw):
        return cls(cm, kind="LP", f0=_to_hz(f_cut, unit), **kw)

    @classmethod
    def hp(cls, cm, f_cut, *, unit: str = "Hz", **kw):
        return cls(cm, kind="HP", f0=_to_hz(f_cut, unit), **kw)

    # BP / BR (f0 + bw)
    @classmethod
    def bp(cls, cm, f0, bw, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BP", f0=_to_hz(f0, unit), bw=_to_hz(bw, unit), **kw)

    @classmethod
    def br(cls, cm, f0, bw, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BR", f0=_to_hz(f0, unit), bw=_to_hz(bw, unit), **kw)

    # BP / BR (edges)
    @classmethod
    def bp_edges(cls, cm, f_l, f_u, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BP", f_edges=(_to_hz(f_l, unit), _to_hz(f_u, unit)), **kw)

    @classmethod
    def br_edges(cls, cm, f_l, f_u, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BR", f_edges=(_to_hz(f_l, unit), _to_hz(f_u, unit)), **kw)

    # ------------------------------------------------------- _qu_scale
    def _qu_scale(self):
        """Коэффициент масштабирования добротности.

        LP/HP: qu = Q    -> scale = 1
        BP/BR: qu = Q*FBW -> scale = FBW
        """
        if self.kind in {"BP", "BR"}:
            return self.fbw
        return 1.0

    # ------------------------------------------------------- Ω-mapping
    def _omega(self, f_hz: np.ndarray) -> np.ndarray:
        """Преобразование **f → Ω** для заданного типа фильтра.

        Используется epsilon для избежания деления на ноль.
        """
        k = self.kind
        f = f_hz.astype(float, copy=False)
        eps = np.finfo(float).eps

        if k == "LP":  # Ω = f / f_cut
            cut = self.f_edges[0]
            return f / (cut + eps)

        if k == "HP":  # Ω = f_cut / f
            cut = self.f_edges[1]
            return (cut + eps) / np.maximum(f, eps)

        # BP / BR
        f0, bw = self.f0, self.bw
        f_safe = np.maximum(f, eps)
        ratio = f_safe / f0 - f0 / f_safe  # r − 1/r, r = f/f0
        if k == "BP":
            return (f0 / bw) * ratio
        # BR
        denom = np.where(ratio == 0, eps, ratio)
        return (bw / f0) / denom

    # -------------------------------------------------- Ω → f grid
    def freq_grid(
        self,
        omega,
        *,
        unit: str = "Hz",
        as_rf: bool = False,
    ):
        """Обратное преобразование **Ω → f**.

        Для **BP**/**BR** возвращается «двойной» массив частот (нижняя и верхняя ветви).
        """
        om = np.asarray(omega, dtype=float)
        k = self.kind
        eps = np.finfo(float).eps

        if k == "LP":  # Ω = f / f_cut
            f = om * self.f0
        elif k == "HP":  # Ω = f_cut / f
            with np.errstate(divide="ignore", invalid="ignore"):
                f = self.f0 / np.maximum(om, eps)
        else:
            bw, f0 = self.bw, self.f0
            sgn = np.sign(om)
            om_abs = np.abs(om) + eps
            with np.errstate(divide="ignore", invalid="ignore"):
                X = (bw / f0) * om_abs if k == "BP" else (bw / f0) / om_abs
                r = 0.5 * (X + np.sqrt(X ** 2 + 4.0))
            f_high = r * f0
            f_low = f0 / r
            f = np.where(sgn > 0, f_high, np.where(sgn < 0, f_low, f0))

        mult = rf.frequency.Frequency.multiplier_dict[unit.lower()]
        f_out = f / mult

        if not as_rf:
            return f_out

        # skrf.Frequency требует монотонность
        if np.any(np.diff(f_out) <= 0):
            f_sorted = np.sort(f_out)
        else:
            f_sorted = f_out
        return rf.Frequency.from_f(f_sorted, unit=unit.lower())

    # ------------------------------------------------ Touchstone helpers
    def _device_params(self) -> Dict[str, float | str]:
        """Параметры устройства без дублирования (используем исходную спецификацию)."""
        d: Dict[str, float | str] = {"device": "Filter", "kind": self.kind}
        if self.kind in {"LP", "HP"}:
            cut = self.f_edges[0] or self.f_edges[1]
            d["f_cut"] = cut
            return d

        # BP / BR
        d["f_center"] = self.f0
        if self._spec == "bw":
            d["bw"] = self.bw
        elif self._spec == "fbw":
            d["fbw"] = self.fbw
        else:  # edges
            f_l, f_u = self.f_edges
            d["f_lower"] = f_l
            d["f_upper"] = f_u
        return d

    @classmethod
    def _from_touchstone(
        cls,
        params: Mapping[str, float | str],
        cm: CouplingMatrix,
        *,
        kind: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Внутренний хелпер для `Device.from_touchstone`."""
        kind = (kind or params.get("kind", "BP")).upper()

        if kind in {"LP", "HP"}:
            return cls(cm, kind=kind, f0=float(params["f_cut"]), name=name)

        if "bw" in params:
            return cls(cm, kind=kind, f0=float(params["f_center"]), bw=float(params["bw"]), name=name)
        if "fbw" in params:
            return cls(cm, kind=kind, f0=float(params["f_center"]), fbw=float(params["fbw"]), name=name)
        return cls(cm, kind=kind, f_edges=(float(params["f_lower"]), float(params["f_upper"])), name=name)

    # ------------------------------------------------------------- repr
    def __repr__(self) -> str:
        base = super().__repr__()
        if self.kind in {"LP", "HP"}:
            cut = self.f_edges[0] or self.f_edges[1]
            extra = f", f_cut={cut/1e9:.3g} GHz"
        else:
            if self._spec == "bw":
                extra = f", f0={self.f0/1e9:.3g} GHz, bw={self.bw/1e6:.3g} MHz"
            elif self._spec == "fbw":
                extra = f", f0={self.f0/1e9:.3g} GHz, fbw={self.fbw*100:.2f} %"
            else:
                f_l, f_u = self.f_edges
                extra = f", f_l={f_l/1e9:.3g} GHz, f_u={f_u/1e9:.3g} GHz"
        return base[:-1] + extra + ")"


# ────────────────────────────────────────────────────────────────────────────
#                        Multiplexer (скрытая заготовка)
# ────────────────────────────────────────────────────────────────────────────

class Multiplexer(Device):
    """Каркас N-канального мультиплексора (один общий вход)."""

    __slots__ = ("channels",)

    def __init__(
        self,
        cm: CouplingMatrix,
        channels: Sequence[Filter],
        *,
        name: Optional[str] = None,
    ):
        super().__init__(cm, name=name)
        self.channels = list(channels)

    def _omega(self, f_hz: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Полная реализация Multiplexer появится в будущих версиях."
        )

    def _qu_scale(self):
        return 1.0  # пока используем ту же нормировку, что и LP/HP


__all__ = [
    "Device",
    "Filter",
    "Multiplexer",
]


