#mwlab/filters/devices.py
"""
mwlab.filters.devices
=====================
Высокоуровневый слой «устройств» (Device-level) поверх матрицы связи:

* **Device**  – абстрактное базовое устройство (N-порт, любое число звеньев);
* **Filter**  – 2-портовый одно-полосный фильтр (LP / HP / BP / BR);
* **Multiplexer** – заготовка N-канального мультиплексора (скрыта из __all__).

Модуль предоставляет:

* прямой расчёт S-матрицы устройства в шкале **f (Гц)**;
* преобразование сеток Ω ↔ f с учётом типа фильтра;
* круглую триаду «CouplingMatrix ↔ Filter ↔ Touchstone»;
* фабрики-шорткаты `Filter.lp / .hp / .bp / .br / .bp_edges / .br_edges`.

Все вычисления на NumPy; при вызове `cm.sparams` фильтр автоматически
делегирует работе с backend-ами (NumPy / Torch) в модуле
`mwlab.filters.cm`.

-------------------------------------------------------------------------------
Ограничения текущей версии
--------------------------
* Реализованы **только 2-портовые** фильтры; при попытке создать фильтр
  с `ports ≠ 2` возбуждается `ValueError`.
* Класс `Multiplexer` доступен, но не завершён и не экспортируется через
  `__all__`.

-------------------------------------------------------------------------------
```python
import numpy as np
from mwlab.filters.topologies import get_topology
from mwlab.filters.cm         import CouplingMatrix
from mwlab.filters.devices    import Filter

# 1) топология «folded» на 4 резонатора
topo = get_topology("folded", order=4)

cm = CouplingMatrix(
    topo,
    M_vals={
        "M1_2": 1.05, "M2_3": 0.90, "M3_4": 1.05, "M1_4": 0.25,
        "M1_5": 0.60, "M4_6": 0.80,          # связи порт-резонатор
    },
    Q=7_000,
)

# 2) полосовой фильтр: f0 = 2 ГГц, BW = 200 МГц
flt = Filter.bp(cm, f0=2.0, bw=0.2, unit="GHz", name="Demo-BP")

# 3) расчёт S-параметров
f = np.linspace(1.6e9, 2.4e9, 801)      # Гц
S = flt.sparams(f)                      # shape = (801, 2, 2)

# 4) сохранение в Touchstone
ts = flt.to_touchstone(f, unit="Hz")
ts.save("demo_bp.s2p")
```
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import skrf as rf

from mwlab.filters.cm import CouplingMatrix
from mwlab.filters.topologies import Topology
from mwlab.io.touchstone import TouchstoneData

# ────────────────────────────────────────────────────────────────────────────
#                               helpers
# ────────────────────────────────────────────────────────────────────────────


def _as_frequency(
    f, *, unit: str = "Hz"
) -> Tuple[np.ndarray, rf.Frequency | None]:
    """
    Приводит *f* к ``ndarray`` в Гц.

    Если передан объект ``rf.Frequency`` –
    возвращает ``(f_numpy, rf.Frequency)`` для сохранения исходных единиц.
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
            f"Неизвестная единица частоты: {unit!r}. "
            f"Допустимые единицы: {valid}"
        ) from err
    return float(val) * mult


# ────────────────────────────────────────────────────────────────────────────
#                                   Device
# ────────────────────────────────────────────────────────────────────────────


class Device(ABC):
    """Абстрактное устройство, описываемое матрицей связи."""

    __slots__ = ("cm", "name")

    def __init__(self, cm: CouplingMatrix, *, name: str | None = None):
        self.cm = cm
        self.name = str(name) if name is not None else None

    # ---------------------------------------------------------------- props
    @property
    def order(self) -> int:  # noqa: D401
        return self.cm.topo.order

    @property
    def ports(self) -> int:  # noqa: D401
        return self.cm.topo.ports

    # ---------------------------------------------------------------- Ω-map
    @abstractmethod
    def _omega(self, f_hz: np.ndarray) -> np.ndarray: ...  # noqa: D401

    # ---------------------------------------------------------------- API – S-параметры
    def sparams(
        self,
        f_hz,
        *,
        backend: str = "numpy",
        method: str = "auto",
    ):
        """
        Комплексная матрица **S(f)** в реальной шкале частот.

        Параметры
        ----------
        f_hz : array-like | rf.Frequency
            Сетка частот **в Гц** или объект `skrf.Frequency`.
        backend : {"numpy","torch"}
            На каком backend-е считать `CouplingMatrix.sparams`.
        method : {"auto","inv","solve"}
            Алгоритм обращения матрицы (см. `cm_sparams`).
        """
        f_arr, _ = _as_frequency(f_hz)
        omega = self._omega(f_arr)
        return self.cm.sparams(omega, backend=backend, method=method)

    # -------------------------- Touchstone экспорт -----------------------
    def to_touchstone(
        self,
        f_hz,
        *,
        unit: str = "Hz",
        backend: str = "numpy",
        method: str = "auto",
    ) -> TouchstoneData:
        """
        Возвращает `TouchstoneData` с готовой сетью `skrf.Network`
        и полным словарём метаданных.
        """
        f_arr, freq_obj = _as_frequency(f_hz)
        S = self.sparams(f_arr, backend=backend, method=method)

        freq = freq_obj or rf.Frequency.from_f(f_arr, unit=unit)
        net = rf.Network(frequency=freq, s=S)

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
        """
        Восстанавливает объект-потомок `Device` из Touchstone-структуры.
        """
        dev = ts.params.get("device", "").lower()
        if dev and dev != "filter":
            raise ValueError(
                f"from_touchstone: ожидалось устройство 'Filter', получено {dev!r}"
            )

        cm = CouplingMatrix.from_dict(None, ts.params)  # topology inference
        return Filter._from_touchstone(ts.params, cm, kind=kind, name=name)

    # ---------------------------------------------------------------- params
    def _device_params(self) -> Dict[str, float | str]:
        """Что писать в `TouchstoneData.params` (переопределяется потомками)."""
        return {"device": type(self).__name__}

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # noqa: D401
        extra = f", name={self.name!r}" if self.name else ""
        return (
            f"{type(self).__name__}(order={self.order}, ports={self.ports}{extra})"
        )


# ────────────────────────────────────────────────────────────────────────────
#                                  Filter
# ────────────────────────────────────────────────────────────────────────────


class Filter(Device):
    """
    Универсальный **2-портовый** одно-полосный фильтр.

    Поддерживаемые типы (`kind`):

    * LP — *low-pass*
    * HP — *high-pass*
    * BP — *band-pass*
    * BR — *band-reject*

    Задание полосы:

    | kind | допустимые параметры |
    |------|----------------------|
    | LP / HP | `f_cut` |
    | BP / BR | `f_edges=(f_l,f_u)` **или** `f0,bw` **или** `f0,fbw` |

    Фабрики-шорткаты: `Filter.lp / hp / bp / br / bp_edges / br_edges`.
    """

    __slots__ = (
        "kind",
        "f_edges",
        "f0",
        "bw",
        "fbw",
        "_spec",  # как была задана полоса: "cut" | "bw" | "fbw" | "edges"
    )

    # ------------------------------------------------------- internal utils
    @staticmethod
    def _resolve_edges(
        kind: str,
        f_edges,
        f0,
        bw,
        fbw,
    ) -> Tuple[str, float, float | None, float | None, Tuple[float | None, float | None]]:
        """
        Унифицирует входные частотные аргументы.

        Возвращает кортеж
        ``(spec, f0, bw, fbw, (f_l,f_u))``, где *spec* показывает,
        какая именно спецификация была использована.
        """
        kind = kind.upper()
        # ---------- LP / HP ------------------------------------------------
        if kind in {"LP", "HP"}:
            # ── принимаем либо f0, либо f_edges = (cut,None)/(None,cut) ──
            if f_edges is not None:
                if len(f_edges) != 2:
                    raise ValueError("f_edges должен быть кортежем (low,high)")
                low, high = f_edges
                if kind == "LP":
                    f_cut = low if low is not None else high
                else:
                    f_cut = high if high is not None else low
            else:
                f_cut = f0

            if f_cut is None:
                raise ValueError("LP/HP: необходимо указать f_cut (f0 или f_edges)")

            # (f_l,f_u) : LP → (f_cut,None),  HP → (None,f_cut)
            edges = (f_cut, None) if kind == "LP" else (None, f_cut)
            return "cut", float(f_cut), None, None, edges

        # ---------- BP / BR ------------------------------------------------
        spec_cnt = sum(
            bool(x)
            for x in (
                f_edges is not None,
                f0 is not None and bw is not None,
                f0 is not None and fbw is not None,
            )
        )
        if spec_cnt != 1:
            raise ValueError(
                "BP/BR: задайте ровно одну комбинацию "
                "(f_edges) | (f0+bw) | (f0+fbw)"
            )

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
            fbw_ = bw_ / f0_
            f_l, f_u = f0_ - bw_ / 2, f0_ + bw_ / 2
            return "bw", f0_, bw_, fbw_, (f_l, f_u)

        # (c) f0 + fbw
        if f0 is None:
            raise ValueError("требуется указать f0 вместе с fbw")
        f0_ = float(f0)
        fbw_ = float(fbw)
        bw_ = fbw_ * f0_
        f_l, f_u = f0_ - bw_ / 2, f0_ + bw_ / 2
        return "fbw", f0_, bw_, fbw_, (f_l, f_u)

        # should never reach here
        raise RuntimeError("unreachable")

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        cm: CouplingMatrix,
        *,
        kind: str,
        f_edges: Tuple[float | None, float | None] | None = None,
        f0: float | None = None,
        bw: float | None = None,
        fbw: float | None = None,
        name: str | None = None,
    ):
        # ограничение порта (в будущем можно ослабить)
        if cm.topo.ports != 2:
            raise ValueError("Filter поддерживает только 2-портовые матрицы (ports == 2)")

        super().__init__(cm, name=name)

        self.kind = kind = kind.upper()
        if kind not in {"LP", "HP", "BP", "BR"}:
            raise ValueError("kind должен быть LP / HP / BP / BR")

        spec, self.f0, self.bw, self.fbw, self.f_edges = self._resolve_edges(
            kind, f_edges, f0, bw, fbw
        )
        self._spec = spec  # запоминаем, чтобы корректно экспортировать params

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
        return cls(
            cm,
            kind="BP",
            f_edges=(_to_hz(f_l, unit), _to_hz(f_u, unit)),
            **kw,
        )

    @classmethod
    def br_edges(cls, cm, f_l, f_u, *, unit: str = "Hz", **kw):
        return cls(
            cm,
            kind="BR",
            f_edges=(_to_hz(f_l, unit), _to_hz(f_u, unit)),
            **kw,
        )

    # ------------------------------------------------------- Ω-mapping
    def _omega(self, f_hz: np.ndarray) -> np.ndarray:  # noqa: D401
        """
        Преобразование **f → Ω** для заданного типа фильтра.

        *Важное замечание.*  Чтобы избежать деления на ноль или
        отрицательные частоты, применяем ``eps = np.finfo(float).eps``.
        """
        k = self.kind
        f = f_hz.astype(float, copy=False)
        eps = np.finfo(float).eps

        if k == "LP":  # Ω = f / f_cut
            return f / (self.f_edges[0] + eps)

        if k == "HP":  # Ω = f_cut / f
            return (self.f_edges[1] + eps) / np.maximum(f, eps)

        # BP / BR
        f0, bw = self.f0, self.bw
        f_safe = np.maximum(f, eps)
        ratio = f_safe / f0 - f0 / f_safe  # r − 1/r, r = f/f0
        if k == "BP":
            return (f0 / bw) * ratio
        return (bw / f0) / np.where(ratio == 0, eps, ratio)  # BR

    # -------------------------------------------------- Ω → f grid
    def freq_grid(
            self,
            omega,
            *,
            unit: str = "Hz",
            as_rf: bool = False,
    ):
        """
        Обратное преобразование **Ω → f**.

        Parameters
        ----------
        omega : array-like
            Нормированная частота(ы) Ω.
        unit : str, default "Hz"
            Единицы результата.
        as_rf : bool, default False
            *True* → вернуть `skrf.Frequency`, иначе `ndarray`.

        Для **BP** и **BR** метод всегда возвращает *двойной* монотонно
        возрастающий массив частот, объединяя ветви ниже и выше f₀ и
        убирая возможный дубликат в точке стыка.
        """
        om = np.asarray(omega, dtype=float)
        k = self.kind
        eps = np.finfo(float).eps  # защита от деления на 0

        # ------------------------------------------------------------------ LP / HP
        if k == "LP":  # Ω = f / f_cut
            f = om * self.f0
        elif k == "HP":  # Ω = f_cut / f
            with np.errstate(divide="ignore", invalid="ignore"):
                f = self.f0 / np.maximum(om, eps)
        # ------------------------------------------------------------------ BP / BR
        else:
            bw, f0 = self.bw, self.f0
            sgn = np.sign(om)  # –1, 0, +1
            om_abs = np.abs(om) + eps

            with np.errstate(divide="ignore", invalid="ignore"):
                X = (bw / f0) * om_abs if k == "BP" else (bw / f0) / om_abs
                r = 0.5 * (X + np.sqrt(X ** 2 + 4.0))

            f_high = r * f0  # Ω > 0
            f_low = f0 / r   # Ω < 0
            f = np.where(sgn > 0, f_high,
                         np.where(sgn < 0, f_low, f0))  # Ω = 0 → f₀

        # ------------------------------------------------------------------ to unit
        mult = rf.frequency.Frequency.multiplier_dict[unit.lower()]
        f_out = f / mult

        if not as_rf:
            return f_out

        # `skrf.Frequency` требует монотонность → страхуемся сортировкой копии
        if np.any(np.diff(f_out) <= 0):
            f_sorted = np.sort(f_out)
        else:
            f_sorted = f_out

        return rf.Frequency.from_f(f_sorted, unit=unit.lower())

    # ------------------------------------------------ Touchstone helpers
    def _device_params(self) -> Dict[str, float | str]:
        """
        Формирует словарь параметров **только** из исходной спецификации
        (чтобы избежать дублирования bw/fbw/edges).
        """
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
        kind: str | None = None,
        name: str | None = None,
    ):
        """Внутренний хелпер для `Device.from_touchstone`."""
        kind = (kind or params.get("kind", "BP")).upper()

        # --- LP / HP
        if kind in {"LP", "HP"}:
            return cls(cm, kind=kind, f0=float(params["f_cut"]), name=name)

        # --- BP / BR
        if "bw" in params:  # исходная спецификация BW
            return cls(
                cm,
                kind=kind,
                f0=float(params["f_center"]),
                bw=float(params["bw"]),
                name=name,
            )
        if "fbw" in params:  # исходная спецификация FBW
            return cls(
                cm,
                kind=kind,
                f0=float(params["f_center"]),
                fbw=float(params["fbw"]),
                name=name,
            )
        # границы
        return cls(
            cm,
            kind=kind,
            f_edges=(float(params["f_lower"]), float(params["f_upper"])),
            name=name,
        )

    # ------------------------------------------------------------- repr
    def __repr__(self):  # noqa: D401
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
        name: str | None = None,
    ):
        super().__init__(cm, name=name)
        self.channels = list(channels)

    # пока не реализовано
    def _omega(self, f_hz: np.ndarray) -> np.ndarray:  # noqa: D401
        raise NotImplementedError(
            "Полная реализация Multiplexer появится в будущих версиях."
        )


