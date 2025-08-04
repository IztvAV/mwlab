#mwlab/filters/devices.py
"""
mwlab.filters.devices
=====================

Высокоуровневый слой «устройств» (Device-level) поверх матрицы связи.

Что делает модуль
-----------------
* **Device** — абстрактный N-портовый прибор, рассчитывающий S-параметры по частоте *f*.
* **Filter** — 2-портовый одно-полосный фильтр (LP / HP / BP / BR) с отображениями Ω↔f.
* **Multiplexer** — заготовка N-канального мультиплексора (API совместим, реализация неполная).

Ключевые особенности (после обновления)
---------------------------------------
* **Ленивый импорт `scikit-rf`**: зависимость нужна только при работе с Touchstone
  и при запросе выдачи `rf.Frequency`. Если `skrf` не установлен, остальной функционал
  (расчёт S-параметров) полностью работоспособен.
* **Единая трактовка единиц**:
  - Все публичные методы, принимающие частоты, имеют параметр `unit` (по умолчанию `"Hz"`),
    который **интерпретирует входной числовой массив** (если на входе не `rf.Frequency`).
  - Если на вход подан `rf.Frequency`, он уже содержит частоты в Гц, и `unit` используется
    только как *оформление* (например, для Touchstone-файла), но на вычисления не влияет.
* **Новые Ω-методы**:
  - `Device.sparams_omega(omega, ...)` — прямой расчёт S(Ω), минуя обсуждение единиц.
  - `Filter.to_touchstone_omega(omega, f_unit="Hz", ...)` — строит Touchstone напрямую из Ω,
    сам переводя в *f* и подставляя нужные единицы частоты в выходном файле.
* **Чёткий контракт `Filter.freq_grid`**: параметр `unit` — это **единица результата**.
  Возвращаются либо числовые частоты в указанных единицах, либо `rf.Frequency` при `as_rf=True`.
* **Валидация для LP/HP**: если передан `f_edges` и оба края указаны (не `None`), это ошибка —
  для LP/HP нужен ровно один «край» (`f_cut`).
* **Документация**: докстринги уточнены (про единицы, батч-оси и устойчивость в окрестности Ω=0).

Зависимости
-----------
* `numpy` — вспомогательные операции и работа с сетками частот.
* `torch`  — все расчёты S-параметров выполняются средствами Torch.
* `scikit-rf` (необязательная) — только для Touchstone I/O и для возврата `rf.Frequency`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Mapping, Sequence, Tuple, Optional, Any

import numpy as np
import torch

from mwlab.filters.cm import CouplingMatrix
from mwlab.filters.cm_core import DEFAULT_DEVICE
from mwlab.io.touchstone import TouchstoneData


# ────────────────────────────────────────────────────────────────────────────
#                         Внутренние вспомогательные утилиты
# ────────────────────────────────────────────────────────────────────────────

# Локальный словарь множителей единиц (без зависимости от scikit-rf)
_UNIT_MULT = {
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}


def _to_hz(val: float | np.ndarray, unit: str) -> float | np.ndarray:
    """
    Переводит число/массив *val* из единиц `unit` ('Hz'/'kHz'/'MHz'/'GHz') в Гц.

    Бросает ValueError при неизвестной единице.
    """
    key = unit.lower()
    try:
        mult = _UNIT_MULT[key]
    except KeyError as err:
        valid = ", ".join(sorted(_UNIT_MULT.keys()))
        raise ValueError(f"Неизвестная единица частоты: {unit!r}. Допустимые: {valid}") from err
    return np.asarray(val, dtype=float) * mult


def _require_skrf():
    """
    Лениво импортирует scikit-rf.

    Возвращает модуль `skrf` или бросает информативный ImportError, если пакет не установлен.
    """
    try:
        import skrf as rf  # type: ignore
        return rf
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ImportError(
            "Эта операция требует установленный пакет 'scikit-rf'. "
            "Установите его, например: pip install scikit-rf"
        ) from err


def _as_frequency(f: Any, *, unit: str = "Hz") -> Tuple[np.ndarray, Optional[Any]]:
    """
    Приводит аргумент *f* к кортежу `(f_hz, freq_obj)`.

    Варианты:
    * Если *f* — числовой скаляр/массив, трактуется как частоты в единицах `unit`
      и конвертируется в Гц: `f_hz = np.asarray(f)*mult(unit)`. `freq_obj = None`.
    * Если *f* — `rf.Frequency`, извлекается `f.f` (Гц) и возвращается исходный
      объект как `freq_obj` (для последующего использования в Touchstone).

    Примечание: импорт `skrf` выполняется лениво. Если `skrf` не установлен,
    путь с `rf.Frequency` просто никогда не сработает (поскольку объект создать нельзя).
    """
    # Попытаться опознать rf.Frequency, если пакет установлен
    rf = None
    try:
        import skrf as _rf  # type: ignore
        rf = _rf
    except Exception:
        rf = None

    if rf is not None and isinstance(f, rf.Frequency):  # type: ignore
        f_hz = f.f.copy()
        return f_hz, f  # freq_obj = исходный Frequency

    # Числа → массив → перевод в Гц
    arr = np.asarray(f, dtype=float)
    if arr.ndim == 0:  # скаляр → (1,)
        arr = arr.reshape(1)
    f_hz = _to_hz(arr, unit=unit)
    return np.asarray(f_hz, dtype=float), None


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

    Особенности API
    ---------------
    * Методы, принимающие частоты, интерпретируют **числовые** массивы как частоты
      в единицах `unit` (по умолчанию Гц) и сами переводят их во внутренние Гц.
      Если передан `rf.Frequency`, он уже в Гц — вычисления используют его без
      преобразований (параметр `unit` влияет лишь на оформление вывода, например,
      в Touchstone).
    * Поддерживаются **batch-оси**: если вы передаёте f как массив формы
      `(..., F)`, ядро Torch выполнит вещание (broadcast) над ведущими осями.
      Частотная ось должна быть **последней**.
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
        """
        Физическая добротность резонаторов (скаляр или вектор) или None.

        Если `cm.qu` не задан, возвращает None. Иначе делит `cm.qu` на
        масштаб `self._qu_scale()` (для BP/BR — это FBW), возвращая физический Q.
        """
        if self.cm.qu is None:
            return None
        return np.asarray(self.cm.qu, dtype=float) / self._qu_scale()

    # ----------------------------------------------------- abstract methods
    @abstractmethod
    def _omega(self, f_hz: np.ndarray) -> np.ndarray:
        """Преобразование **f(Гц)** → **Ω** для данного устройства."""
        ...

    @abstractmethod
    def _qu_scale(self) -> float | np.ndarray:
        """
        Отношение qu / Q для устройства.

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
        f,
        *,
        unit: str = "Hz",
        device: str | torch.device = DEFAULT_DEVICE,
        method: str = "auto",
        fix_sign: bool = False,
    ) -> torch.Tensor:
        """
        Комплексная матрица **S(f)** в реальной шкале частот.

        Параметры
        ----------
        f : array-like | rf.Frequency
            Сетка частот. Если это числа — трактуются как частоты в единицах `unit`
            и конвертируются в Гц. Если это `rf.Frequency` — берётся `f.f` (Гц).
            Поддерживаются ведущие batch-оси; частотная ось должна быть последней.
        unit : {"Hz","kHz","MHz","GHz"}
            Единицы, *в которых заданы числовые частоты* (если f — числа).
            Для `rf.Frequency` не используется при вычислениях.
        device : torch.device | str
            Устройство вычислений (по умолчанию авто).
        method : {"auto","inv","solve"}
            Алгоритм обращения матрицы (см. `cm_core.solve_sparams`).
        fix_sign : bool
            Инвертировать ли знак S12/S21 для 2-портового случая.

        Returns
        -------
        torch.Tensor
            Тензор формы (..., F, P, P) complex64 (или (F,P,P), если без batch-осей).
        """
        f_hz, _ = _as_frequency(f, unit=unit)  # приводим к Гц
        omega = self._omega(f_hz)
        return self.cm.sparams(omega, device=device, method=method, fix_sign=fix_sign)

    def sparams_omega(
        self,
        omega,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        method: str = "auto",
        fix_sign: bool = False,
    ) -> torch.Tensor:
        """
        Прямой расчёт комплексной матрицы **S(Ω)**.

        Этот метод удобен, когда вы работаете в нормированной шкале и хотите
        избежать любых вопросов с единицами частоты.
        """
        omega_arr = np.asarray(omega, dtype=float)
        return self.cm.sparams(omega_arr, device=device, method=method, fix_sign=fix_sign)

    # -------------------------- Touchstone экспорт -----------------------
    def to_touchstone(
        self,
        f,
        *,
        unit: str = "Hz",
        device: str | torch.device = DEFAULT_DEVICE,
        method: str = "auto",
        fix_sign: bool = False,
    ) -> TouchstoneData:
        """
        Возвращает `TouchstoneData` с готовой сетью `skrf.Network`.

        Аргумент `f` трактуется так же, как в `sparams(...)`:
        - Если `f` — числа: это частоты в единицах `unit`; внутренние расчёты
          выполняются в Гц, а объект `rf.Frequency` для Touchstone формируется
          с лейблом `unit`.
        - Если `f` — `rf.Frequency`: используется **как есть** (в Гц) и
          определяет метки оси частоты в выходном файле.

        Все метаданные (параметры фильтра/матрицы связи) кладутся в `params`.
        """
        # 1) Посчитать S(f) консистентно с трактовкой sparams(...)
        S_t = self.sparams(f, unit=unit, device=device, method=method, fix_sign=fix_sign)
        S_np = S_t.detach().cpu().numpy()

        # 2) Подготовить объект rf.Frequency для Touchstone
        rf = _require_skrf()
        if isinstance(f, rf.Frequency):  # type: ignore
            freq = f
        else:
            arr = np.asarray(f, dtype=float)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            freq = rf.Frequency.from_f(arr, unit=unit.lower())

        # 3) Сборка Network и TouchstoneData
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
        """
        Восстанавливает объект-потомок `Device` из Touchstone-структуры.

        Ожидается, что `ts.params["device"]` либо не задан, либо равен "Filter".
        Для других устройств (в будущем) возможны специализированные конструкторы.
        """
        dev = str(ts.params.get("device", "")).lower()
        if dev and dev != "filter":
            raise ValueError(
                f"from_touchstone: ожидалось устройство 'Filter', получено {dev!r}"
            )

        cm = CouplingMatrix.from_dict(None, ts.params)  # topology inference
        return Filter._from_touchstone(ts.params, cm, kind=kind, name=name)

    # ---------------------------------------------------------------- params
    def _device_params(self) -> Dict[str, float | str]:
        """Параметры устройства для записи в Touchstone (переопределяется в наследниках)."""
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
    | kind  | допустимые параметры                        |
    |-------|---------------------------------------------|
    | LP/HP | `f_cut`                                     |
    | BP/BR | `f_edges=(f_l,f_u)` **или** `f0,bw` **или** `f0,fbw` |

    Пояснения:
    ----------
    * `fbw` — **относительная** полоса (доля), например `0.02` = 2 %.
    * Методы, принимающие частоты, поддерживают batch-оси в точности как ядро:
      частотная ось — последняя.
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
        """
        Унифицирует входные частотные аргументы.

        Возвращает кортеж `(spec, f0, bw, fbw, (f_l, f_u))`, где *spec* фиксирует,
        каким образом была задана полоса: "cut" | "bw" | "fbw" | "edges".
        """
        kind = kind.upper()

        # ---------- LP / HP ------------------------------------------------
        if kind in {"LP", "HP"}:
            if f_edges is not None:
                if len(f_edges) != 2:
                    raise ValueError("f_edges должен быть кортежем (low, high)")
                low, high = f_edges
                # Важно: оба края задавать нельзя — для LP/HP нужен ровно один край (f_cut)
                if (low is not None) and (high is not None):
                    raise ValueError("LP/HP: укажите только один край (f_cut), не оба.")
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
        return cls(cm, kind="LP", f0=float(_to_hz(f_cut, unit)), **kw)

    @classmethod
    def hp(cls, cm, f_cut, *, unit: str = "Hz", **kw):
        return cls(cm, kind="HP", f0=float(_to_hz(f_cut, unit)), **kw)

    # BP / BR (f0 + bw)
    @classmethod
    def bp(cls, cm, f0, bw, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BP", f0=float(_to_hz(f0, unit)), bw=float(_to_hz(bw, unit)), **kw)

    @classmethod
    def br(cls, cm, f0, bw, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BR", f0=float(_to_hz(f0, unit)), bw=float(_to_hz(bw, unit)), **kw)

    # BP / BR (edges)
    @classmethod
    def bp_edges(cls, cm, f_l, f_u, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BP", f_edges=(float(_to_hz(f_l, unit)), float(_to_hz(f_u, unit))), **kw)

    @classmethod
    def br_edges(cls, cm, f_l, f_u, *, unit: str = "Hz", **kw):
        return cls(cm, kind="BR", f_edges=(float(_to_hz(f_l, unit)), float(_to_hz(f_u, unit))), **kw)

    # ------------------------------------------------------- _qu_scale
    def _qu_scale(self):
        """
        Коэффициент масштабирования добротности.

        LP/HP: qu = Q    → scale = 1
        BP/BR: qu = Q*FBW → scale = FBW
        """
        if self.kind in {"BP", "BR"}:
            return self.fbw
        return 1.0

    # ------------------------------------------------------- Ω-mapping
    def _omega(self, f_hz: np.ndarray) -> np.ndarray:
        """
        Преобразование **f → Ω** для заданного типа фильтра.

        Численная защита: используется epsilon для избежания деления на ноль.
        """
        k = self.kind
        f = np.asarray(f_hz, dtype=float)
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
        """
        Обратное преобразование **Ω → f**.

        Параметр `unit` определяет **единицы результата**:
        функция возвращает массив частот в указанных единицах. Если `as_rf=True`,
        возвращается объект `rf.Frequency` с соответствующей единицей.

        Для **BP**/**BR** возвращаемая сетка является «двойной» в том смысле,
        что положительные и отрицательные Ω соответствуют верхней и нижней ветвям
        полосы (см. формулы ниже). Численная устойчивость обеспечивается
        защитой от делений на ноль вблизи Ω = 0.

        Примечание: если `as_rf=True`, для создания `rf.Frequency` выполняется
        ленивый импорт `scikit-rf`.
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

        # Привести к требуемым единицам вывода
        mult = _UNIT_MULT.get(unit.lower())
        if mult is None:
            valid = ", ".join(sorted(_UNIT_MULT.keys()))
            raise ValueError(f"Неизвестная единица частоты: {unit!r}. Допустимые: {valid}")
        f_out = f / mult  # делим, чтобы получить значения в выбранной единице

        if not as_rf:
            return f_out

        # Для rf.Frequency требуется монотонность; отсортируем при необходимости
        f_sorted = np.sort(f_out) if np.any(np.diff(f_out) <= 0) else f_out
        rf = _require_skrf()
        return rf.Frequency.from_f(f_sorted, unit=unit.lower())

    # ---------------------- Ω → Touchstone (удобный шорткат) --------------
    def to_touchstone_omega(
        self,
        omega,
        *,
        f_unit: str = "Hz",
        device: str | torch.device = DEFAULT_DEVICE,
        method: str = "auto",
        fix_sign: bool = False,
    ) -> TouchstoneData:
        """
        Удобный шорткат: строит Touchstone напрямую из сетки **Ω**.

        1) Переводит Ω → f в единицах `f_unit` (числовой массив).
        2) Вызывает `to_touchstone(f, unit=f_unit, ...)`.
        """
        f = self.freq_grid(omega, unit=f_unit, as_rf=False)
        return self.to_touchstone(f, unit=f_unit, device=device, method=method, fix_sign=fix_sign)

    # ------------------------------------------------ Touchstone helpers
    def _device_params(self) -> Dict[str, float | str]:
        """
        Параметры устройства без дублирования (используется исходная спецификация).

        Для LP/HP:
            {'device': 'Filter', 'kind': 'LP'|'HP', 'f_cut': <float>}
        Для BP/BR:
            {'device': 'Filter', 'kind': 'BP'|'BR', 'f_center': f0, 'bw'| 'fbw' | 'f_lower'/'f_upper': ...}
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
        kind: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Внутренний хелпер для `Device.from_touchstone`.

        Восстанавливает объект `Filter` по параметрам, сохранённым ранее через
        `to_touchstone()`: читает тип, центральную частоту и способ задания полосы.
        """
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
    """Каркас N-канального мультиплексора (один общий вход).

    Полная реализация будет добавлена в будущих версиях.
    """

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



