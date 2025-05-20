# mwlab/opt/objectives/selectors.py
"""
selectors.py
============
«Что» извлекаем из `rf.Network`.

В MVP предусмотрены два реальных селектора:

* **SMagSelector**          – |Sₘₙ| (модуль) в дБ или линейном масштабе;
* **AxialRatioSelector**    – «классическая» формула AR = |tan(…)| или
  AR(dB) = 20 log₁₀(ARₗᵢₙ).

Каждый селектор возвращает **два** вектора `freq, values`, чтобы
агрегатор мог, при желании, знать частотную сетку.
"""
from __future__ import annotations

from typing import Tuple, Sequence

import numpy as np
import skrf as rf

from .base import BaseSelector, register_selector


# ────────────────────────────────────────────────────────────────────────────
# helpers
def _band_mask(freq_ghz: np.ndarray, band: Tuple[float, float] | None):
    if band is None:
        return np.ones_like(freq_ghz, dtype=bool)
    f1, f2 = band
    return (freq_ghz >= f1) & (freq_ghz <= f2)


# ────────────────────────────────────────────────────────────────────────────
@register_selector(("s_mag", "sdb", "smag"))
class SMagSelector(BaseSelector):
    """Выбор |Sₘₙ| (магнитуда) в дБ или линейном масштабе.

    Parameters
    ----------
    m, n : int
        Порты **с единицы** (S₁₁→1,1).
    band : (float,float) | None
        Полоса частот (GHz). None → вся сетка.
    db : bool, default=True
        Если True — возвращаем в дБ (20 log₁₀).
    """

    def __init__(
        self,
        m: int,
        n: int,
        *,
        band: Tuple[float, float] | None = None,
        db: bool = True,
        name: str | None = None,
    ):
        self.m, self.n = m - 1, n - 1        # 0-based
        self.band = band
        self.db = db
        self.name = name or f"S{m}{n}_{'dB' if db else 'mag'}"

    # ---------------------------------------------------------------- call
    def __call__(self, net: rf.Network):
        freq_ghz = net.frequency.f / 1e9
        mask = _band_mask(freq_ghz, self.band)

        if self.db:
            vals = net.s_db[mask, self.m, self.n]
        else:
            vals = net.s_mag[mask, self.m, self.n]

        return freq_ghz[mask], vals


# ────────────────────────────────────────────────────────────────────────────
@register_selector(("axial_ratio", "ar"))
class AxialRatioSelector(BaseSelector):
    """Расчёт **Axial Ratio** по S₃₁ / S₄₁ (базовый селектор для RHCP-антенн).

    Формула (линейная):
        C = |S₃₁|,  D = |S₄₁|,
        A = arg(S₃₁), B = arg(S₄₁),
        AR = |tan(½·asin(2·C·D/(C²+D²)·sin(A-B)))|

    Parameters
    ----------
    band : (float,float) | None
        Диапазон частот, ГГц; None → вся сетка.
    db : bool, default=True
        Возвращать ли AR в dB (20 log₁₀).
    """

    def __init__(self, *, band=None, db=True):
        self.band = band
        self.db = db
        self.name = f"AxialRatio_{'dB' if db else 'lin'}"

    # ---------------------------------------------------------------- call
    def __call__(self, net: rf.Network):
        freq_ghz = net.frequency.f / 1e9
        mask = _band_mask(freq_ghz, self.band)

        S31 = net.s[mask, 2, 0]   # порты 3->1 (0-based)
        S41 = net.s[mask, 3, 0]

        C, D = np.abs(S31), np.abs(S41)
        A, B = np.angle(S31), np.angle(S41)

        num = 2 * C * D * np.sin(A - B)
        den = C**2 + D**2
        AR_lin = np.abs(np.tan(0.5 * np.arcsin(num / den)))

        vals = 20 * np.log10(AR_lin) if self.db else AR_lin
        return freq_ghz[mask], vals


# ────────────────────────────────────────────────────────────────────────────
# TODO-селекторы – заглушки
@register_selector("phase")
class PhaseSelector(BaseSelector):
    def __init__(self, *args, **kw): ...
    def __call__(self, net): raise NotImplementedError


@register_selector("group_delay")
class GroupDelaySelector(BaseSelector):
    def __init__(self, *args, **kw): ...
    def __call__(self, net): raise NotImplementedError
