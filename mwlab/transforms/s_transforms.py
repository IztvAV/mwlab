# transforms/s_transforms.py
"""
mwlab.transforms.s_transforms
-----------------------------
Набор «S‑трансформов» – функций, которые принимают `skrf.Network`
и возвращают новый (или модифицированный) `skrf.Network`.

* S_Crop              – обрезка по диапазону частот
* S_Resample          – пересэмплирование
* S_AddNoise          – добавление шума (амплитуда+фаза **или** Re/Im)
* S_PhaseShiftDelay   – случайная/фиксированная задержка Δτ (ps)
* S_PhaseShiftAngle   – постоянный фазовый сдвиг Δφ (град)

Все классы можно свободно комбинировать через TComposite:

    tf = TComposite([
        S_Crop(1e9, 10e9),
        S_AddNoise(sigma_db=(0.05, 0.02), sigma_deg=0.5),
        S_PhaseShiftDelay(tau_ps=(-10, 5)),
    ])

    tf = S_PhaseShiftAngle(phi_deg=15)   # фиксированный 15°

    tf = S_PhaseShiftAngle(phi_deg=(15, 5))   # случайное распределение N(μ, σ) с μ = 15° и σ = 5°
"""

from __future__ import annotations

from typing import Callable, Tuple, Union

import numpy as np
import skrf


# --------------------------------------------------------------------- helpers
def _make_sampler(
    value: Union[float, Tuple[float, float], Callable[[], float]],
    name: str,
) -> Callable[[], float]:
    """
    Универсальный «привод к сэмплеру».

    • число  → lambda: число           (фиксированное значение)
    • (μ, σ) → lambda: N(μ, σ)         (нормальное распределение)
    • callable → возвращается как есть

    Используется в трансформах для параметров, которые могут быть
    либо фиксированными, либо случайными.
    """
    if callable(value):
        return value

    # кортеж из двух чисел → нормальное распределение
    if isinstance(value, tuple) and len(value) == 2:
        mu, sigma = value
        return lambda: float(np.random.normal(mu, sigma))

    # одиночное число
    if isinstance(value, (int, float)):
        return lambda: float(value)

    raise TypeError(
        f"{name}: ожидается число, (mu, sigma) или callable, получено {type(value)}"
    )


# ===================================================================== основные
class S_Crop:
    """
    Обрезает сеть по диапазону частот [f_start, f_stop].

    Parameters
    ----------
    f_start, f_stop : float
        Границы диапазона в *unit*.
    unit : str
        'Hz', 'kHz', 'MHz', 'GHz'.  По‑умолчанию 'Hz'.
    """

    def __init__(self, f_start: float, f_stop: float, unit: str = "Hz"):
        if f_start >= f_stop:
            raise ValueError("f_start должен быть меньше f_stop")
        self.f_start = f_start
        self.f_stop = f_stop
        self.unit = unit

    def __call__(self, network: skrf.Network) -> skrf.Network:
        return network.cropped(
            f_start=self.f_start,
            f_stop=self.f_stop,
            unit=self.unit,
        )


class S_Resample:
    """
    Интерполирует сеть до нового количества точек *или* набора частот.

    Parameters
    ----------
    freq_or_n : int | array‑like | skrf.Frequency
        • int        – новое число точек (равномерная сетка)
        • array      – конкретные частоты (в тех же единицах, что у сети)
        • Frequency  – готовый объект
    **kwargs      : передаются в `scipy.interpolate.interp1d`
    """

    def __init__(self, freq_or_n, **kwargs):
        self.freq_or_n = freq_or_n
        self.kwargs = kwargs

    def __call__(self, network: skrf.Network) -> skrf.Network:
        ntwk = network.copy()
        arg = self.freq_or_n

        # array‑like → переводим в Гц
        if isinstance(arg, (list, tuple, np.ndarray)):
            unit = network.frequency.unit or "Hz"
            mult = skrf.frequency.Frequency.multiplier_dict[unit.lower()]
            arg = np.asarray(arg) * mult

        ntwk.resample(arg, **self.kwargs)
        return ntwk


# ================================================================== аугментация
class S_AddNoise:
    """
    Добавляет случайный шум к S‑параметрам.

    *По‑умолчанию* искажаются модуль (в дБ) и фаза (в градусах).

    Parameters
    ----------
    sigma_db  : float | (μ, σ) | callable
        σ амплитуды (дБ).  (μ,σ) – нормальное распределение.
    sigma_deg : float | (μ, σ) | callable
        σ фазы (градусы).
    cartesian : bool, optional
        True  → шум добавляется к Re/Im (N(0,σ)).
        False → модуль/фаза.  [default=False]
    """

    def __init__(
        self,
        *,
        sigma_db: Union[float, Tuple[float, float], Callable[[], float]] = 0.05,
        sigma_deg: Union[float, Tuple[float, float], Callable[[], float]] = 0.5,
        cartesian: bool = False,
    ):
        self.sigma_db_sampler = _make_sampler(sigma_db, "sigma_db")
        self.sigma_deg_sampler = _make_sampler(sigma_deg, "sigma_deg")
        self.cartesian = cartesian

    def __call__(self, net: skrf.Network) -> skrf.Network:
        ntwk = net.copy()
        s = ntwk.s

        if self.cartesian:
            sigma = self.sigma_db_sampler()  # трактуем как лин σ
            noise = (
                np.random.normal(0, sigma, size=s.shape)
                + 1j * np.random.normal(0, sigma, size=s.shape)
            )
            ntwk.s = s + noise
        else:
            σ_db = self.sigma_db_sampler()
            σ_deg = self.sigma_deg_sampler()
            mag_noise = 10 ** (np.random.normal(0, σ_db, size=s.shape) / 20)
            phase_noise = np.deg2rad(
                np.random.normal(0, σ_deg, size=s.shape)
            )
            ntwk.s = s * mag_noise * np.exp(1j * phase_noise)

        return ntwk


class S_PhaseShiftDelay:
    """
    Общий фазовый сдвиг, вызванный задержкой кабеля Δτ.

    Parameters
    ----------
    tau_ps : float | (μ, σ) | callable
        Задержка в пикосекундах.  Формат как в _make_sampler().
    """

    def __init__(
        self,
        tau_ps: Union[float, Tuple[float, float], Callable[[], float]] = 0.0,
    ):
        self.tau_sampler = _make_sampler(tau_ps, "tau_ps")

    def __call__(self, net: skrf.Network) -> skrf.Network:
        ntwk = net.copy()
        tau = self.tau_sampler() * 1e-12  # ps → s
        phase = np.exp(-1j * 2 * np.pi * ntwk.f * tau)
        ntwk.s *= phase[:, None, None]
        return ntwk


class S_PhaseShiftAngle:
    """
    Постоянное смещение фазы (одинаковое на всех частотах).

    Parameters
    ----------
    phi_deg : float | (μ, σ) | callable
        Фаза в градусах.  Можно задавать распределение.
    """

    def __init__(
        self,
        phi_deg: Union[float, Tuple[float, float], Callable[[], float]] = 0.0,
    ):
        self.phi_sampler = _make_sampler(phi_deg, "phi_deg")

    def __call__(self, net: skrf.Network) -> skrf.Network:
        ntwk = net.copy()
        phi = np.deg2rad(self.phi_sampler())
        ntwk.s *= np.exp(1j * phi)
        return ntwk


# --------------------------------------------------------------------- экспорт
__all__ = [
    "S_Crop",
    "S_Resample",
    "S_AddNoise",
    "S_PhaseShiftDelay",
    "S_PhaseShiftAngle",
]
