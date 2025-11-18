import re

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy.polynomial.polynomial import polyfit

import common
import configs
from filters import SamplerTypes, MWFilter, CouplingMatrix
from filters.codecs import MWFilterTouchstoneCodec
from mwlab import TouchstoneDataset
from matplotlib.widgets import CheckButtons
import skrf as rf
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import cauchy_method


def _wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def _fit_gvz_inverse_square(w, tau, mask=None, eps=1e-2):
    def model_tau(omega, a2, C):
        return -a2 / omega**2 - C

    w = np.asarray(w).ravel()
    tau = np.asarray(tau).ravel()
    full_mask = (np.abs(w) > eps)
    if mask is not None:
        full_mask &= mask

    w_fit = w[full_mask]
    tau_fit = tau[full_mask]

    if len(w_fit) < 3:
        raise ValueError("Недостаточно точек для аппроксимации")

    try:
        popt, _ = curve_fit(model_tau, w_fit, tau_fit, p0=(1.0, 0.0), maxfev=10000)
        a2, C = popt
    except Exception as e:
        print("Ошибка при curve_fit:", e)
        a2, C = 0.0, 0.0

    return a2, C


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_gvz_hyperbola_one_side(
    w,
    tau,
    side="right",        # "right" → w > 0, "left" → w < 0
    w_start=None,        # от какого |w| начинать аппроксимацию (обязательно)
    w_end=None,          # до какого |w|, если None — до края данных
    eps_zero=1e-6,       # защита от деления на ноль
    plot=True,
    tau_ylim=None,       # например (-100, 100)
):
    """
    Аппроксимация групповой задержки гиперболой:

        tau(w) ≈ -a / w^2 - C

    на одной стороне по частоте (w>0 или w<0).

    Параметры
    ---------
    w : array_like
        Нормированные частоты (1D).
    tau : array_like
        Групповая задержка (1D, той же длины).
    side : {"right", "left"}
        "right"  → аппроксимируем участок w > 0,
        "left"   → аппроксимируем участок w < 0.
    w_start : float
        От какого |w| начинать аппроксимацию на выбранной стороне.
    w_end : float or None
        До какого |w| выполнять аппроксимацию. Если None — до крайнего
        значения по соответствующей стороне.
    eps_zero : float
        Минимальное |w|, чтобы не делить совсем на ноль.
    plot : bool
        Строить ли график.
    tau_ylim : tuple or None
        Лимиты по оси Y для графика ГВЗ, например (-100, 100).

    Возвращает
    ----------
    a, C : float
        Коэффициенты гиперболы: tau(w) ≈ -a / w^2 - C.
    info : dict
        Вспомогательная информация (RMSE, MAE, маска и т.п.).
    """
    w = np.asarray(w, float).ravel()
    tau = np.asarray(tau, float).ravel()
    assert w.shape == tau.shape, "w и tau должны иметь одинаковую длину"

    if w_start is None:
        raise ValueError("Нужно задать w_start (от какой |w| начинать аппроксимацию).")

    # выбор стороны
    if side == "right":
        mask_side = w > 0
        w_min_side = max(w_start, eps_zero)
        w_max_side = np.max(w[mask_side]) if w_end is None else w_end
        mask_range = (w >= w_min_side) & (w <= w_max_side)
    elif side == "left":
        mask_side = w < 0
        # для левой стороны берём w ∈ [-w_end, -w_start]
        if w_end is None:
            w_min_side = np.min(w[mask_side])
        else:
            w_min_side = -w_end
        w_max_side = -max(w_start, eps_zero)
        mask_range = (w >= w_min_side) & (w <= w_max_side)
    else:
        raise ValueError("side должен быть 'right' или 'left'.")

    mask = mask_side & mask_range & (np.abs(w) > eps_zero)

    if np.count_nonzero(mask) < 3:
        raise ValueError("Недостаточно точек в выбранной области для аппроксимации (нужно ≥ 3).")

    w_fit = w[mask]
    tau_fit = tau[mask]

    # модель: -a / w^2 - C
    def hyper_tau(w_local, a, C):
        return -a / (w_local**2) - C

    # начальное приближение
    a0 = 1.0
    C0 = float(np.mean(-tau_fit))  # грубый старт
    p0 = (a0, C0)

    popt, pcov = curve_fit(hyper_tau, w_fit, tau_fit, p0=p0, maxfev=20000)
    a_fit, C_fit = popt

    tau_pred = hyper_tau(w_fit, a_fit, C_fit)
    err = tau_fit - tau_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    print("=== Аппроксимация ГВЗ гиперболой -a/w^2 - C ===")
    print(f"Сторона:       {side}")
    if side == "right":
        print(f"Область fit:   w ∈ [{w_min_side:.3f}, {w_max_side:.3f}]")
    else:
        print(f"Область fit:   w ∈ [{w_min_side:.3f}, {w_max_side:.3f}] (левая сторона)")
    print(f"a (гипербола): {a_fit:.6e}")
    print(f"C (константа): {C_fit:.6e}")
    print(f"RMSE:          {rmse:.6e}")
    print(f"MAE:           {mae:.6e}")

    if plot:
        w_plot = np.linspace(w_fit.min(), w_fit.max(), 500)
        tau_plot = hyper_tau(w_plot, a_fit, C_fit)

        plt.figure(figsize=(8, 4))
        # вся ГВЗ
        plt.plot(w, tau, ".", alpha=0.3, label="ГВЗ (все точки)")
        # фитируемый участок
        plt.plot(w_fit, tau_fit, "o", label="Точки fit")
        # гипербола
        plt.plot(w_plot, tau_plot, "-", label="Гипербола -a/w² - C")

        # границы области
        plt.axvline(w_min_side, color="k", linestyle="--", alpha=0.4)
        plt.axvline(w_max_side, color="k", linestyle="--", alpha=0.4)

        plt.xlabel("Нормированная частота w")
        plt.ylabel("Групповая задержка τ(w)")
        plt.title("Аппроксимация ГВЗ гиперболой на одной стороне")
        if tau_ylim is not None:
            plt.ylim(*tau_ylim)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    info = {
        "a": a_fit,
        "C": C_fit,
        "rmse": rmse,
        "mae": mae,
        "mask": mask,
        "w_fit": w_fit,
        "tau_fit": tau_fit,
    }
    return a_fit, C_fit, info

def fit_phase_hyperbola_one_side(
    w,
    S,
    side="right",        # "right" → w > 0, "left" → w < 0
    w_start=None,        # от какой |w| начинать аппроксимацию (обязателен)
    w_end=None,          # до какой |w| (опционально)
    plot=True,
):
    """
    Аппроксимация wrapped фазы одной гиперболой вида:
        phi(w) ≈ C + A / w
    + построение графика.

    Параметры
    ---------
    w : array_like
        Нормированные частоты (1D).
    S : array_like
        Комплексный вектор S-параметра (той же длины, что и w).
    side : {"right", "left"}
        "right"  → аппроксимируем на участке w > 0,
        "left"   → аппроксимируем на участке w < 0.
    w_start : float
        От какого |w| начинать аппроксимацию на выбранной стороне.
        Для side="right": область w ∈ [w_start, w_end]
        Для side="left" : область w ∈ [-w_end, -w_start]
        (если w_end не задан, то до края данных).
    w_end : float or None
        До какого |w| аппроксимировать. Если None — до максимума по соответствующей стороне.
    plot : bool
        Строить ли график фазы и гиперболы.

    Возвращает
    ----------
    A, C : float
        Коэффициенты гиперболы phi(w) ≈ C + A / w.
    info : dict
        Служебная информация: RMSE, MAE, маска точек и т.п.
    """
    w = np.asarray(w, float).ravel()
    S = np.asarray(S, complex).ravel()
    assert w.shape == S.shape, "w и S должны иметь одинаковую длину"

    if w_start is None:
        raise ValueError("Нужно задать w_start (от какой |w| начинать аппроксимацию).")

    # wrapped фаза
    phi = np.angle(S)  # в радианах

    # Выбор стороны
    if side == "right":
        mask_side = w > 0
        w_min_side = w_start
        w_max_side = np.max(w[mask_side]) if w_end is None else w_end
        mask_range = (w >= w_min_side) & (w <= w_max_side)
    elif side == "left":
        mask_side = w < 0
        # для левой стороны берём w ∈ [-w_end, -w_start]
        w_min_side = -w_end if w_end is not None else np.min(w[mask_side])
        w_max_side = -w_start
        mask_range = (w >= w_min_side) & (w <= w_max_side)
    else:
        raise ValueError("side должен быть 'right' или 'left'.")

    mask = mask_side & mask_range

    if np.count_nonzero(mask) < 3:
        raise ValueError("Недостаточно точек в выбранной области для аппроксимации (нужно ≥ 3).")

    w_fit = w[mask]
    phi_fit = phi[mask]

    # Модель гиперболы: C + A / w
    def hyperbola_model(w_local, A, C):
        return -C - A / w_local*2

    # Подбор A, C по МНК (curve_fit)
    # Начальное приближение: A=0, C = среднее по фазе
    p0 = (0.0, float(np.mean(phi_fit)))
    popt, pcov = curve_fit(hyperbola_model, w_fit, phi_fit, p0=p0, maxfev=20000)
    A_fit, C_fit = popt

    # Предсказание и метрики качества
    phi_pred = hyperbola_model(w_fit, A_fit, C_fit)
    err = phi_fit - phi_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    # Лог в радианах и градусах
    print("=== Результаты аппроксимации wrapped фазы гиперболой + константой ===")
    print(f"Сторона:        {side}")
    if side == "right":
        print(f"Область fit:    w ∈ [{w_min_side:.3f}, {w_max_side:.3f}]")
    else:
        print(f"Область fit:    w ∈ [{w_min_side:.3f}, {w_max_side:.3f}] (левая сторона)")
    print(f"A (гипербола):  {A_fit:.6e} (рад * w)")
    print(f"C (константа):  {C_fit:.6e} рад  ({np.degrees(C_fit):.3f}°)")
    print(f"RMSE:           {rmse:.6e} рад  ({np.degrees(rmse):.3f}°)")
    print(f"MAE:            {mae:.6e} рад  ({np.degrees(mae):.3f}°)")

    # График
    if plot:
        w_plot = np.linspace(w_fit.min(), w_fit.max(), 500)
        phi_plot = hyperbola_model(w_plot, A_fit, C_fit)

        plt.figure(figsize=(8, 4))
        # вся фаза
        plt.plot(w, np.degrees(phi), ".", alpha=0.3, label="ФЧХ (wrapped, все точки)")
        # точки, по которым фитировали
        plt.plot(w_fit, np.degrees(phi_fit), "o", label="Точки fit (wrapped)")
        # гипербола
        plt.plot(w_plot, np.degrees(phi_plot), "-", label="Гипербола + C (fit)")

        # вертикальные границы области fit
        plt.axvline(w_min_side, color="k", linestyle="--", alpha=0.4)
        plt.axvline(w_max_side, color="k", linestyle="--", alpha=0.4)

        plt.xlabel("Нормированная частота w")
        plt.ylabel("Фаза, градусы")
        plt.title("Аппроксимация wrapped ФЧХ гиперболой + константой (одна сторона)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    info = {
        "A": A_fit,
        "C": C_fit,
        "rmse_rad": rmse,
        "mae_rad": mae,
        "rmse_deg": np.degrees(rmse),
        "mae_deg": np.degrees(mae),
        "mask": mask,
        "w_fit": w_fit,
        "phi_fit": phi_fit,
    }
    return A_fit, C_fit, info



def _angle_diff(a, b):
    """Кратчайшая разность углов a и b (рад) с учётом 2π."""
    return _wrap_to_pi(a - b)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def _wrap_to_pi(x):
    """Заворачивает угол в (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _angle_diff(a, b):
    """Кратчайшая разность углов a и b (рад) с учётом 2π."""
    return _wrap_to_pi(a - b)


def estimate_phi0_dl_wrapped_from_vectors(
    f_hz,          # array (N,) — частоты в Гц
    S_vec,         # array (N,) — S-параметр (комплексный)
    w_norm,        # array (N,) — нормированные частоты
    w0,            # одна точка нормированной частоты (например, 1.5)
    eps_mu,        # коэффициент в beta(f)=2*pi*eps_mu*f
    label="S11",
    verbose=True,
    plot=True,
    dl_min=1e-4,   # нижняя граница для dl (>0)
    alpha_pi=1,  # вес прилипания к ±pi на Этапе 1
    beta_sym=0.1, # вес симметрии на Этапе 2 (чтобы она была в приоритете)
):
    """
    Двухэтапная оценка phi0 и dl.

    Этап 1 (только phi0, dl=0):
      - основное:  phi_corr(-w0) ≈ -phi_corr(+w0)
      - вторично:  phi_corr(-w0) ≈ +pi, phi_corr(+w0) ≈ -pi

      phi_corr(w) = angle(S(w)) + 2*phi0

    Этап 2 (фиксируем phi0=phi0_opt, находим dl):
      - основное:  phi_corr(-w0) ≈ -phi_corr(+w0)
      - вторично:  фаза на краях как можно более «ровная» (|dphi/dw| минимальна)

      phi_corr(w) = angle(S(w)) + 2*(phi0_opt + beta(f)*dl),
      beta(f) = 2*pi*eps_mu*f

    Важные моменты:
      - работаем только с wrapped фазой (np.angle, _wrap_to_pi)
      - точки ±w0 берутся через линейную интерполяцию по w_norm
      - dl >= dl_min > 0
    """

    f_hz   = np.asarray(f_hz,   float).ravel()
    S_vec  = np.asarray(S_vec,  complex).ravel()
    w_norm = np.asarray(w_norm, float).ravel()
    assert f_hz.shape == S_vec.shape == w_norm.shape

    # ---- интерполяция по w_norm для f и S ----
    f_interp    = interp1d(w_norm, f_hz,       kind="linear", bounds_error=True)
    s_re_interp = interp1d(w_norm, S_vec.real, kind="linear", bounds_error=True)
    s_im_interp = interp1d(w_norm, S_vec.imag, kind="linear", bounds_error=True)

    # точки ±w0
    w_plus  = float(+w0)
    w_minus = float(-w0)

    f_plus  = float(f_interp(w_plus))
    f_minus = float(f_interp(w_minus))

    S_plus  = s_re_interp(w_plus)  + 1j * s_im_interp(w_plus)
    S_minus = s_re_interp(w_minus) + 1j * s_im_interp(w_minus)

    phi_plus_meas  = np.angle(S_plus)
    phi_minus_meas = np.angle(S_minus)

    beta_plus  = 2 * np.pi * eps_mu * f_plus
    beta_minus = 2 * np.pi * eps_mu * f_minus

    # ---- ЭТАП 1: оптимизируем только phi0 (dl=0) ----
    def objective_phi0(x):
        phi0 = float(x[0])
        # корректируем только постоянкой
        phi_plus_corr  = phi_plus_meas  + 2 * phi0
        phi_minus_corr = phi_minus_meas + 2 * phi0

        # симметрия: phi(-w0) ~ -phi(+w0)
        err_odd = phi_minus_corr + phi_plus_corr
        term_odd = err_odd**2

        # «прилипание» к +pi / -pi
        d_minus = _angle_diff(phi_minus_corr, +np.pi)
        d_plus  = _angle_diff(phi_plus_corr,  -np.pi)
        term_pi = d_minus**2 + d_plus**2

        return term_odd + alpha_pi * term_pi

    res1 = minimize(
        fun=objective_phi0,
        x0=np.array([0.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)]
    )
    phi0_opt = float(res1.x[0])

    # ---- ЭТАП 2: оптимизируем dl при фиксированном phi0 ----
    phi_meas_full = np.angle(S_vec)
    beta_all      = 2 * np.pi * eps_mu * f_hz

    # маска "краёв" по нормированной частоте: |w| >= |w0|
    edge_mask = np.abs(w_norm) >= abs(w0-0.2)

    def objective_dl(x):
        dl = float(x[0])

        # фаза по всей сетке после коррекции (phi0 уже учтён)
        phi_corr_all = phi_meas_full + 2 * (phi0_opt + beta_all * dl)
        phi_corr_all_wr = _wrap_to_pi(phi_corr_all)

        # симметрия в ±w0
        S_plus_corr  = S_plus  * np.exp(1j * (2*phi0_opt + 2*beta_plus  * dl))
        S_minus_corr = S_minus * np.exp(1j * (2*phi0_opt + 2*beta_minus * dl))

        phi_plus_corr  = _wrap_to_pi(np.angle(S_plus_corr))
        phi_minus_corr = _wrap_to_pi(np.angle(S_minus_corr))

        err_odd = phi_minus_corr + phi_plus_corr
        term_odd = err_odd**2  # это главный критерий, будет умножен на beta_sym

        # «ровность» фазы на краях: минимизируем |dphi/dw|^2 там, где |w| >= |w0|
        # работаем с wrapped-фазой, чтобы не использовать unwrap
        if np.any(edge_mask):
            dphi_dw = np.gradient(phi_corr_all_wr, w_norm)
            dphi_edges = dphi_dw[edge_mask]
            term_flat = np.mean(dphi_edges**2)
        else:
            term_flat = 0.0

        return beta_sym * term_odd + term_flat

    res2 = minimize(
        fun=objective_dl,
        x0=np.array([max(dl_min * 10, 1e-4)], dtype=float),
        method="L-BFGS-B",
        bounds=[(dl_min, None)]
    )
    dl_opt = float(res2.x[0])

    # ---- финальная фаза и S-параметры по всей сетке ----
    phi_corr_full = phi_meas_full + 2 * (phi0_opt + beta_all * dl_opt)
    phi_corr_full_wr = _wrap_to_pi(phi_corr_full)

    S_corr_full = S_vec * np.exp(1j * (2 * phi0_opt + 2 * beta_all * dl_opt))

    # финальные значения в точках ±w0
    S_plus_corr_fin  = S_plus  * np.exp(1j * (2*phi0_opt + 2*beta_plus  * dl_opt))
    S_minus_corr_fin = S_minus * np.exp(1j * (2*phi0_opt + 2*beta_minus * dl_opt))
    phi_plus_fin  = _wrap_to_pi(np.angle(S_plus_corr_fin))
    phi_minus_fin = _wrap_to_pi(np.angle(S_minus_corr_fin))

    if verbose:
        print("----", label, " RESULT ----")
        print("Stage 1 (phi0) success:", res1.success, "  fun =", res1.fun)
        print("Stage 2 (dl)   success:", res2.success, "  fun =", res2.fun)
        print(f"phi0 = {phi0_opt:.6f} rad  ({np.degrees(phi0_opt):.2f}°)")
        print(f"dl   = {dl_opt:.6e}  (dl >= {dl_min})")
        print(f"[{label}] До коррекции (interp @ ±w0):")
        print(f"  phi(+w0={w0})  = {phi_plus_meas:.6f} рад  ({np.degrees(phi_plus_meas):.2f}°)")
        print(f"  phi(-w0={-w0}) = {phi_minus_meas:.6f} рад  ({np.degrees(phi_minus_meas):.2f}°)")
        print(f"[{label}] После коррекции (phi0 + dl):")
        print(f"  phi_corr(+w0)  = {phi_plus_fin:.6f} рад  ({np.degrees(phi_plus_fin):.2f}°)")
        print(f"  phi_corr(-w0)  = {phi_minus_fin:.6f} рад  ({np.degrees(phi_minus_fin):.2f}°)")
        print(
            f"  odd check: phi_corr(+w0)+phi_corr(-w0) = "
            f"{(phi_plus_fin + phi_minus_fin):.6e} рад "
            f"({np.degrees(phi_plus_fin + phi_minus_fin):.6e}°)"
        )

    # ---- график фазы до/после ----
    if plot:
        plt.figure(figsize=(8, 4))

        plt.plot(w_norm, np.degrees(2.5/w_norm) + 149, label='Гипербола')
        plt.plot(w_norm, np.degrees(phi_meas_full),   label='phi meas (wrapped)')
        plt.plot(w_norm, np.degrees(phi_corr_full_wr), label='phi corr (wrapped)')
        plt.ylim([-180, 180])

        plt.axvline(+w0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(-w0, color='r', linestyle='--', alpha=0.5)

        plt.scatter(
            [w0, -w0],
            [np.degrees(phi_plus_fin), np.degrees(phi_minus_fin)],
            color='k', zorder=5, label='corr @ ±w0 (interp)'
        )

        plt.xlabel('Нормированная частота w')
        plt.ylabel('Фаза (°)')
        plt.title(f'{label}: фаза до/после (двухэтапная коррекция)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # можно отдельно посмотреть градиент на краях
        dphi_dw_final = np.gradient(phi_corr_full_wr, w_norm)
        plt.figure(figsize=(8, 4))
        plt.plot(w_norm, dphi_dw_final, label='dφ/dw (wrapped)')
        plt.axvline(+w0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(-w0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Нормированная частота w')
        plt.ylabel('dφ/dw')
        plt.title(f'{label}: производная фазы после коррекции')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    return phi0_opt, dl_opt, phi_corr_full_wr, S_corr_full



def fit_phase_edges_curvefit(
    w,
    net: rf.Network,
    center_points=(-1.5, 1.5),
    phase_ylim_deg=(-180, 180),
    correct_freq_dependence=True,
    fit_on_extrapolated=True,
    plot=True
):
    w = np.asarray(w, float).ravel()
    S = np.asarray(net.s)
    assert S.shape[-2:] == (2, 2)

    def find_centering_shift(y, w, w1, w2):
        """Подбор постоянного смещения фазы."""
        interp_y = interp1d(w, y, kind='linear', bounds_error=True)
        y1 = interp_y(w1)
        y2 = interp_y(w2)
        shift = -0.5 * (y1 + y2)
        return shift

    phi1_c = phi2_c = phi1_b = phi2_b = 0.0
    res = {}

    # --------------------------------------------------------
    #     ОБРАБОТКА S11 И S22
    # --------------------------------------------------------
    for name, ij in [('S11', (0, 0)), ('S22', (1, 1))]:
        z = S[:, ij[0], ij[1]]
        phi_wrapped = np.angle(z)

        # --------------------------------------------------------
        #      1) Центрирование фазы
        # --------------------------------------------------------
        w1, w2 = center_points

        phi_shift = find_centering_shift(phi_wrapped, w, -1.4, 1.4)
        phi_centered = _wrap_to_pi(phi_wrapped + phi_shift)
        phi_c = -0.5 * phi_shift

        # Значения фазы до/после центрирования → ДЛЯ ВЫВОДА
        interp_phi_before = interp1d(w, phi_wrapped, kind='linear', bounds_error=True)
        interp_phi_after  = interp1d(w, phi_centered, kind='linear', bounds_error=True)

        phi_bef_1 = float(interp_phi_before(w1))
        phi_bef_2 = float(interp_phi_before(w2))
        phi_aft_1 = float(interp_phi_after(w1))
        phi_aft_2 = float(interp_phi_after(w2))

        print(f"\n{name}:")
        print(f"  Исходная фаза в w1={w1}: {phi_bef_1:+.4f} рад ({np.degrees(phi_bef_1):+.2f}°)")
        print(f"  Исходная фаза в w2={w2}: {phi_bef_2:+.4f} рад ({np.degrees(phi_bef_2):+.2f}°)")
        print(f"  Центрированная фаза в w1={w1}: {phi_aft_1:+.4f} рад ({np.degrees(phi_aft_1):+.2f}°)")
        print(f"  Центрированная фаза в w2={w2}: {phi_aft_2:+.4f} рад ({np.degrees(phi_aft_2):+.2f}°)")
        print(f"  Разность после центрирования: phi(w2)+phi(w1) = {phi_aft_1 + phi_aft_2:+.4e} рад")

        # Следующая производная (уже от центрированной)
        phi_unwrapped = np.unwrap(phi_centered)
        dphi_dw_centered = np.gradient(phi_unwrapped, net.f)

        # plt.figure()
        # plt.title("ГВЗ")
        # plt.plot(net.f, dphi_dw_centered)

        # --------------------------------------------------------
        #      2) Частотно-зависимая коррекция (если включена)
        # --------------------------------------------------------
        if correct_freq_dependence:
            if fit_on_extrapolated:

                spline = InterpolatedUnivariateSpline(w, dphi_dw_centered, k=1, ext='extrapolate')

                # w_ext = np.linspace(center_points[0] - 0.2,
                #                     center_points[1] + 0.2, 1000)
                w_ext = w
                tau_ext = spline(w_ext)

                mask_orig = np.abs(w) > 2.0
                # a2_ext, C_ext = _fit_gvz_inverse_square(w, dphi_dw_centered, mask=mask_orig)

                interp_y = interp1d(w, dphi_dw_centered, kind='linear', bounds_error=True)
                C_ext = abs(interp_y(center_points[0])) - abs(interp_y(center_points[1]))

                a2_ext = 0
            else:
                # Исходная фазовая производная
                dphi_dw_initial = net.group_delay[:, ij[0], ij[1]]
                mask_orig = np.abs(w) > 1.65
                a2_ext, C_ext = _fit_gvz_inverse_square(w, dphi_dw_initial, mask=mask_orig)

            phi_b = -0.5 * C_ext * 3e8

            phi_linear = 2.0 * phi_b * net.f / 3e8
            phi_wrapped = _wrap_to_pi(phi_wrapped + phi_linear)
        else:
            a2_ext = 0
            C_ext = 0
            phi_b = 0

        # --------------------------------------------------------
        #     Сохранение коэффициентов
        # --------------------------------------------------------
        if name == 'S11':
            phi1_c = phi_c
            phi1_b = phi_b
        else:
            phi2_c = phi_c
            phi2_b = phi_b

        print(f"  φ_c = {phi_c:.6f} рад ({np.degrees(phi_c):.2f}°)")
        print(f"  φ_b = {phi_b:.6f} рад/нч ({np.degrees(phi_b):.2f}°/нч)")

        # --------------------------------------------------------
        #     3) График фазы + ВЕРТИКАЛЬНЫЕ ЛИНИИ
        # --------------------------------------------------------
        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(w, np.degrees(phi_wrapped), '--', alpha=0.5, label='Без линейной компоненты')
            plt.plot(w, np.degrees(phi_centered), label='Центрированная φ')

            spline2 = InterpolatedUnivariateSpline(w, phi_centered, k=1, ext='extrapolate')
            w_ext_2 = np.linspace(center_points[0] - 0.2, center_points[1] + 0.2, 500)
            plt.plot(w_ext_2, np.degrees(spline2(w_ext_2)), label='Экстрап. центрированная φ')

            # ВЕРТИКАЛЬНЫЕ ЛИНИИ
            plt.axvline(w1, color='r', linestyle='--', label=f'w1={w1}')
            plt.axvline(w2, color='g', linestyle='--', label=f'w2={w2}')

            # Горизонтальные линии на уровне фазы в этих точках
            plt.axhline(np.degrees(phi_aft_1), color='r', alpha=0.3)
            plt.axhline(np.degrees(phi_aft_2), color='g', alpha=0.3)

            plt.xlabel('Нормированная частота')
            plt.ylabel('Фаза (°)')
            plt.title(f'{name}: центрирование + линии w1/w2')
            plt.grid(True)
            plt.legend()
            plt.ylim(*phase_ylim_deg)
            plt.tight_layout()

        # --------------------------------------------------------
        #     4) График ГВЗ (если включено)
        # --------------------------------------------------------
        if correct_freq_dependence and fit_on_extrapolated and plot:
            plt.figure(figsize=(8, 4))
            plt.plot(w_ext, tau_ext, '-', alpha=0.4, label='Экстрап. ГВЗ')
            plt.plot(w_ext, -a2_ext / w_ext**2 - C_ext, '--', label='Гипербола')
            plt.xlabel('Нормированная частота')
            plt.ylabel('Групповая задержка')
            plt.title(f'{name}: ГВЗ и гипербола')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.ylim([-1e-6, 1e-6])

        res[name] = dict(
            phi_wrapped=phi_wrapped,
            phi_centered=phi_centered,
            dphi_dw=None,
            phi_c=phi_c,
            phi_b=phi_b,
            a2_ext=a2_ext,
            C_ext=C_ext,
            w_ext=None if correct_freq_dependence and fit_on_extrapolated else None,
            tau_ext=None if correct_freq_dependence and fit_on_extrapolated else None,
            phi_before_points=(phi_bef_1, phi_bef_2),
            phi_after_points=(phi_aft_1, phi_aft_2),
        )

    # --------------------------------------------------------
    #   ДЕЭМБЕДДИНГ
    # --------------------------------------------------------
    def _deembed_ports(w, net: rf.Network, phi1_c, phi2_c, phi1_b=0.0, phi2_b=0.0):
        w = np.asarray(w, float).ravel()
        S = np.array(net.s, dtype=np.complex128, copy=True)
        f11 = np.exp(1j * 2.0 * (phi1_c + phi1_b * w))
        f22 = np.exp(1j * 2.0 * (phi2_c + phi2_b * w))
        f21 = -np.exp(1j * ((phi1_c + phi2_c) + (phi1_b + phi2_b) * w))
        S[:, 0, 0] *= f11
        S[:, 1, 1] *= f22
        S[:, 0, 1] *= f21
        S[:, 1, 0] *= f21
        return S

    S_corr = _deembed_ports(w, net, phi1_c, phi2_c, phi1_b, phi2_b)

    return res, S_corr


def plot_phase_histograms(res_list, bins=50):
    """
    Строит гистограммы фазовых коэффициентов по списку результатов res.

    res_list: List[Dict], где каждый res содержит 'S11' и 'S22' с ключами 'phi_c' и 'phi_b'.
    bins: количество бинов в гистограмме.
    """
    phi1_c = [np.degrees(res['S11']['phi_c']) for res in res_list]
    phi2_c = [np.degrees(res['S22']['phi_c']) for res in res_list]
    phi1_b = [np.degrees(res['S11']['phi_b']) for res in res_list]
    phi2_b = [np.degrees(res['S22']['phi_b']) for res in res_list]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].hist(phi1_c, bins=bins, color='skyblue')
    axs[0, 0].set_title("φ1_c (S11), постоянная фаза [°]")

    axs[0, 1].hist(phi2_c, bins=bins, color='lightgreen')
    axs[0, 1].set_title("φ2_c (S22), постоянная фаза [°]")

    axs[1, 0].hist(phi1_b, bins=bins, color='salmon')
    axs[1, 0].set_title("φ1_b (S11), линейная компонента [°/нч]")

    axs[1, 1].hist(phi2_b, bins=bins, color='orchid')
    axs[1, 1].set_title("φ2_b (S22), линейная компонента [°/нч]")

    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel("Значение")
        ax.set_ylabel("Частота")

    plt.tight_layout()


def main():
    i = 0
    work_model = common.WorkModel(configs.ENV_DATASET_PATH, configs.BASE_DATASET_SIZE, SamplerTypes.SAMPLER_SOBOL)
    tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/narrowband")
    tds_cst = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/cst")
    cst_fil = tds_cst[i][1]
    tds_fil = tds[i][1]
    matrix = CouplingMatrix.from_file(f"filters/FilterData/{configs.FILTER_NAME}/measure/cst/16.txt")
    matrix_fil = MWFilter(order=matrix.matrix_order-2, f0=work_model.orig_filter.f0, matrix=matrix.matrix,Q=work_model.orig_filter.Q, frequency=cst_fil.f, bw=work_model.orig_filter.bw)
    matrix_S = matrix_fil.response(cst_fil.f/1e6)

    res_list = []
    # plt.figure()
    # cst_fil.plot_s_re(m=0, n=0, label='S11 Re CST')
    # tds_fil.plot_s_re(m=0, n=0, label='S11 Re orig', ls=':')
    # cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST')
    # tds_fil.plot_s_im(m=0, n=0, label='S11 Im orig', ls='--')
    # plt.title("Im, Re part of S11 before correction")
    #
    # plt.figure()
    # plt.plot(cst_fil.f, MWFilter.to_db(torch.tensor(matrix_S[:, 1, 1], dtype=torch.complex64)))
    # plt.plot(tds_fil.f, MWFilter.to_db(torch.tensor(tds_fil.s[:, 1, 1], dtype=torch.complex64)))
    # plt.legend(["S22 from CST matrix", "S22 from VNA"])
    #
    # plt.figure()
    # plt.plot(cst_fil.f, MWFilter.to_db(torch.tensor(matrix_S[:, 0, 0], dtype=torch.complex64)))
    # plt.plot(tds_fil.f, MWFilter.to_db(torch.tensor(tds_fil.s[:, 0, 0], dtype=torch.complex64)))
    # plt.legend(["S11 from CST matrix", "S11 from VNA"])
    #
    # plt.figure()
    # plt.plot(cst_fil.f, MWFilter.to_db(torch.tensor(matrix_S[:, 0, 0], dtype=torch.complex64)))
    # plt.plot(tds_fil.f, MWFilter.to_db(torch.tensor(tds_fil.s[:, 1, 1], dtype=torch.complex64)))
    # plt.legend(["S11 from CST matrix", "S22 from VNA"])
    #
    # plt.figure()
    # plt.plot(cst_fil.f, MWFilter.to_db(torch.tensor(matrix_S[:, 1, 1], dtype=torch.complex64)))
    # plt.plot(tds_fil.f, MWFilter.to_db(torch.tensor(tds_fil.s[:, 0, 0], dtype=torch.complex64)))
    # plt.legend(["S22 from CST matrix", "S11 from VNA"])
    # # for i in range(len(tds)):
    # networks = [tds[i][1] for i in range(len(tds))]
    # net = tds[i][1]
    # net.plot_s_db(m=0, n=0, label='S11 origin', ls=':')
    # net.plot_s_db(m=1, n=0, label='S21 origin', ls=':')
    # net.plot_s_db(m=1, n=1, label='S22 origin', ls=':')
    # cst_fil.plot_s_db(m=0, n=0, label='S11 CST')
    # cst_fil.plot_s_db(m=1, n=0, label='S21 CST')
    # cst_fil.plot_s_db(m=1, n=1, label='S22 CST')
    #
    # plt.figure()
    # cst_fil.plot_s_re(m=0, n=0, label='S11 Re CST')
    # net.plot_s_re(m=0, n=0, label='S11 Re orig', ls=':')
    # cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST')
    # net.plot_s_im(m=0, n=0, label='S11 Im orig', ls='--')
    # plt.title("Im, Re part of S11 before correction")
    #
    # plt.figure()
    # cst_fil.plot_s_re(m=1, n=0, label='S21 Re CST')
    # net.plot_s_re(m=1, n=0, label='S21 Re orig', ls=':')
    # cst_fil.plot_s_im(m=1, n=0, label='S21 Im CST')
    # net.plot_s_im(m=1, n=0, label='S21 Im orig', ls='--')
    # ax = plt.gca()
    # ax.plot(cst_fil.f, matrix_S[:, 1, 0].real, label='S21 Re matrix', ls='-.')
    # ax.plot(cst_fil.f, matrix_S[:, 1, 0].imag, label='S21 Re matrix', ls='-.')
    # plt.title("Im, Re part of S21 before correction")

    for i in range(0, 1):
        net = tds[i][1]
        cst_fil = tds_cst[i][1]
        w_norm = MWFilter.freq_to_nfreq(net.f/1e6, work_model.orig_filter.f0, work_model.orig_filter.bw)

        w_ext, s11_ext, s21_ext = cauchy_method.extract_coeffs(freq=net.f / 1e6, Q=work_model.orig_filter.Q, f0=work_model.orig_filter.f0,
                                     s11=net.s[:, 0, 0], s21=-net.s[:, 1, 0], N=work_model.orig_filter.order + 2,
                                     nz=6 + 2, bw=work_model.orig_filter.bw)
        f_ext = MWFilter.nfreq_to_freq(w_ext, work_model.orig_filter.f0, work_model.orig_filter.bw)*1e6

        # w — нормированные частоты, net.s[:,0,0] — S11
        s11_ext *= np.exp(1j * (0.440495))
        gd11 = np.gradient(s11_ext, w_ext)
        A_right, C_right, info_right = fit_gvz_hyperbola_one_side(
            w=w_ext,
            tau=gd11,
            side="left",
            w_start=3.8,  # откуда начинать fit на правой стороне
            w_end=None,  # опционально, до куда
            plot=True,
        )
        dl = C_right*3e8/2
        print(f"dl={dl}")
        S_corr = s11_ext*np.exp(1j * (C_right*w_ext))

        # phi0_opt, dl_opt, phi_corr_wrapped, S_corr = estimate_phi0_dl_wrapped_from_vectors(
        #     f_hz=f_ext,
        #     S_vec=s11_ext,
        #     w_norm=w_ext,
        #     w0=1.8,
        #     eps_mu=1/3e8
        # )

        # res, S_def = fit_phase_edges_curvefit(
        #     w_norm, net,
        #     # q=0.35,
        #     center_points=(-1.685, 1.685),
        #     correct_freq_dependence=False,
        #     fit_on_extrapolated=True,
        #     plot=False
        # )
        # cauchy_method.extract_coeffs(freq=net.f/1e6, Q=work_model.orig_filter.Q, f0=work_model.orig_filter.f0, s11=S_def[:, 0, 0], s21=-S_def[:, 1, 0], N=work_model.orig_filter.order+2, nz=6+2, bw=work_model.orig_filter.bw)

        # res_list.append(res)
        # net.s = S_def
        #
        plt.figure()
        cst_fil.plot_s_re(m=0, n=0, label='S11 Re CST')
        plt.plot(f_ext, np.real(S_corr), ls=':')
        # net.plot_s_re(m=0, n=0, label='S11 Re Corr', ls=':')
        cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST')
        plt.plot(f_ext, np.imag(S_corr), ls='--')
        # net.plot_s_im(m=0, n=0, label='S11 Im Corr', ls='--')
        plt.title("Im, Re part of S11 after correction")

        # plt.figure()
        # plt.plot(net.f, np.gradient(np.angle(S_corr), net.f))
        #
        # plt.figure()
        # cst_fil.plot_s_re(m=1, n=0, label='S21 Re CST')
        # net.plot_s_re(m=1, n=0, label='S21 Re Corr', ls=':')
        # cst_fil.plot_s_im(m=1, n=0, label='S21 Im CST')
        # net.plot_s_im(m=1, n=0, label='S21 Im Corr', ls='--')
        # plt.title("Im, Re part of S21 after correction")
    # plot_phase_histograms(res_list)


    # plt.figure()
    # plt.plot(net.f, np.angle(S_def[:, 0, 0]), cst_fil.f, np.angle(cst_fil.s[:, 0, 0]))
    # plt.legend(["Orig S11", "CST S11"])
    # plt.title("Phase S11")
    #
    # plt.figure()
    # plt.plot( net.f, np.angle(S_def[:, 0, 1]), cst_fil.f, np.angle(cst_fil.s[:, 0, 1]))
    # plt.legend(["Orig S21", "CST S21"])
    # plt.title("Phase S21")
    #
    # plt.figure()
    # plt.plot(net.f, net.group_delay[:,0,0], cst_fil.f, cst_fil.group_delay[:,0,0])
    # plt.legend(["Orig", "CST"])
    # plt.title("Group delay")
    plt.show()

if __name__ == "__main__":
    main()

