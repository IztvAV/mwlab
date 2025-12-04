import math
import re
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy.polynomial.polynomial import polyfit
import skrf as rf
import common
from filters import SamplerTypes, MWFilter, CouplingMatrix
from filters.codecs import MWFilterTouchstoneCodec
from mwlab import TouchstoneDataset
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import cauchy_method
from scipy.optimize import curve_fit, minimize_scalar, minimize


def apply_phase_one(S, phi):
    S_new = S * np.exp(-1j * phi)
    return S_new

def apply_phase_all(S11, S21, S22, w, a11, b11, a22, b22):
    phi11 = a11 + b11 * w
    phi22 = a22 + b22 * w
    S11_corr = apply_phase_one(S11, phi11)
    S22_corr = apply_phase_one(S22, phi22)
    S21_corr = apply_phase_one(S21, 0.5 * (phi11 + phi22))
    return S11_corr, S21_corr, S22_corr


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


def de_embedding_2p_network(
    ntw: rf.Network,
    N: int,
    CF: float,
    BW: float,
    M_additional=3,          # int или (M_S11, M_S22)
    Line: float = 2.0,
    X_iter: int = 18,
    err_ak_limit_db: float = -60.0,
    err_cp_limit_db: float = -60.0,
    plot_debug: bool = False,
    log_zeros: bool = True,
    log_metrics: bool = True,
    return_metrics: bool = False,
):
    """
    Фазовый деэмбеддинг 2-портовой сети по алгоритму De_Embedding_2p.m
    (Vector Fitting, выделение фазовой нагрузки и её удаление), с
    возможностью задать отдельный M_additional для S11 и S22.

    Параметры
    ---------
    ntw : skrf.Network
        Двухпортовая сеть (S-параметры ntw.s формы (n_freq, 2, 2)).
    N : int
        Порядок фильтра (число резонаторов).
    CF : float
        Центральная частота (Гц).
    BW : float
        Полоса пропускания (Гц).
    M_additional : int или (int, int)
        Дополнительный порядок для VF:
          - если одно число: используется и для S11, и для S22;
          - если (M11, M22): отдельно для S11 и S22.
    Line : float
        Радиус в s-плоскости, разделяющий нули/полюса “фазовой нагрузки”
        (|s| >= Line) и собственные нули S-параметров (|s| < Line).
    X_iter, err_ak_limit_db, err_cp_limit_db :
        Параметры сходимости VF.
    plot_debug : bool
        Рисовать отладочные графики (фит + скорректированные S).
    log_zeros : bool
        Печатать нули передачи по S11/S22 (фазовые / собственные).
    log_metrics : bool
        Печатать метрики точности аппроксимации.
    return_metrics : bool
        Если True, вернуть (new_ntw, metrics).

    Возвращает
    ----------
    new_ntw : skrf.Network
        Новый объект Network с деэмбеддированными S-параметрами.
    metrics : dict (опционально)
        Словарь метрик для S11 и S22.
    """

    if ntw.number_of_ports != 2:
        raise ValueError("Ожидается двухпортовый Network (number_of_ports == 2)")

    # Нормализуем M_additional к форме (M_S11, M_S22)
    if isinstance(M_additional, (list, tuple)):
        if len(M_additional) != 2:
            raise ValueError("Если M_additional задаётся как последовательность, "
                             "она должна иметь длину 2: (M_S11, M_S22)")
        M_list = (int(M_additional[0]), int(M_additional[1]))
    else:
        M_list = (int(M_additional), int(M_additional))

    f = ntw.f  # частоты в Гц (n_freq,)
    s = ntw.s  # S-параметры (n_freq, 2, 2)

    # Исходные S-параметры
    S11 = s[:, 0, 0]
    S12 = s[:, 0, 1]
    S21 = s[:, 1, 0]
    S22 = s[:, 1, 1]

    # -------- BPF -> LPF нормировка --------
    FBW = BW / CF
    F_band = f.astype(float)

    w_low = (F_band / CF - CF / F_band) / FBW
    s_low = 1j * w_low
    Ns = len(F_band)

    Sxx_list = [S11, S22]
    labels = ["S11", "S22"]
    Extralfa = []   # α(s) для S11 и S22
    ExtrSxx = []    # полная рациональная аппроксимация S11/S22
    metrics = {}    # метрики точности по портам

    # -------- вспомогательная функция для метрик --------
    def _compute_metrics(S_meas, S_fit):
        diff_complex = S_meas - S_fit
        mse_complex = np.mean(np.abs(diff_complex) ** 2)

        mag_meas_db = 20 * np.log10(np.abs(S_meas) + 1e-30)
        mag_fit_db = 20 * np.log10(np.abs(S_fit) + 1e-30)
        diff_mag_db = mag_meas_db - mag_fit_db
        mse_mag_db = np.mean(diff_mag_db ** 2)
        max_abs_mag_db = np.max(np.abs(diff_mag_db))

        phase_meas = np.unwrap(np.angle(S_meas))
        phase_fit = np.unwrap(np.angle(S_fit))
        diff_phase = phase_meas - phase_fit
        mse_phase_rad = np.mean(diff_phase ** 2)
        max_abs_phase_rad = np.max(np.abs(diff_phase))

        return {
            "mse_complex": mse_complex,
            "mse_mag_db": mse_mag_db,
            "max_abs_mag_db": max_abs_mag_db,
            "mse_phase_rad": mse_phase_rad,
            "max_abs_phase_rad": max_abs_phase_rad,
        }

    # -------- Vector Fitting для S11 и S22 (с разным M_additional) --------
    for idx, (Sxx, M_add_port) in enumerate(zip(Sxx_list, M_list)):
        label = labels[idx]
        Sxx = np.asarray(Sxx, dtype=complex)
        M_Sxx = np.diag(Sxx)

        n_tot = N + M_add_port  # полный порядок для данного порта

        # начальные полюса ak (как в MATLAB-скрипте)
        ak = np.zeros((n_tot,), dtype=complex)
        for i in range(n_tot):
            val = -3 + 5.3 / (n_tot - 1) * i
            ak[i] = -0.01 * abs(val) + 1j * val

        tmp_ak = ak.copy()
        b = np.ones((n_tot, 1), dtype=complex)

        for it in range(X_iter):
            # A2(:,k) = 1./(s - ak(k))
            A2 = 1.0 / (s_low[:, None] - ak[None, :])  # (Ns, n_tot)
            A1 = np.hstack([A2, np.ones((Ns, 1), dtype=complex)])
            A = np.diag(ak)

            left = np.hstack([A1, -M_Sxx @ A2])
            right = Sxx.reshape(-1, 1)

            C_all, *_ = np.linalg.lstsq(left, right, rcond=None)

            residue = C_all[:n_tot, 0]
            d = C_all[n_tot, 0]
            cp = C_all[n_tot + 1:, 0]

            A_new = A - b @ cp.reshape(1, -1)
            ak = np.linalg.eigvals(A_new)

            err = np.sum(np.abs(ak - tmp_ak))
            cp_db = 10 * np.log10(np.abs(cp) + 1e-30)

            if (10 * np.log10(err + 1e-30) < err_ak_limit_db and
                    np.max(cp_db) < err_cp_limit_db):
                break

            tmp_ak = ak.copy()

        # -------- нули --------
        A11 = np.diag(ak)
        A_zero = A11 - b @ residue.reshape(1, -1) / d
        zeros = np.linalg.eigvals(A_zero)

        # -------- разделение нулей по радиусу Line --------
        phase_zeros = zeros[np.abs(zeros) >= Line]   # “фазовая нагрузка”
        poly_zeros  = zeros[np.abs(zeros) < Line]    # “собственные” нули Sxx

        if log_zeros:
            print(f"\n=== Zeros for {label} ===")
            print(f"Line radius = {Line}")
            print("Phase-shift-related zeros (|s| >= Line):")
            if phase_zeros.size == 0:
                print("  [none]")
            else:
                for z in phase_zeros:
                    print(f"  {z.real:+.4f} {z.imag:+.4f}j  |s|={abs(z):.4f}")

            print("Intrinsic S-parameter zeros (|s| < Line):")
            if poly_zeros.size == 0:
                print("  [none]")
            else:
                for z in poly_zeros:
                    print(f"  {z.real:+.4f} {z.imag:+.4f}j  |s|={abs(z):.4f}")

        # -------- ПОЛНАЯ рациональная аппроксимация (extr_s) --------
        full_num = np.poly(zeros)
        full_den = np.poly(ak)

        extr_s = np.zeros_like(Sxx, dtype=complex)
        for i, sval in enumerate(s_low):
            extr_s[i] = d * np.polyval(full_num, sval) / np.polyval(full_den, sval)

        # -------- фазовая нагрузка alpha(s) --------
        alfa_zk = phase_zeros
        alfa_ak = ak[np.abs(ak) >= Line]

        if alfa_zk.size == 0 and alfa_ak.size == 0:
            alpha = np.ones_like(Sxx, dtype=complex)
        else:
            TestNum = np.poly(alfa_zk) if alfa_zk.size > 0 else np.array([1.0])
            TestDen = np.poly(alfa_ak) if alfa_ak.size > 0 else np.array([1.0])

            alpha = np.zeros_like(Sxx, dtype=complex)
            for i, sval in enumerate(s_low):
                alpha[i] = d * np.polyval(TestNum, sval) / np.polyval(TestDen, sval)

        Extralfa.append(alpha)
        ExtrSxx.append(extr_s)

        # -------- метрики точности аппроксимации --------
        m = _compute_metrics(Sxx, extr_s)
        metrics[label] = m

        if log_metrics:
            print(f"\n=== Fit metrics for {label} (M_additional={M_add_port}) ===")
            print(f"  mse_complex        = {m['mse_complex']:.3e}")
            print(f"  mse_mag_db         = {m['mse_mag_db']:.3e}")
            print(f"  max_abs_mag_db     = {m['max_abs_mag_db']:.3f} dB")
            print(f"  mse_phase_rad      = {m['mse_phase_rad']:.3e}")
            print(f"  max_abs_phase_rad  = {m['max_abs_phase_rad']:.3f} rad")

    # -------- фаза и деэмбеддинг --------
    alfa1, alfa2 = Extralfa
    phi1 = np.unwrap(np.angle(alfa1))
    phi2 = np.unwrap(np.angle(alfa2))

    New_S11 = -S11  * np.exp(-1j * phi1)
    New_S22 = -S22  * np.exp(-1j * phi2)

    # для передачи — симметричное снятие нагрузки
    New_S12 = (
        S12
        * np.exp(-1j * (0.5 * phi1 + 0.5 * phi2))
    )
    New_S21 = (
        S21
        * np.exp(-1j * (0.5 * phi1 + 0.5 * phi2))
    )

    # -------- собираем новый Network --------
    s_new = s.copy()
    s_new[:, 0, 0] = New_S11
    s_new[:, 1, 1] = New_S22
    s_new[:, 0, 1] = New_S12
    s_new[:, 1, 0] = New_S21

    new_ntw = ntw.copy()
    new_ntw.s = s_new

    # -------- опциональная отрисовка --------
    if plot_debug:
        def plot_compare(F, S_meas, S_fit, title_prefix):
            mag_meas = 20 * np.log10(np.abs(S_meas) + 1e-30)
            mag_fit = 20 * np.log10(np.abs(S_fit) + 1e-30)

            phase_meas = np.unwrap(np.angle(S_meas))
            phase_fit = np.unwrap(np.angle(S_fit))

            plt.figure(figsize=(10, 8))
            plt.suptitle(f"{title_prefix}: измеренные vs аппроксимация")

            # Real
            plt.subplot(2, 2, 1)
            plt.plot(F, np.real(S_meas), label="Re(meas)")
            plt.plot(F, np.real(S_fit), linestyle='--', label="Re(fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Re")
            plt.legend()

            # Imag
            plt.subplot(2, 2, 2)
            plt.plot(F, np.imag(S_meas), label="Im(meas)")
            plt.plot(F, np.imag(S_fit), linestyle='--', label="Im(fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Im")
            plt.legend()

            # |S| in dB
            plt.subplot(2, 2, 3)
            plt.plot(F, mag_meas, label="|S| dB (meas)")
            plt.plot(F, mag_fit, linestyle='--', label="|S| dB (fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Magnitude, dB")
            plt.legend()

            # Phase
            plt.subplot(2, 2, 4)
            plt.plot(F, phase_meas, label="phase(meas)")
            plt.plot(F, phase_fit, linestyle='--', label="phase(fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Phase, rad")
            plt.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # S11: измеренные vs аппроксимация
        plot_compare(F_band, S11, ExtrSxx[0], f"S11 (M_add={M_list[0]})")

        # S22: измеренные vs аппроксимация
        plot_compare(F_band, S22, ExtrSxx[1], f"S22 (M_add={M_list[1]})")

        # -------- графики Re/Im скорректированных S-параметров --------
        plt.figure(figsize=(10, 8))
        plt.suptitle("Corrected S-parameters: real & imag parts")

        # New S11
        plt.subplot(2, 2, 1)
        plt.plot(F_band, np.real(New_S11), label="Re(S11)")
        plt.plot(F_band, np.imag(New_S11), label="Im(S11)", linestyle='--')
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S11")
        plt.legend()

        # New S22
        plt.subplot(2, 2, 2)
        plt.plot(F_band, np.real(New_S22), label="Re(S22)")
        plt.plot(F_band, np.imag(New_S22), label="Im(S22)", linestyle='--')
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S22")
        plt.legend()

        # New S12
        plt.subplot(2, 2, 3)
        plt.plot(F_band, np.real(New_S12), label="Re(S12)")
        plt.plot(F_band, np.imag(New_S12), label="Im(S12)", linestyle='--')
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S12")
        plt.legend()

        # New S21
        plt.subplot(2, 2, 4)
        plt.plot(F_band, np.real(New_S21), label="Re(S21)")
        plt.plot(F_band, np.imag(New_S21), label="Im(S21)", linestyle='--')
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S21")
        plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if return_metrics:
        return new_ntw, metrics
    else:
        return new_ntw


# ============ 1. VF для одного порта ============

def _vf_fit_one_port(
    F_band,
    Sxx,
    N: int,
    M_additional: int,
    CF: float,
    BW: float,
    Line: float = 2.0,
    X_iter: int = 18,
    err_ak_limit_db: float = -60.0,
    err_cp_limit_db: float = -60.0,
):
    """
    Vector Fitting для одного S-параметра (S11 или S22) с данным M_additional.
    """

    F_band = np.asarray(F_band, dtype=float)
    Sxx = np.asarray(Sxx, dtype=complex)
    if F_band.shape != Sxx.shape:
        raise ValueError("F_band и Sxx должны иметь одинаковую длину")

    FBW = BW / CF
    w_low = (F_band / CF - CF / F_band) / FBW
    s_low = 1j * w_low

    n_tot = N + int(M_additional)
    M_Sxx = np.diag(Sxx)

    # --- начальные полюса ---
    ak = np.zeros((n_tot,), dtype=complex)
    for i in range(n_tot):
        val = -3 + 5.3 / (n_tot - 1) * i
        ak[i] = -0.01 * abs(val) + 1j * val

    tmp_ak = ak.copy()
    b = np.ones((n_tot, 1), dtype=complex)

    for _ in range(X_iter):
        A2 = 1.0 / (s_low[:, None] - ak[None, :])
        A1 = np.hstack([A2, np.ones((len(F_band), 1), dtype=complex)])
        A = np.diag(ak)

        left = np.hstack([A1, -M_Sxx @ A2])
        right = Sxx.reshape(-1, 1)

        C_all, *_ = np.linalg.lstsq(left, right, rcond=None)

        residue = C_all[:n_tot, 0]
        d = C_all[n_tot, 0]          # <--- d сохраняем
        cp = C_all[n_tot + 1:, 0]

        A_new = A - b @ cp.reshape(1, -1)
        ak = np.linalg.eigvals(A_new)

        err = np.sum(np.abs(ak - tmp_ak))
        cp_db = 10 * np.log10(np.abs(cp) + 1e-30)

        if (10 * np.log10(err + 1e-30) < err_ak_limit_db and
                np.max(cp_db) < err_cp_limit_db):
            break

        tmp_ak = ak.copy()

    # --- нули/полюса полных Sxx ---
    A11 = np.diag(ak)
    A_zero = A11 - b @ residue.reshape(1, -1) / d
    zeros = np.linalg.eigvals(A_zero)

    phase_zeros = zeros[np.abs(zeros) >= Line]
    intrinsic_zeros = zeros[np.abs(zeros) < Line]

    # ПОЛНЫЕ полиномы Sxx (как в MATLAB TestNum/TestDen)
    full_num = np.poly(zeros)
    full_den = np.poly(ak)

    extr_s = np.zeros_like(Sxx, dtype=complex)
    for i, sval in enumerate(s_low):
        extr_s[i] = d * np.polyval(full_num, sval) / np.polyval(full_den, sval)

    # фазовая нагрузка alpha(s)
    alfa_zk = phase_zeros
    alfa_ak = ak[np.abs(ak) >= Line]

    if alfa_zk.size == 0 and alfa_ak.size == 0:
        alpha = np.ones_like(Sxx, dtype=complex)
    else:
        TestNum = np.poly(alfa_zk) if alfa_zk.size > 0 else np.array([1.0])
        TestDen = np.poly(alfa_ak) if alfa_ak.size > 0 else np.array([1.0])
        alpha = np.zeros_like(Sxx, dtype=complex)
        for i, sval in enumerate(s_low):
            alpha[i] = d * np.polyval(TestNum, sval) / np.polyval(TestDen, sval)

    # --- метрики (как раньше) ---
    def _compute_metrics(S_meas, S_fit):
        diff_complex = S_meas - S_fit
        mse_complex = np.mean(np.abs(diff_complex) ** 2)

        mag_meas_db = 20 * np.log10(np.abs(S_meas) + 1e-30)
        mag_fit_db = 20 * np.log10(np.abs(S_fit) + 1e-30)
        diff_mag_db = mag_meas_db - mag_fit_db
        mse_mag_db = np.mean(diff_mag_db ** 2)
        max_abs_mag_db = np.max(np.abs(diff_mag_db))

        phase_meas = np.unwrap(np.angle(S_meas))
        phase_fit = np.unwrap(np.angle(S_fit))
        diff_phase = phase_meas - phase_fit
        mse_phase_rad = np.mean(diff_phase ** 2)
        max_abs_phase_rad = np.max(np.abs(diff_phase))

        return {
            "mse_complex": mse_complex,
            "mse_mag_db": mse_mag_db,
            "max_abs_mag_db": max_abs_mag_db,
            "mse_phase_rad": mse_phase_rad,
            "max_abs_phase_rad": max_abs_phase_rad,
        }

    metrics = _compute_metrics(Sxx, extr_s)

    return {
        "M_additional": int(M_additional),
        "extr_s": extr_s,           # Sxx на исходном диапазоне
        "alpha": alpha,             # фазовая нагрузка на исходном диапазоне
        "zeros_all": zeros,
        "phase_zeros": phase_zeros,
        "intrinsic_zeros": intrinsic_zeros,
        "full_num": full_num,       # <--- полином числителя Sxx
        "full_den": full_den,       # <--- полином знаменателя Sxx
        "d": d,                     # <--- коэффициент d
        "metrics": metrics,
    }

def vf_extrapolate_port_normalized(vf_result, w_low_new):
    """
    Экстраполировать S-параметр в НОРМИРОВАННОЙ области частот.

    Параметры
    ---------
    vf_result : dict
        Результат _vf_fit_one_port (или аналогичный), содержащий:
        - "full_num": np.ndarray
        - "full_den": np.ndarray
        - "d": complex
    w_low_new : array_like
        Новый массив нормированных частот w_low (реальный 1D массив).

    Возвращает
    ----------
    S_ext : np.ndarray (complex)
        Экстраполированный S(w_low_new).
    """
    w_low_new = np.asarray(w_low_new, dtype=float)
    s_new = 1j * w_low_new

    full_num = vf_result["full_num"]
    full_den = vf_result["full_den"]
    d = vf_result["d"]

    S_ext = d * np.polyval(full_num, s_new) / np.polyval(full_den, s_new)
    return S_ext

# ============ 2. Автоподбор M_additional (с сохранением результатов) ============

def auto_select_M_additional(
    ntw: rf.Network,
    N: int,
    CF: float,
    BW: float,
    M_candidates=(0, 1, 2, 3),
    Line: float = 2.0,
    mag_tol_db: float = 0.2,
    phase_tol_deg: float = 5.0,
    X_iter: int = 18,
    err_ak_limit_db: float = -60.0,
    err_cp_limit_db: float = -60.0,
):
    """
    Подбор M_additional отдельно для S11 и S22.

    Возвращает:
      best_M: {'S11': M_best_11, 'S22': M_best_22}
      best_results: {'S11': result, 'S22': result}   # result как в _vf_fit_one_port
      all_details: {'S11': {M: entry, ...}, 'S22': {...}}
    """

    if ntw.number_of_ports != 2:
        raise ValueError("Ожидается двухпортовый Network")

    f = ntw.f
    s = ntw.s
    S11 = s[:, 0, 0]
    S22 = s[:, 1, 1]

    ports = {"S11": S11, "S22": S22}
    phase_tol_rad = np.deg2rad(phase_tol_deg)

    best_M = {}
    best_results = {}
    all_details = {"S11": {}, "S22": {}}

    def _score(metr):
        # простая линейная комбинация максимумов для магн. и фазы
        return (metr["max_abs_mag_db"] +
                (180 / np.pi) * metr["max_abs_phase_rad"] * 0.1)

    for label, Sxx in ports.items():
        all_results_label = {}
        # считаем ВСЕ кандидаты (каждый VF считается один раз)
        for M_add in sorted(M_candidates):
            res = _vf_fit_one_port(
                F_band=f,
                Sxx=Sxx,
                N=N,
                M_additional=M_add,
                CF=CF,
                BW=BW,
                Line=Line,
                X_iter=X_iter,
                err_ak_limit_db=err_ak_limit_db,
                err_cp_limit_db=err_cp_limit_db,
            )
            metr = res["metrics"]
            n_intrinsic = len(res["intrinsic_zeros"])
            entry = {
                "result": res,
                "score": _score(metr),
                "n_intrinsic_zeros": n_intrinsic,
            }
            all_results_label[M_add] = entry

        all_details[label] = all_results_label

        # выбираем минимальный M_add, удовлетворяющий порогам
        chosen_M = None
        for M_add in sorted(M_candidates):
            entry = all_results_label[M_add]
            metr = entry["result"]["metrics"]
            n_intrinsic = entry["n_intrinsic_zeros"]

            if (n_intrinsic >= N and
                metr["max_abs_mag_db"] <= mag_tol_db and
                metr["max_abs_phase_rad"] <= phase_tol_rad):
                chosen_M = M_add
                break

        # если никто не прошёл пороги — выбираем лучший по score (с приоритетом n_intrinsic>=N)
        if chosen_M is None:
            candidates = [
                (M_add, entry)
                for M_add, entry in all_results_label.items()
                if entry["n_intrinsic_zeros"] >= N
            ]
            if candidates:
                chosen_M = min(candidates, key=lambda x: x[1]["score"])[0]
            else:
                chosen_M = min(all_results_label.items(), key=lambda x: x[1]["score"])[0]

        best_M[label] = chosen_M
        best_results[label] = all_results_label[chosen_M]["result"]

    return best_M, best_results, all_details


# ============ 3. Автовыбор + деэмбеддинг БЕЗ повторного VF ============

def de_embedding_2p_network_autoM(
    ntw: rf.Network,
    N: int,
    CF: float,
    BW: float,
    M_candidates=(0, 1, 2, 3),
    Line: float = 2.0,
    mag_tol_db: float = 0.2,
    phase_tol_deg: float = 5.0,
    X_iter: int = 18,
    err_ak_limit_db: float = -60.0,
    err_cp_limit_db: float = -60.0,
    plot_debug: bool = False,
):
    """
    Один вызов:
      1) подбирает M_additional для S11 и S22;
      2) использует уже посчитанные VF-результаты (без повторного VF);
      3) печатает нули и метрики для выбранных M_additional;
      4) возвращает деэмбеддированный Network + служебную инфу.
    """

    if ntw.number_of_ports != 2:
        raise ValueError("Ожидается двухпортовый Network")

    f = ntw.f
    s = ntw.s
    S11 = s[:, 0, 0]
    S12 = s[:, 0, 1]
    S21 = s[:, 1, 0]
    S22 = s[:, 1, 1]

    # 1) автоподбор + ВСЕ результаты VF
    best_M, best_results, all_details = auto_select_M_additional(
        ntw,
        N=N,
        CF=CF,
        BW=BW,
        M_candidates=M_candidates,
        Line=Line,
        mag_tol_db=mag_tol_db,
        phase_tol_deg=phase_tol_deg,
        X_iter=X_iter,
        err_ak_limit_db=err_ak_limit_db,
        err_cp_limit_db=err_cp_limit_db,
    )

    print("\n=== Auto-selected M_additional ===")
    print(f"S11: M_additional = {best_M['S11']}")
    print(f"S22: M_additional = {best_M['S22']}")

    # 2) берём готовые результаты для S11 и S22
    res11 = best_results["S11"]
    res22 = best_results["S22"]

    alfa1 = res11["alpha"]
    alfa2 = res22["alpha"]
    ExtrS11 = res11["extr_s"]
    ExtrS22 = res22["extr_s"]
    metrics = {"S11": res11["metrics"], "S22": res22["metrics"]}

    # 3) лог нулей и метрик для выбранных M_additional
    for label, res in (("S11", res11), ("S22", res22)):
        print(f"\n=== Zeros for {label} (M_additional={res['M_additional']}) ===")
        print(f"Line radius = {Line}")
        print("Phase-shift-related zeros (|s| >= Line):")
        if res["phase_zeros"].size == 0:
            print("  [none]")
        else:
            for z in res["phase_zeros"]:
                print(f"  {z.real:+.4f} {z.imag:+.4f}j  |s|={abs(z):.4f}")

        print("Intrinsic S-parameter zeros (|s| < Line):")
        if res["intrinsic_zeros"].size == 0:
            print("  [none]")
        else:
            for z in res["intrinsic_zeros"]:
                print(f"  {z.real:+.4f} {z.imag:+.4f}j  |s|={abs(z):.4f}")

        m = res["metrics"]
        print(f"\n=== Fit metrics for {label} (M_additional={res['M_additional']}) ===")
        print(f"  mse_complex        = {m['mse_complex']:.3e}")
        print(f"  mse_mag_db         = {m['mse_mag_db']:.3e}")
        print(f"  max_abs_mag_db     = {m['max_abs_mag_db']:.3f} dB")
        print(f"  mse_phase_rad      = {m['mse_phase_rad']:.3e}")
        print(f"  max_abs_phase_rad  = {m['max_abs_phase_rad']:.3f} rad")

    # 4) деэмбеддинг, используя уже посчитанные alpha
    phi1 = np.unwrap(np.angle(alfa1))
    phi2 = np.unwrap(np.angle(alfa2))

    New_S11 = -S11  / (np.exp(1j * phi1))
    New_S22 = -S22  / (np.exp(1j * phi2))
    New_S12 = (
        S12 / (np.exp(1j * (0.5 * phi1 + 0.5 * phi2)))
    )
    New_S21 = (
        S21 / (np.exp(1j * (0.5 * phi1 + 0.5 * phi2)))
    )

    s_new = s.copy()
    s_new[:, 0, 0] = New_S11
    s_new[:, 1, 1] = New_S22
    s_new[:, 0, 1] = New_S12
    s_new[:, 1, 0] = New_S21

    new_ntw = ntw.copy()
    new_ntw.s = s_new

    # 5) графики
    if plot_debug:
        F_band = f.astype(float)

        def plot_compare(F, S_meas, S_fit, title_prefix):
            mag_meas = 20 * np.log10(np.abs(S_meas) + 1e-30)
            mag_fit = 20 * np.log10(np.abs(S_fit) + 1e-30)
            phase_meas = np.unwrap(np.angle(S_meas))
            phase_fit = np.unwrap(np.angle(S_fit))

            plt.figure(figsize=(10, 8))
            plt.suptitle(f"{title_prefix}: измеренные vs аппроксимация")

            plt.subplot(2, 2, 1)
            plt.plot(F, np.real(S_meas), label="Re(meas)")
            plt.plot(F, np.real(S_fit), linestyle='--', label="Re(fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Re")
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(F, np.imag(S_meas), label="Im(meas)")
            plt.plot(F, np.imag(S_fit), linestyle='--', label="Im(fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Im")
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(F, mag_meas, label="|S| dB (meas)")
            plt.plot(F, mag_fit, linestyle='--', label="|S| dB (fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Magnitude, dB")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(F, phase_meas, label="phase(meas)")
            plt.plot(F, phase_fit, linestyle='--', label="phase(fit)")
            plt.grid(True)
            plt.xlabel("f, Hz")
            plt.ylabel("Phase, rad")
            plt.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_compare(F_band, S11, ExtrS11, f"S11 (M_add={best_M['S11']})")
        plot_compare(F_band, S22, ExtrS22, f"S22 (M_add={best_M['S22']})")

        plt.figure(figsize=(10, 8))
        plt.suptitle("Corrected S-parameters: real & imag parts")

        plt.subplot(2, 2, 1)
        plt.plot(F_band, np.real(New_S11), label="Re(S11)")
        plt.plot(F_band, np.imag(New_S11), linestyle='--', label="Im(S11)")
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S11")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(F_band, np.real(New_S22), label="Re(S22)")
        plt.plot(F_band, np.imag(New_S22), linestyle='--', label="Im(S22)")
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S22")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(F_band, np.real(New_S12), label="Re(S12)")
        plt.plot(F_band, np.imag(New_S12), linestyle='--', label="Im(S12)")
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S12")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(F_band, np.real(New_S21), label="Re(S21)")
        plt.plot(F_band, np.imag(New_S21), linestyle='--', label="Im(S21)")
        plt.grid(True)
        plt.xlabel("f, Hz")
        plt.ylabel("S21")
        plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return new_ntw, best_M, metrics, all_details


# def fit_phase_edges_curvefit(
#     w,
#     net: rf.Network,
#     center_points=(-1.5, 1.5),
#     phase_ylim_deg=(-180, 180),
#     plot=True
# ):
#     if center_points is None:
#         w_left = w[0]
#         w_right = w[-1]
#         w_th = min(abs(w_left), abs(w_right))
#         center_points = (-w_th, w_th)
#
#
#     w = np.asarray(w, float).ravel()
#     S = np.asarray(net.s)
#     assert S.shape[-2:] == (2, 2)
#
#     def find_phase_shift(phi_wrapped, w, w1, w2):
#         def objective(shift):
#             phi = _wrap_to_pi(phi_wrapped - shift)
#             phi_interp = interp1d(w, phi, kind='linear', bounds_error=True)
#             res = np.abs(_wrap_to_pi(phi_interp(w2) + phi_interp(w1)))
#             return res
#
#         from scipy.optimize import minimize_scalar
#         result = minimize_scalar(objective, bounds=(-np.pi, np.pi), method='bounded')
#         best_shift = result.x
#         print(
#             f"Оптимальный фазовый сдвиг (при φ({w2}) = -φ({w1})): {best_shift:.6f} рад ({np.degrees(best_shift):.2f}°)")
#         return best_shift
#
#     phi1_c = phi2_c = phi1_b = phi2_b = 0.0
#     res = {}
#
#     # --------------------------------------------------------
#     #     ОБРАБОТКА S11 И S22
#     # --------------------------------------------------------
#     for name, ij in [('S11', (0, 0)), ('S22', (1, 1))]:
#         z = S[:, ij[0], ij[1]]
#         phi_wrapped = np.angle(z)
#
#         # --------------------------------------------------------
#         #      1) Центрирование фазы
#         # --------------------------------------------------------
#         w1, w2 = center_points
#
#         phi_shift = find_phase_shift(phi_wrapped, w, w1, w2)
#         phi_centered = _wrap_to_pi(phi_wrapped - phi_shift)
#         phi_c = -0.5 * phi_shift
#
#         # Значения фазы до/после центрирования → ДЛЯ ВЫВОДА
#         interp_phi_before = interp1d(w, phi_wrapped, kind='linear', bounds_error=True)
#         interp_phi_after  = interp1d(w, phi_centered, kind='linear', bounds_error=True)
#
#         phi_bef_1 = float(interp_phi_before(w1))
#         phi_bef_2 = float(interp_phi_before(w2))
#         phi_aft_1 = float(interp_phi_after(w1))
#         phi_aft_2 = float(interp_phi_after(w2))
#
#         print(f"\n{name}:")
#         print(f"  Исходная фаза в w1={w1}: {phi_bef_1:+.4f} рад ({np.degrees(phi_bef_1):+.2f}°)")
#         print(f"  Исходная фаза в w2={w2}: {phi_bef_2:+.4f} рад ({np.degrees(phi_bef_2):+.2f}°)")
#         print(f"  Центрированная фаза в w1={w1}: {phi_aft_1:+.4f} рад ({np.degrees(phi_aft_1):+.2f}°)")
#         print(f"  Центрированная фаза в w2={w2}: {phi_aft_2:+.4f} рад ({np.degrees(phi_aft_2):+.2f}°)")
#         print(f"  Разность после центрирования: phi(w2)+phi(w1) = {phi_aft_1 + phi_aft_2:+.4e} рад")
#
#         # Следующая производная (уже от центрированной)
#         phi_unwrapped = np.unwrap(phi_centered)
#         dphi_dw_centered = np.gradient(phi_unwrapped, abs(w))
#
#         # --------------------------------------------------------
#         #     Сохранение коэффициентов
#         # --------------------------------------------------------
#         if name == 'S11':
#             phi1_c = phi_c
#         else:
#             phi2_c = phi_c
#
#         print(f"  φ_c = {phi_c:.6f} рад ({np.degrees(phi_c):.2f}°)")
#
#         # --------------------------------------------------------
#         #     3) График фазы + ВЕРТИКАЛЬНЫЕ ЛИНИИ
#         # --------------------------------------------------------
#         if plot:
#             plt.figure(figsize=(8, 4))
#             plt.plot(w, np.degrees(np.angle(z)), '-.', alpha=0.5, label='Исходная')
#             plt.plot(w, np.degrees(phi_wrapped), '--', alpha=0.5, label='Без линейной компоненты φ')
#             plt.plot(w, np.degrees(phi_centered), label='Центрированная φ')
#             # plt.plot(w, np.degrees(phi_centered*1/w), label='Центрированная φ c гиперболой')
#             # phi_centered_with_hip = interp1d(w, _wrap_to_pi(phi_centered*_wrap_to_pi(1/w)), kind='linear', bounds_error=True)
#             # print(f"С гиперболой: {phi_centered_with_hip(w1)}/{phi_centered_with_hip(w2)}")
#
#             # ВЕРТИКАЛЬНЫЕ ЛИНИИ
#             plt.axvline(w1, color='r', linestyle='--', label=f'w1={w1}')
#             plt.axvline(w2, color='g', linestyle='--', label=f'w2={w2}')
#
#             # Горизонтальные линии на уровне фазы в этих точках
#             plt.axhline(np.degrees(phi_aft_1), color='r', alpha=0.3)
#             plt.axhline(np.degrees(phi_aft_2), color='g', alpha=0.3)
#
#             plt.xlabel('Нормированная частота')
#             plt.ylabel('Фаза (°)')
#             plt.title(f'{name}: центрирование + линии w1/w2')
#             plt.grid(True)
#             plt.legend()
#             plt.ylim(*phase_ylim_deg)
#             plt.tight_layout()
#
#             phi_centered_with_hip = interp1d(w, _wrap_to_pi((np.unwrap(phi_centered) + (2*-0.0256*w))), kind='linear',
#                                              bounds_error=True)
#             print(f"С гиперболой: {phi_centered_with_hip(w1)}/{phi_centered_with_hip(w2)}. В нуле: {phi_centered_with_hip(0)}")
#             plt.figure(figsize=(8, 4))
#             plt.plot(w, np.unwrap(phi_centered) + (-2*0.0256*w), label='deemebedeed')
#             plt.plot(w, np.unwrap(phi_centered), label='centered')
#             plt.plot(w,  -2*0.0256*w)
#             plt.legend()
#
#         res[name] = dict(
#             phi_wrapped=phi_wrapped,
#             phi_centered=phi_centered,
#             dphi_dw=None,
#             phi_c=phi_c,
#             phi_before_points=(phi_bef_1, phi_bef_2),
#             phi_after_points=(phi_aft_1, phi_aft_2),
#         )
#
#     # --------------------------------------------------------
#     #   ДЕЭМБЕДДИНГ
#     # --------------------------------------------------------
#     def _deembed_ports(w, net: rf.Network, phi1_c, phi2_c, phi1_b=0.0, phi2_b=0.0):
#         w = np.asarray(w, float).ravel()
#         S = np.array(net.s, dtype=np.complex128, copy=True)
#         phi1 = -2.0 * (phi1_c + phi1_b * w)
#         phi2 = -2.0 * (phi2_c + phi2_b * w)
#         S[:, 0, 0] = apply_phase_one(S[:, 0, 0], phi1)
#         S[:, 1, 1] = apply_phase_one(S[:, 1, 1], phi2)
#         S[:, 0, 1] = -apply_phase_one(S[:, 0, 1], 0.5*(phi1 + phi2))
#         S[:, 1, 0] = -apply_phase_one(S[:, 1, 0], 0.5*(phi1 + phi2))
#         # f11 = np.exp(1j * 2.0 * (phi1_c + phi1_b * w))
#         # f22 = np.exp(1j * 2.0 * (phi2_c + phi2_b * w))
#         # f21 = -np.exp(1j * ((phi1_c + phi2_c) + (phi1_b + phi2_b) * w))
#         # S[:, 0, 0] *= f11
#         # S[:, 1, 1] *= f22
#         # S[:, 0, 1] *= f21
#         # S[:, 1, 0] *= f21
#         return S
#
#     S_corr = _deembed_ports(w, net, phi1_c, phi2_c, phi1_b, phi2_b)
#
#     return res, S_corr

def fit_phase_edges_curvefit(
    w,
    net: rf.Network,
    center_points=(-1.5, 1.5),
    phase_ylim_deg=(-180, 180),
    plot=True,
    verbose=True,
):
    if center_points is None:
        w_left = w[0]
        w_right = w[-1]
        w_th = min(abs(w_left), abs(w_right))
        center_points = (-w_th, w_th)

    w = np.asarray(w, float).ravel()
    S = np.asarray(net.s)
    assert S.shape[-2:] == (2, 2)

    def find_phase_shift_fast_local(phi_wrapped, w, w1, w2):
        # один интерполятор
        phi_interp = interp1d(w, phi_wrapped, kind='linear', bounds_error=True)

        phi1_0 = float(phi_interp(w1))
        phi2_0 = float(phi_interp(w2))
        shift0 = 0.5 * _wrap_to_pi(phi1_0 + phi2_0)

        def objective(shift):
            phi1 = _wrap_to_pi(phi1_0 - shift)
            phi2 = _wrap_to_pi(phi2_0 - shift)
            return abs(_wrap_to_pi(phi1 + phi2))

        result = minimize_scalar(
            objective,
            bounds=(-np.pi, +np.pi),
            method='bounded',
            options={"xatol": 1e-5},
        )
        best_shift = result.x
        if verbose:
            print(
                f"Оптимальный фазовый сдвиг (φ({w2}) = -φ({w1})): "
                f"{best_shift:.6f} рад ({np.degrees(best_shift):.2f}°)"
            )
        return best_shift

    phi1_c = phi2_c = phi1_b = phi2_b = 0.0
    res = {}

    w1, w2 = center_points

    # --------------------------------------------------------
    #     ОБРАБОТКА S11 И S22
    # --------------------------------------------------------
    for name, ij in [('S11', (0, 0)), ('S22', (1, 1))]:
        z = S[:, ij[0], ij[1]]
        phi_wrapped = np.angle(z)

        # 1) Центрирование фазы
        phi_shift = find_phase_shift_fast_local(phi_wrapped, w, w1, w2)
        phi_centered = _wrap_to_pi(phi_wrapped - phi_shift)
        phi_c = -0.5 * phi_shift

        # Для логов — интерполяция до/после
        interp_phi_before = interp1d(w, phi_wrapped, kind='linear', bounds_error=True)
        interp_phi_after  = interp1d(w, phi_centered, kind='linear', bounds_error=True)

        phi_bef_1 = float(interp_phi_before(w1))
        phi_bef_2 = float(interp_phi_before(w2))
        phi_aft_1 = float(interp_phi_after(w1))
        phi_aft_2 = float(interp_phi_after(w2))

        if verbose:
            print(f"\n{name}:")
            print(f"  Исходная фаза в w1={w1}: {phi_bef_1:+.4f} рад ({np.degrees(phi_bef_1):+.2f}°)")
            print(f"  Исходная фаза в w2={w2}: {phi_bef_2:+.4f} рад ({np.degrees(phi_bef_2):+.2f}°)")
            print(f"  Центрированная фаза в w1={w1}: {phi_aft_1:+.4f} рад ({np.degrees(phi_aft_1):+.2f}°)")
            print(f"  Центрированная фаза в w2={w2}: {phi_aft_2:+.4f} рад ({np.degrees(phi_aft_2):+.2f}°)")
            print(f"  Разность после центрирования: phi(w2)+phi(w1) = {phi_aft_1 + phi_aft_2:+.4e} рад")
            print(f"  φ_c = {phi_c:.6f} рад ({np.degrees(phi_c):.2f}°)")

        if name == 'S11':
            phi1_c = phi_c
        else:
            phi2_c = phi_c

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(w, np.degrees(phi_wrapped), '--', alpha=0.5, label='Исходная φ')
            plt.plot(w, np.degrees(phi_centered), label='Центрированная φ')
            plt.axvline(w1, color='r', linestyle='--', label=f'w1={w1}')
            plt.axvline(w2, color='g', linestyle='--', label=f'w2={w2}')
            plt.axhline(np.degrees(phi_aft_1), color='r', alpha=0.3)
            plt.axhline(np.degrees(phi_aft_2), color='g', alpha=0.3)
            plt.xlabel('Нормированная частота')
            plt.ylabel('Фаза (°)')
            plt.title(f'{name}: центрирование + w1/w2')
            plt.grid(True)
            plt.legend()
            plt.ylim(*phase_ylim_deg)
            plt.tight_layout()

        res[name] = dict(
            phi_wrapped=phi_wrapped,
            phi_centered=phi_centered,
            dphi_dw=None,  # если понадобится — вернёшь градиент
            phi_c=phi_c,
            phi_before_points=(phi_bef_1, phi_bef_2),
            phi_after_points=(phi_aft_1, phi_aft_2),
        )

    # --------------------------------------------------------
    #   ДЕЭМБЕДДИНГ
    # --------------------------------------------------------
    def _deembed_ports(w, net: rf.Network, phi1_c, phi2_c, phi1_b=0.0, phi2_b=0.0):
        w = np.asarray(w, float).ravel()
        S = np.array(net.s, dtype=np.complex128, copy=True)
        phi1 = -2.0 * (phi1_c + phi1_b * w)
        phi2 = -2.0 * (phi2_c + phi2_b * w)
        S[:, 0, 0] = apply_phase_one(S[:, 0, 0], phi1)
        S[:, 1, 1] = apply_phase_one(S[:, 1, 1], phi2)
        S[:, 0, 1] = -apply_phase_one(S[:, 0, 1], 0.5 * (phi1 + phi2))
        S[:, 1, 0] = -apply_phase_one(S[:, 1, 0], 0.5 * (phi1 + phi2))
        return S

    S_corr = _deembed_ports(w, net, phi1_c, phi2_c, phi1_b, phi2_b)

    return res, S_corr


class PhaseLoadingExtractor:
    """
    Класс, объединяющий:
      1) Центрирование краёв фазы (φ_c для S11, S22);
      2) Подбор частотно-зависимой линейной составляющей b11, b22
         через нейросеть и локальную оптимизацию.
    """

    def __init__(self, inference_model, work_model, reference_filter):
        self.inference_model = inference_model
        self.work_model = work_model
        self.reference_filter = reference_filter

        # для "тёплого старта" оптимизации b11, b22
        self._last_x0 = np.array([0.0, 0.0], dtype=float)

        # для хранения последних коэффициентов
        self.last_phi1_c = None
        self.last_phi2_c = None
        self.last_b11_opt = None
        self.last_b22_opt = None

    # ------------------------------------------------------------------
    #                  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ------------------------------------------------------------------

    @staticmethod
    def make_network(frequencies, S11, S21, S22, name="Corrected"):
        n = len(frequencies)
        S = np.zeros((n, 2, 2), dtype=complex)
        S[:, 0, 0] = S11
        S[:, 0, 1] = S21
        S[:, 1, 0] = S21
        S[:, 1, 1] = S22
        ntw = rf.Network(
            frequency=rf.Frequency.from_f(frequencies, unit='Hz'),
            s=S,
            name=name,
        )
        return ntw

    @staticmethod
    def apply_phase_all(S11, S21, S22, w, a11, b11, a22, b22):
        """
        Та же логика, что и в твоей apply_phase_all:
        φ11 = a11 + b11*w
        φ22 = a22 + b22*w
        φ21 = 0.5*(φ11+φ22)
        """
        phi11 = a11 + b11 * w
        phi22 = a22 + b22 * w
        S11_corr = S11 * np.exp(-1j * phi11)
        S22_corr = S22 * np.exp(-1j * phi22)
        S21_corr = S21 * np.exp(-1j * 0.5 * (phi11 + phi22))
        return S11_corr, S21_corr, S22_corr

    # ------------------------------------------------------------------
    #          1. ЦЕНТРИРОВАНИЕ КРАЁВ ФАЗЫ (φ_c для S11, S22)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_phase_shift_fast_local(phi_wrapped, w, w1, w2, verbose=False):
        """
        Ускоренный поиск фазового сдвига по двум точкам (w1, w2)
        через minimize_scalar с дешёвым objective.
        """
        phi_interp = interp1d(w, phi_wrapped, kind='linear', bounds_error=True)

        phi1_0 = float(phi_interp(w1))
        phi2_0 = float(phi_interp(w2))
        shift0 = 0.5 * _wrap_to_pi(phi1_0 + phi2_0)

        def objective(shift):
            phi1 = _wrap_to_pi(phi1_0 - shift)
            phi2 = _wrap_to_pi(phi2_0 - shift)
            return abs(_wrap_to_pi(phi1 + phi2))

        result = minimize_scalar(
            objective,
            bounds=(-np.pi, +np.pi),
            method='bounded',
            options={"xatol": 1e-5},
        )
        best_shift = result.x
        if verbose:
            print(
                f"Оптимальный фазовый сдвиг (φ({w2}) = -φ({w1})): "
                f"{best_shift:.6f} рад ({np.degrees(best_shift):.2f}°)"
            )
        return best_shift

    def fit_edges(
        self,
        w,
        net: rf.Network,
        center_points=(-1.5, 1.5),
        phase_ylim_deg=(-180, 180),
        plot=True,
        verbose=True,
    ):
        """
        Твой fit_phase_edges_curvefit, упакованный как метод класса.
        Возвращает:
          - res: dict с фазой до/после и φ_c по портам
          - S_corr: массив S после деэмбеддинга по φ_c
        Плюс сохраняет last_phi1_c, last_phi2_c.
        """
        if center_points is None:
            w_left = w[0]
            w_right = w[-1]
            w_th = min(abs(w_left), abs(w_right))
            center_points = (-w_th, w_th)

        w = np.asarray(w, float).ravel()
        S = np.asarray(net.s)
        assert S.shape[-2:] == (2, 2)

        phi1_c = phi2_c = phi1_b = phi2_b = 0.0
        res = {}

        w1, w2 = center_points

        # ------------------ ОБРАБОТКА S11 и S22 ------------------
        for name, ij in [('S11', (0, 0)), ('S22', (1, 1))]:
            z = S[:, ij[0], ij[1]]
            phi_wrapped = np.angle(z)

            # 1) Центрирование фазы
            phi_shift = self._find_phase_shift_fast_local(phi_wrapped, w, w1, w2, verbose=verbose)
            phi_centered = _wrap_to_pi(phi_wrapped - phi_shift)
            phi_c = -0.5 * phi_shift

            # Логи до/после
            interp_phi_before = interp1d(w, phi_wrapped, kind='linear', bounds_error=True)
            interp_phi_after  = interp1d(w, phi_centered, kind='linear', bounds_error=True)

            phi_bef_1 = float(interp_phi_before(w1))
            phi_bef_2 = float(interp_phi_before(w2))
            phi_aft_1 = float(interp_phi_after(w1))
            phi_aft_2 = float(interp_phi_after(w2))

            if verbose:
                print(f"\n{name}:")
                print(f"  Исходная фаза в w1={w1}: {phi_bef_1:+.4f} рад ({np.degrees(phi_bef_1):+.2f}°)")
                print(f"  Исходная фаза в w2={w2}: {phi_bef_2:+.4f} рад ({np.degrees(phi_bef_2):+.2f}°)")
                print(f"  Центрированная фаза в w1={w1}: {phi_aft_1:+.4f} рад ({np.degrees(phi_aft_1):+.2f}°)")
                print(f"  Центрированная фаза в w2={w2}: {phi_aft_2:+.4f} рад ({np.degrees(phi_aft_2):+.2f}°)")
                print(f"  Разность после центрирования: phi(w2)+phi(w1) = {phi_aft_1 + phi_aft_2:+.4e} рад")
                print(f"  φ_c = {phi_c:.6f} рад ({np.degrees(phi_c):.2f}°)")

            if name == 'S11':
                phi1_c = phi_c
            else:
                phi2_c = phi_c

            if plot:
                plt.figure(figsize=(8, 4))
                plt.plot(w, np.degrees(phi_wrapped), '--', alpha=0.5, label='Исходная φ')
                plt.plot(w, np.degrees(phi_centered), label='Центрированная φ')
                plt.axvline(w1, color='r', linestyle='--', label=f'w1={w1}')
                plt.axvline(w2, color='g', linestyle='--', label=f'w2={w2}')
                plt.axhline(np.degrees(phi_aft_1), color='r', alpha=0.3)
                plt.axhline(np.degrees(phi_aft_2), color='g', alpha=0.3)
                plt.xlabel('Нормированная частота')
                plt.ylabel('Фаза (°)')
                plt.title(f'{name}: центрирование + w1/w2')
                plt.grid(True)
                plt.legend()
                plt.ylim(*phase_ylim_deg)
                plt.tight_layout()

            res[name] = dict(
                phi_wrapped=phi_wrapped,
                phi_centered=phi_centered,
                phi_c=phi_c,
                phi_before_points=(phi_bef_1, phi_bef_2),
                phi_after_points=(phi_aft_1, phi_aft_2),
            )

        # Сохраняем константы в объект
        self.last_phi1_c = phi1_c
        self.last_phi2_c = phi2_c

        # ------------------ ДЕЭМБЕДДИНГ по φ_c ------------------
        def _deembed_ports(w, net: rf.Network, phi1_c, phi2_c, phi1_b=0.0, phi2_b=0.0):
            w = np.asarray(w, float).ravel()
            S_loc = np.array(net.s, dtype=np.complex128, copy=True)
            phi1 = -2.0 * (phi1_c + phi1_b * w)
            phi2 = -2.0 * (phi2_c + phi2_b * w)
            S_loc[:, 0, 0] = apply_phase_one(S_loc[:, 0, 0], phi1)
            S_loc[:, 1, 1] = apply_phase_one(S_loc[:, 1, 1], phi2)
            S_loc[:, 0, 1] = -apply_phase_one(S_loc[:, 0, 1], 0.5 * (phi1 + phi2))
            S_loc[:, 1, 0] = -apply_phase_one(S_loc[:, 1, 0], 0.5 * (phi1 + phi2))
            return S_loc

        S_corr = _deembed_ports(w, net, phi1_c, phi2_c, phi1_b, phi2_b)

        return res, S_corr

    # ------------------------------------------------------------------
    #      2. ПОДБОР ЛИНЕЙНОЙ СОСТАВЛЯЮЩЕЙ (b11, b22) ЧЕРЕЗ NN
    # ------------------------------------------------------------------

    def _make_phase_objective(self, ntw_orig, w_norm):
        """
        Обёртка над твоим make_phase_objective, но как метод.
        """
        f = ntw_orig.f
        S_raw = ntw_orig.s

        S11_raw_full = S_raw[:, 0, 0]
        S21_raw_full = S_raw[:, 0, 1]
        S22_raw_full = S_raw[:, 1, 1]

        ntw_de = ntw_orig.copy()
        ntw_de.name = "De-embedded_for_NN"

        w_norm = np.asarray(w_norm, dtype=np.float64)

        def objective(params, verbose=False):
            b11_opt, b22_opt = params

            # 1) деэмбед по кандидату (только линейная часть)
            S11_de, S21_de, S22_de = self.apply_phase_all(
                S11_raw_full, S21_raw_full, S22_raw_full,
                w_norm,
                a11=0.0, b11=-b11_opt,
                a22=0.0, b22=-b22_opt,
            )

            S_new = np.empty_like(S_raw)
            S_new[:, 0, 0] = S11_de
            S_new[:, 0, 1] = S21_de
            S_new[:, 1, 0] = S21_de
            S_new[:, 1, 1] = S22_de
            ntw_de.s = S_new

            # 2) NN → CM + своя фазовая нагрузка
            pred_params = self.inference_model.predict_x(ntw_de)

            pred_filter = self.work_model.create_filter_from_prediction(
                ntw_de, self.reference_filter, pred_params, self.work_model.codec
            )
            S_pred = pred_filter.s
            S11_ideal = S_pred[:, 0, 0]
            S21_ideal = S_pred[:, 0, 1]
            S22_ideal = S_pred[:, 1, 1]

            # 3) фаза от сети
            a11_nn = float(pred_params.get("a11", 0.0))
            b11_nn = float(pred_params.get("b11", 0.0))
            a22_nn = float(pred_params.get("a22", 0.0))
            b22_nn = float(pred_params.get("b22", 0.0))

            # 4) суммарная фаза
            a11_total = a11_nn
            a22_total = a22_nn
            b11_total = b11_opt + b11_nn
            b22_total = b22_opt + b22_nn

            S11_final, S21_final, S22_final = self.apply_phase_all(
                S11_ideal, S21_ideal, S22_ideal,
                w_norm,
                a11=a11_total, b11=b11_total,
                a22=a22_total, b22=b22_total,
            )

            loss = (
                np.mean(np.abs(S11_final - S11_raw_full)) +
                np.mean(np.abs(S21_final - S21_raw_full)) +
                np.mean(np.abs(S22_final - S22_raw_full))
            )

            if verbose:
                print(
                    f"[b11_opt={b11_opt:.4f}, b22_opt={b22_opt:.4f}] "
                    f"b11_nn={b11_nn:.4f}, b22_nn={b22_nn:.4f}, "
                    f"loss={loss:.6f}"
                )

            return loss

        return objective

    @staticmethod
    def _coarse_grid_search(obj, b_min=-np.pi, b_max=np.pi, n_grid=25):
        b_vals = np.linspace(b_min, b_max, n_grid)

        best_loss = np.inf
        best_b11 = None
        best_b22 = None

        for b11 in b_vals:
            for b22 in b_vals:
                loss = obj((b11, b22))
                if loss < best_loss:
                    best_loss = loss
                    best_b11, best_b22 = b11, b22

        print(f"[GRID] best_loss={best_loss:.6f} at b11_opt={best_b11:.4f}, b22_opt={best_b22:.4f}")
        return np.array([best_b11, best_b22])

    @staticmethod
    def _refine_local(x0, obj):
        res = minimize(
            lambda x: obj(x),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 2e-4, 'fatol': 5e-4}
        )
        print(f"[NELDER-MEAD] x_opt={res.x}, loss={res.fun:.6f}, success={res.success}")
        return res

    def optimize_phase_loading(self, ntw_orig, w_norm, use_grid=False):
        """
        Подбор b11_opt, b22_opt для заданного Network и нормированной оси w_norm.
        Возвращает OptimizeResult; обновляет self.last_b11_opt, self.last_b22_opt.
        """
        start_time = time.time()

        w_norm = np.asarray(w_norm, dtype=np.float64)
        obj = self._make_phase_objective(ntw_orig, w_norm)

        if use_grid:
            x0 = self._coarse_grid_search(obj, b_min=-np.pi, b_max=np.pi, n_grid=25)
        else:
            x0 = self._last_x0

        res = self._refine_local(x0, obj)
        self._last_x0 = res.x.copy()
        self.last_b11_opt, self.last_b22_opt = res.x

        stop_time = time.time()
        print(f"Optimize phase loading time: {stop_time - start_time:.3f} sec")
        return res

    def extract_all(
        self,
        ntw_orig: rf.Network,
        w_norm,
        center_points=None,
        phase_ylim_deg=(-180, 180),
        plot_edges=False,
        use_grid=False,
        verbose=True,
    ):
        """
        Полный цикл извлечения фазовой нагрузки:

          1) fit_edges:
             - находит φ_c для S11 и S22 по краям полосы (center_points);
             - снимает постоянную составляющую фазы с S-параметров;
             - возвращает S_corr (деэмбед по φ_c).

          2) optimize_phase_loading:
             - подбирает b11_opt, b22_opt (линейная составляющая)
               в нормированной области частот w_norm с учётом нейросети.

          3) формирует полностью деэмбеддированный Network
             (с удалённой постоянной и линейной составляющей фазы)
             по аналогии с твоим ручным кодом.
        """
        # --- 1. Снятие постоянной составляющей фазы (φ_c) ---
        edges_result, S_corr = self.fit_edges(
            w=w_norm,
            net=ntw_orig,
            center_points=center_points,
            phase_ylim_deg=phase_ylim_deg,
            plot=plot_edges,
            verbose=verbose,
        )

        # Network после деэмбеддинга по φ_c
        f = ntw_orig.f
        S11_corr = S_corr[:, 0, 0]
        S21_corr = S_corr[:, 0, 1]
        S22_corr = S_corr[:, 1, 1]
        ntw_edges_deembedded = self.make_network(
            f, S11_corr, S21_corr, S22_corr,
            name="deembedded_const_phase"
        )

        # --- 2. Оптимизация b11, b22 на уже центрированной сети ---
        opt_result = self.optimize_phase_loading(
            ntw_orig=ntw_edges_deembedded,
            w_norm=w_norm,
            use_grid=use_grid,
        )
        b11_opt = self.last_b11_opt
        b22_opt = self.last_b22_opt

        # --- 3. Полный деэмбеддинг линейной фазы (как в твоём коде) ---
        w = np.asarray(w_norm, dtype=float)

        S_lin = np.array(ntw_edges_deembedded.s, dtype=np.complex128, copy=True)

        # фазы на портах
        phi11_lin = w * b11_opt
        phi22_lin = w * b22_opt
        phi21_lin = 0.5 * (phi11_lin + phi22_lin)

        # применяем обратную фазу: умножаем на exp(+j * φ)
        S_lin[:, 0, 0] = apply_phase_one(S_lin[:, 0, 0], -phi11_lin)
        S_lin[:, 1, 1] = apply_phase_one(S_lin[:, 1, 1], -phi22_lin)
        S_lin[:, 0, 1] = apply_phase_one(S_lin[:, 0, 1], -phi21_lin)
        S_lin[:, 1, 0] = apply_phase_one(S_lin[:, 1, 0], -phi21_lin)

        ntw_full_deembedded = rf.Network(
            frequency=ntw_edges_deembedded.frequency,
            s=S_lin,
            name="fully_deembedded_phase",
        )

        result = {
            "phi1_c": self.last_phi1_c,
            "phi2_c": self.last_phi2_c,
            "b11_opt": b11_opt,
            "b22_opt": b22_opt,
            "ntw_deembedded": ntw_full_deembedded,     # φ_c + линейная часть сняты
        }
        return result


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
    # tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/narrowband")
    tds = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/24.10.25/non-shifted")
    tds_cst = TouchstoneDataset(f"filters/FilterData/{configs.FILTER_NAME}/measure/cst")
    cst_fil = tds_cst[i][1]
    tds_fil = tds[i][1]
    # matrix = CouplingMatrix.from_file(f"filters/FilterData/{configs.FILTER_NAME}/measure/cst/16.txt")
    # matrix_fil = MWFilter(order=matrix.matrix_order-2, f0=work_model.orig_filter.f0, matrix=matrix.matrix,Q=work_model.orig_filter.Q, frequency=cst_fil.f, bw=work_model.orig_filter.bw)
    # matrix_S = matrix_fil.response(cst_fil.f/1e6)

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

    for i in range(4, 7):
        net = tds[i][1]
        cst_fil = tds_cst[i][1]
        w_norm = MWFilter.freq_to_nfreq(net.f/1e6, work_model.orig_filter.f0, work_model.orig_filter.bw)

        # ntw_de, best_M, metrics, details = de_embedding_2p_network_autoM(
        #     net,
        #     N=work_model.orig_filter.order,
        #     CF=work_model.orig_filter.f0 * 1e6,
        #     BW=work_model.orig_filter.bw * 1e6,
        #     M_candidates=(0, 1, 2, 3),
        #     Line=1.5,
        #     mag_tol_db=0.2,
        #     phase_tol_deg=5.0,
        #     plot_debug=False,
        # )

        res, S_def = fit_phase_edges_curvefit(
            w_norm, net,
            # q=0.35,
            center_points=None,
            plot=False
        )


        # w_ext, s11_ext, s21_ext = cauchy_method.extract_coeffs(freq=net.f / 1e6, Q=work_model.orig_filter.Q, f0=work_model.orig_filter.f0,
        #                              s11=net.s[:, 0, 0], s21=-net.s[:, 1, 0], N=work_model.orig_filter.order + 2,
        #                              nz=6 + 2, bw=work_model.orig_filter.bw)
        # f_ext = MWFilter.nfreq_to_freq(w_ext, work_model.orig_filter.f0, work_model.orig_filter.bw)*1e6
        #
        # # w — нормированные частоты, net.s[:,0,0] — S11
        # s11_ext *= np.exp(1j * -(2*res['S11']['phi_c']))
        # dl = C_right
        # print(f"dl={dl}")
        # S_corr = s11_ext*np.exp(1j * (dl*w_ext))

        # phi0_opt, dl_opt, phi_corr_wrapped, S_def = estimate_phi0_dl_wrapped_from_vectors(
        #     f_hz=net.f,
        #     S_vec=net.s[:, 0, 0],
        #     w_norm=w_norm,
        #     w0=None,
        #     eps_mu=1/3e8
        # )

        # res, S_def = fit_phase_edges_curvefit(
        #     w_norm, net,
        #     # q=0.35,
        #     center_points=(-1.685, 1.685),
        #     correct_freq_dependence=True,
        #     fit_on_extrapolated=True,
        #     plot=False
        # )

        # res_list.append(res)
        # net.s = S_def
        w_ext, s11_ext, s21_ext = cauchy_method.extract_coeffs(freq=net.f / 1e6, Q=work_model.orig_filter.Q, f0=work_model.orig_filter.f0,
                                     s11=S_def[:, 0, 0], s21=S_def[:, 1, 1], N=work_model.orig_filter.order,
                                     nz=work_model.orig_filter.order, bw=work_model.orig_filter.bw)
        f_ext = MWFilter.nfreq_to_freq(w_ext, work_model.orig_filter.f0, work_model.orig_filter.bw) * 1e6

        gd11 = np.gradient(np.angle(s11_ext), w_ext)
        # plt.plot(w, min(gvz[0], gvz[-1])*np.ones_like(w))
        # plt.title("ГВЗ S11")
        print(f"For S11 gvz: {gd11[0]:.3f}, {gd11[-1]:.3f}")
        # A_right, C_right, info_right = fit_gvz_hyperbola_one_side(
        #     w=w_ext,
        #     tau=gd11,
        #     side="left",
        #     w_start=2.0,  # откуда начинать fit на правой стороне
        #     w_end=None,  # опционально, до куда
        #     plot=False,
        # )
        C_right = gd11[0] - gd11[-1]
        print(f"C_rigth for S11: {-0.5*C_right} rad, {np.degrees(-0.5*C_right)} deg")
        S11_corr = s11_ext * np.exp(1j * (C_right*w_ext))

        gd22 = np.gradient(np.angle(s21_ext), w_ext)
        print(f"For S11 gvz: {gd22[0]:.3f}, {gd22[-1]:.3f}")
        # A_right, C_right, info_right = fit_gvz_hyperbola_one_side(
        #     w=w_ext,
        #     tau=gd22,
        #     side="left",
        #     w_start=2.0,  # откуда начинать fit на правой стороне
        #     w_end=None,  # опционально, до куда
        #     plot=False,
        # )
        C_right = gd22[0] - gd22[-1]
        print(f"C_rigth for S22: {-0.5 * C_right} rad, {np.degrees(-0.5 * C_right)} deg")
        S22_corr = s21_ext * np.exp(1j * (C_right * w_ext))

        plt.figure()
        cst_fil.plot_s_re(m=0, n=0, label='S11 Re CST')
        plt.plot(f_ext, np.real(S11_corr), ls=':')
        # ntw_de.plot_s_re(m=0, n=0, label='S11 Re Corr', ls=':')
        cst_fil.plot_s_im(m=0, n=0, label='S11 Im CST')
        plt.plot(f_ext, np.imag(S11_corr), ls='--')
        # ntw_de.plot_s_im(m=0, n=0, label='S11 Im Corr', ls='--')
        plt.title("Im, Re part of S11 after correction")

        # plt.figure()
        # plt.plot(net.f, np.gradient(np.angle(S_corr), net.f))
        #
        # plt.figure()
        # cst_fil.plot_s_re(m=1, n=0, label='S21 Re CST')
        # # net.plot_s_re(m=1, n=0, label='S21 Re Corr', ls=':')
        # ntw_de.plot_s_re(m=1, n=0, label='S21 Re Corr', ls=':')
        # cst_fil.plot_s_im(m=1, n=0, label='S21 Im CST')
        # # net.plot_s_im(m=1, n=0, label='S21 Im Corr', ls='--')
        # ntw_de.plot_s_im(m=1, n=0, label='S21 Im Corr', ls='--')
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

