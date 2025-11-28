import numpy as np
from scipy.optimize import minimize
from scipy import optimize
import torch
import time
import matplotlib.pyplot as plt
from filters.filter import MWFilter
from filters.filter.couplilng_matrix import CouplingMatrix
from scipy.optimize import minimize
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
from filters.datasets.theoretical_dataset_generator import DatasetMWFilter
from scipy.optimize import least_squares
from torch import compile
import sensivity


def create_bounds(cm: CouplingMatrix,
                  abs_floor_main: float = 0.2,   # для соседних (i,i+1)
                  abs_floor_cross: float = 0.01,    # для перекрёстных
                  abs_floor_diag: float = 0.1,     # для диагонали
                  rel_main: float = 2.0,
                  rel_cross: float = 3.0,
                  rel_diag: float = 2.0):
    M = cm.matrix.clone().detach().float()
    n = M.shape[0]
    Bmin = torch.zeros_like(M)
    Bmax = torch.zeros_like(M)

    for (i, j) in cm.links:
        v = float(M[i, j])
        if i == j:
            # width = max(abs_floor_diag, rel_diag * abs(v))
            width = abs_floor_diag
        elif j == i + 1:
            # width = max(abs_floor_main, rel_main * abs(v))
            width = abs_floor_main
        else:
            # width = max(abs_floor_cross, rel_cross * abs(v))
            width = abs_floor_cross

        Bmin[i, j] = v - width
        Bmax[i, j] = v + width
        Bmin[j, i] = Bmin[i, j]
        Bmax[j, i] = Bmax[i, j]

    return CouplingMatrix(Bmin), CouplingMatrix(Bmax)



# def mix_s_params(s_net0, s_data, lambda_):
#     return (1.0 - lambda_) * s_net0 + lambda_ * s_data
#
#
# def smooth_conv1d(y: torch.Tensor, win: int = 7) -> torch.Tensor:
#     # простое симметричное сглаживание (win нечётный)
#     win = int(win) if win % 2 == 1 else int(win+1)
#     k = torch.ones(win, dtype=y.dtype, device=y.device) / win
#     pad = win // 2
#     y_pad = torch.nn.functional.pad(y[None, None, :], (pad, pad), mode='replicate')
#     ys = torch.nn.functional.conv1d(y_pad, k[None, None, :]).squeeze()
#     return ys
#
#
# def parabolic_refine(xm1, x0, xp1, fm1, f0, fp1):
#     """
#     Параболическая интерполяция минимума по трём точкам (x-частоты, f-дБ).
#     Возвращает уточнённую частоту x* и значение f(x*).
#     """
#     # смещения по оси индексов: -1, 0, +1
#     denom = (fm1 - 2*f0 + fp1)
#     denom = denom if abs(denom) > 1e-12 else 1e-12
#     # смещение вершины параболы (в шагах дискретизации)
#     delta = 0.5 * (fm1 - fp1) / denom
#     # ограничим чтобы не выскочить за соседние узлы
#     delta = float(np.clip(delta, -1.0, 1.0))
#     # линейная интерполяция частоты
#     step = (xp1 - x0)  # равномерная сетка -> константа, но делаем обобщённо
#     x_star = x0 + delta * step
#     # значение в вершине (не обязательно нужно)
#     f_star = f0 - 0.25 * (fm1 - fp1) * delta
#     return x_star, f_star
#
# def detect_zeros_generic(
#     f_norm: torch.Tensor,     # [N]
#     s: torch.Tensor,          # [N] complex (S21, S11 или S22)
#     search_mask: torch.Tensor,# [N] bool: где ищем нули (ПП для отражений, СП для передачи)
#     smooth_win: int = 7,
#     min_prom_db: float = 12.0,
#     min_dist_pts: int = 5,
# ):
#     y_db = 20*torch.log10(torch.abs(s) + 1e-8)
#     y_db_s = smooth_conv1d(y_db, smooth_win)               # смягчаем шум
#     # лок. минимумы внутри маски
#     y = y_db_s
#     cond = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])          # минимум
#     cond = cond & search_mask[1:-1]
#     idx = torch.nonzero(cond).squeeze() + 1
#     if idx.numel() == 0:
#         return [], []
#
#     # фильтр по "prominence"
#     keep = []
#     N = y.shape[0]
#     for i in idx.tolist():
#         left, right = max(i-10,0), min(i+11,N)
#         bg = torch.quantile(y[left:right], 0.9)            # локальный фон
#         depth = float(bg - y[i])                           # ск. глубина (дБ)
#         if depth >= min_prom_db:
#             keep.append(i)
#
#     # разнесём минимумы
#     keep = sorted(keep)
#     filtered, last = [], -10**9
#     for i in keep:
#         if i - last >= min_dist_pts:
#             filtered.append(i); last = i
#
#     # уточнение частоты параболой
#     zeros_f, zeros_i = [], []
#     for i in filtered:
#         if 0 < i < N-1:
#             x_star, _ = parabolic_refine(
#                 f_norm[i-1].item(), f_norm[i].item(), f_norm[i+1].item(),
#                 y[i-1].item(),       y[i].item(),      y[i+1].item()
#             )
#             zeros_f.append(float(x_star)); zeros_i.append(int(i))
#         else:
#             zeros_f.append(float(f_norm[i].item())); zeros_i.append(int(i))
#     return zeros_f, zeros_i
#

# def optimize_cm(pred_filter: DatasetMWFilter, orig_filter: DatasetMWFilter):
#     import time
#     import numpy as np
#     import torch
#     from scipy.optimize import minimize
#     import matplotlib.pyplot as plt
#
#     # ---------- helpers ----------
#     def gaussian_windows(f_norm: torch.Tensor, centers, sigma=0.7):
#         w = torch.zeros_like(f_norm, dtype=torch.float32)
#         for c in centers:
#             w = w + torch.exp(-0.5 * ((f_norm - float(c)) / sigma) ** 2)
#         return w
#
#     def detect_zeros_generic(f_norm: torch.Tensor, s: torch.Tensor, search_mask: torch.Tensor,
#                              smooth_win: int = 7, min_prom_db: float = 12.0, min_dist_pts: int = 5):
#         def to_db_safe(z: torch.Tensor, eps: float = 1e-8):
#             return 20 * torch.log10(torch.abs(z) + eps)
#
#         def smooth_conv1d(y: torch.Tensor, win: int = 7) -> torch.Tensor:
#             win = int(win) if win % 2 == 1 else int(win + 1)
#             k = torch.ones(win, dtype=y.dtype, device=y.device) / win
#             pad = win // 2
#             y_pad = torch.nn.functional.pad(y[None, None, :], (pad, pad), mode='replicate')
#             ys = torch.nn.functional.conv1d(y_pad, k[None, None, :]).squeeze()
#             return ys
#
#         y_db = to_db_safe(s)
#         y_db_s = smooth_conv1d(y_db, smooth_win)
#         y = y_db_s
#         N = y.shape[0]
#         cond = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
#         cond = cond & search_mask[1:-1]
#         idx = torch.nonzero(cond).squeeze() + 1
#         if idx.numel() == 0:
#             return [], []
#         keep = []
#         idx = idx.tolist()
#         if type(idx) == int:
#             idx = [idx]
#         for i in idx:
#             left, right = max(i - 10, 0), min(i + 11, N)
#             bg = torch.quantile(y[left:right], 0.9)
#             depth = float(bg - y[i])
#             if depth >= min_prom_db:
#                 keep.append(i)
#         keep = sorted(keep)
#         filtered, last = [], -10 ** 9
#         for i in keep:
#             if i - last >= min_dist_pts:
#                 filtered.append(i);
#                 last = i
#         zeros_f, zeros_i = [], []
#         for i in filtered:
#             zeros_f.append(float(f_norm[i].item()))
#             zeros_i.append(int(i))
#         return zeros_f, zeros_i
#
#     def detect_peaks_generic(f_norm: torch.Tensor, s: torch.Tensor, search_mask: torch.Tensor):
#         def to_db_safe(z: torch.Tensor, eps: float = 1e-8):
#             return 20 * torch.log10(torch.abs(z) + eps)
#
#         y_db = to_db_safe(s)
#         N = y_db.shape[0]
#         cond = (y_db[1:-1] > y_db[:-2]) & (y_db[1:-1] > y_db[2:])
#         cond = cond & search_mask[1:-1]
#         idx = torch.nonzero(cond).squeeze() + 1
#         peaks_f = [float(f_norm[i].item()) for i in idx.tolist()]
#         return peaks_f, idx.tolist()
#
#     def gradient_loss(mag_pred, mag_origin, weight=1.0):
#         grad_pred = mag_pred[1:] - mag_pred[:-1]
#         grad_origin = mag_origin[1:] - mag_origin[:-1]
#         return ((grad_pred - grad_origin) ** 2).mean() * weight
#
#     # ---------- prepare constants ----------
#     print("Start optimize (L-BFGS-B)")
#     start_time = time.time()
#     results = {}
#
#     matrix_order = pred_filter.coupling_matrix.matrix_order
#     links = pred_filter.coupling_matrix.links
#
#     # частотные веса
#     freq_weights = torch.tensor(
#         [min(abs(1.0 / float(f)) if float(f) != 0.0 else 1e9, 0.90) for f in pred_filter.f_norm],
#         dtype=torch.float32
#     )
#
#     # оригинальные цели
#     s11_origin = torch.tensor(orig_filter.s[:, 0, 0], dtype=torch.complex64)
#     s21_origin = torch.tensor(orig_filter.s[:, 1, 0], dtype=torch.complex64)
#     s22_origin = torch.tensor(orig_filter.s[:, 1, 1], dtype=torch.complex64)
#
#     # сохраняем оригинальные значения как константы
#     s11_origin_abs_orig = torch.abs(s11_origin).detach()
#     s21_origin_abs_orig = torch.abs(s21_origin).detach()
#     s22_origin_abs_orig = torch.abs(s22_origin).detach()
#
#     s11_origin_db_orig = DatasetMWFilter.to_db(s11_origin).detach()
#     s21_origin_db_orig = DatasetMWFilter.to_db(s21_origin).detach()
#     s22_origin_db_orig = DatasetMWFilter.to_db(s22_origin).detach()
#
#     f_norm_t = torch.tensor(pred_filter.f_norm, dtype=torch.float32)
#     pb = (-1, +1)
#     pass_mask = (f_norm_t >= pb[0]) & (f_norm_t <= pb[1])
#     stop_mask = ~pass_mask
#
#     # Обнаружение нулей ВСЕХ S-параметров
#     f0_21, idx_21 = detect_zeros_generic(f_norm_t, s21_origin, stop_mask,
#                                          min_prom_db=8.0, min_dist_pts=3)
#     f0_11, idx_11 = detect_zeros_generic(f_norm_t, s11_origin, pass_mask,  # S11 нули в полосе
#                                          min_prom_db=1.0, min_dist_pts=1)  # Более чувствительные параметры
#     f0_22, idx_22 = detect_zeros_generic(f_norm_t, s22_origin, pass_mask,  # S22 нули в полосе
#                                          min_prom_db=1.0, min_dist_pts=1)
#
#     print(f"Detected zeros - S21: {f0_21}, S11: {f0_11}, S22: {f0_22}")
#
#     # УЛУЧШЕННЫЕ веса для областей нулей ВСЕХ S-параметров
#     zero_weights_21 = gaussian_windows(f_norm_t, f0_21, sigma=0.01).detach() * 12.0  # Самый высокий вес
#     zero_weights_11 = gaussian_windows(f_norm_t, f0_11, sigma=0.015).detach() * 8.0  # Высокий вес
#     zero_weights_22 = gaussian_windows(f_norm_t, f0_22, sigma=0.015).detach() * 8.0  # Высокий вес
#
#     # ДОПОЛНИТЕЛЬНО: веса для пиков отражения (S11, S22)
#     peaks_11, _ = detect_peaks_generic(f_norm_t, s11_origin, pass_mask)
#     peaks_22, _ = detect_peaks_generic(f_norm_t, s22_origin, pass_mask)
#     peak_weights_11 = gaussian_windows(f_norm_t, peaks_11, sigma=0.025).detach() * 3.0
#     peak_weights_22 = gaussian_windows(f_norm_t, peaks_22, sigma=0.025).detach() * 3.0
#
#     # Обнаруживаем пики передачи
#     peaks_21, _ = detect_peaks_generic(f_norm_t, s21_origin, pass_mask)
#     peak_weights_21 = gaussian_windows(f_norm_t, peaks_21, sigma=0.02).detach() * 2.0
#
#     fast_calc = FastMN2toSParamCalculation(matrix_order=matrix_order,
#                                            wlist=pred_filter.f_norm,
#                                            Q=pred_filter.Q,
#                                            fbw=pred_filter.fbw)
#
#     mat_shape = pred_filter.coupling_matrix.matrix.shape
#     base_mat = torch.zeros(mat_shape, dtype=torch.float32)
#     i_idx = torch.tensor([i for i, _ in links], dtype=torch.long)
#     j_idx = torch.tensor([j for _, j in links], dtype=torch.long)
#
#     # ---------- IMPROVED cost function с учетом ВСЕХ нулей ----------
#     def cost_with_grad(x_np, targets):
#         # ---- УЛУЧШЕННЫЕ гиперпараметры ----
#         ZERO_WEIGHT_S21 = 12.0  # Самый высокий вес для нулей передачи
#         ZERO_WEIGHT_S11 = 8.0  # Высокий вес для нулей отражения
#         ZERO_WEIGHT_S22 = 8.0  # Высокий вес для нулей отражения
#         PEAK_WEIGHT = 2.0
#         SLOPE_WEIGHT = 1.5
#         SYMMETRY_WEIGHT = 2.0  # Новый вес для симметрии S11 и S22
#
#         x = torch.from_numpy(x_np.astype(np.float32)).clone().requires_grad_(True)
#
#         # распаковываем цели
#         s11_target_abs, s21_target_abs, s22_target_abs, s11_target_db, s21_target_db, s22_target_db = targets
#
#         # матрица из факторов
#         mat = base_mat.clone()
#         mat[i_idx, j_idx] = x
#         mat[j_idx, i_idx] = x
#
#         # S-параметры
#         # _, s11_pred, s21_pred, s22_pred = fast_calc.RespM2(mat, with_s22=True)
#         _, s11_pred, s21_pred = fast_calc.RespM2(mat, with_s22=False)
#
#         s11p_abs = torch.abs(s11_pred)
#         s21p_abs = torch.abs(s21_pred)
#         # s22p_abs = torch.abs(s22_pred)
#
#         # Базовые потери по амплитуде
#         diff_abs = torch.cat([
#             s11p_abs - s11_target_abs,
#             s21p_abs - s21_target_abs,
#             # s22p_abs - s22_target_abs,
#         ], dim=0)
#
#         l1_abs = diff_abs.abs().mean()
#         mse_abs = (diff_abs * diff_abs).mean()
#
#         # Потери в dB
#         def to_db_fast_from_abs(mag: torch.Tensor, eps: float = 1e-8):
#             return 20.0 * torch.log10(mag + eps)
#
#         s11p_db = to_db_fast_from_abs(s11p_abs)
#         s21p_db = to_db_fast_from_abs(s21p_abs)
#         # s22p_db = to_db_fast_from_abs(s22p_abs)
#
#         diff_db = torch.cat([
#             s11p_db - s11_target_db,
#             s21p_db - s21_target_db,
#             # s22p_db - s22_target_db,
#         ], dim=0)
#
#         l1_db = diff_db.abs().mean()
#         mse_db = (diff_db * diff_db).mean()
#
#         # УСИЛЕННЫЕ потери для нулей ВСЕХ S-параметров
#         zero_loss_21 = ((s21p_abs ** 2) * zero_weights_21).mean() * ZERO_WEIGHT_S21
#         zero_loss_11 = ((s11p_abs ** 2) * zero_weights_11).mean() * ZERO_WEIGHT_S11
#         # zero_loss_22 = ((s22p_abs ** 2) * zero_weights_22).mean() * ZERO_WEIGHT_S22
#
#         # Потери для сохранения пиков ВСЕХ S-параметров
#         peak_loss_21 = ((s21p_abs - s21_target_abs) ** 2 * peak_weights_21).mean() * PEAK_WEIGHT
#         peak_loss_11 = ((s11p_abs - s11_target_abs) ** 2 * peak_weights_11).mean() * PEAK_WEIGHT
#         # peak_loss_22 = ((s22p_abs - s22_target_abs) ** 2 * peak_weights_22).mean() * PEAK_WEIGHT
#
#         # Потери для крутых склонов ВСЕХ S-параметров
#         slope_loss_21 = gradient_loss(s21p_abs, s21_target_abs, SLOPE_WEIGHT)
#         slope_loss_11 = gradient_loss(s11p_abs, s11_target_abs, SLOPE_WEIGHT)
#         # slope_loss_22 = gradient_loss(s22p_abs, s22_target_abs, SLOPE_WEIGHT)
#
#         # НОВЫЕ: потери для симметрии S11 и S22
#         # symmetry_loss = torch.mean((s11p_abs - s22p_abs) ** 2) * SYMMETRY_WEIGHT
#
#         # Комбинированные потери
#         loss_core = (l1_abs + mse_abs + l1_db + mse_db) * 3.0
#         loss_zeros = zero_loss_21 + zero_loss_11 #+ zero_loss_22
#         loss_peaks = peak_loss_21 + peak_loss_11 #+ peak_loss_22
#         loss_slopes = slope_loss_21 + slope_loss_11 #+ slope_loss_22
#
#         total_loss = torch.sqrt(loss_core) + loss_zeros + loss_peaks + loss_slopes #+ symmetry_loss
#
#         if torch.isnan(total_loss) or torch.isinf(total_loss):
#             total_loss = torch.tensor(1e6, dtype=torch.float32)
#
#         total_loss.backward()
#         gx = x.grad.detach().cpu().numpy()
#
#         # УЛУЧШЕННАЯ отладочная информация - все нули
#         if np.random.random() < 0.02:
#             current_zeros_21, _ = detect_zeros_generic(f_norm_t, s21_pred, stop_mask, min_prom_db=3.0)
#             current_zeros_11, _ = detect_zeros_generic(f_norm_t, s11_pred, pass_mask, min_prom_db=2.0)
#             # current_zeros_22, _ = detect_zeros_generic(f_norm_t, s22_pred, pass_mask, min_prom_db=2.0)
#             print(f"Loss: {total_loss.item():.4f}")
#             print(f"  S21 zeros: {current_zeros_21}")
#             print(f"  S11 zeros: {current_zeros_11}")
#             # print(f"  S22 zeros: {current_zeros_22}")
#
#         return float(total_loss.detach().cpu().item()), gx
#
#     # ---------- optimization ----------
#     x_current = pred_filter.coupling_matrix.factors
#
#     Mmin, Mmax = create_bounds(
#         CouplingMatrix(CouplingMatrix.from_factors(torch.tensor(x_current, dtype=torch.float64),
#                                                    pred_filter.coupling_matrix.links,
#                                                    pred_filter.coupling_matrix.matrix_order)))
#     bounds = [(min(m_min, m_max), max(m_min, m_max)) for m_min, m_max in tuple(zip(Mmin.factors, Mmax.factors))]
#
#     # Многошаговая оптимизация
#     lambda_values = np.linspace(0.2, 1.0, 5)
#
#     best_result = None
#     best_loss = float('inf')
#
#     for step, lambda_val in enumerate(lambda_values):
#         print(f"Step {step + 1}/{len(lambda_values)}: lambda={lambda_val:.2f}")
#
#         # Смешиваем цели
#         s11_pred_init = torch.tensor(pred_filter.s[:, 0, 0], dtype=torch.complex64)
#         s21_pred_init = torch.tensor(pred_filter.s[:, 1, 0], dtype=torch.complex64)
#         s22_pred_init = torch.tensor(pred_filter.s[:, 1, 1], dtype=torch.complex64)
#
#         s11_mixed = lambda_val * s11_origin + (1 - lambda_val) * s11_pred_init
#         s21_mixed = lambda_val * s21_origin + (1 - lambda_val) * s21_pred_init
#         s22_mixed = lambda_val * s22_origin + (1 - lambda_val) * s22_pred_init
#
#         # Подготавливаем цели для cost function
#         targets = (
#             torch.abs(s11_mixed).detach(),
#             torch.abs(s21_mixed).detach(),
#             torch.abs(s22_mixed).detach(),
#             DatasetMWFilter.to_db(s11_mixed).detach(),
#             DatasetMWFilter.to_db(s21_mixed).detach(),
#             DatasetMWFilter.to_db(s22_mixed).detach()
#         )
#
#         result = minimize(
#             fun=cost_with_grad,
#             x0=x_current.detach().cpu().numpy().astype(np.float64),
#             method='L-BFGS-B',
#             jac=True,
#             bounds=bounds,
#             args=(targets,),
#             options={'disp': True, 'maxiter': 10000, 'ftol': 1e-8, 'gtol': 1e-8}
#         )
#
#         if result.fun < best_loss and result.success:
#             best_loss = result.fun
#             best_result = result
#             x_current = torch.tensor(result.x, dtype=torch.float32)
#             print(f"New best loss: {best_loss:.6f}")
#
#     if best_result is None:
#         best_result = result
#         print("Using last result as best")
#
#     stop_time = time.time()
#     print(f"Total optimize time: {stop_time - start_time:.3f} sec")
#
#     optim_matrix = CouplingMatrix.from_factors(
#         torch.tensor(best_result.x, dtype=torch.float32),
#         links,
#         matrix_order
#     )
#
#     # ---------- IMPROVED results visualization ----------
#     w, s11_opt, s21_opt, s22_opt = fast_calc.RespM2(optim_matrix, with_s22=True)
#     s11_opt_db = MWFilter.to_db(s11_opt)
#     s21_opt_db = MWFilter.to_db(s21_opt)
#     s22_opt_db = MWFilter.to_db(s22_opt)
#
#     # Проверяем конечные нули
#     final_zeros_21, _ = detect_zeros_generic(f_norm_t, s21_opt, stop_mask, min_prom_db=3.0)
#     final_zeros_11, _ = detect_zeros_generic(f_norm_t, s11_opt, pass_mask, min_prom_db=2.0)
#     final_zeros_22, _ = detect_zeros_generic(f_norm_t, s22_opt, pass_mask, min_prom_db=2.0)
#
#     print(f"Final zeros - S21: {final_zeros_21}, S11: {final_zeros_11}, S22: {final_zeros_22}")
#
#     plt.figure(figsize=(14, 10))
#
#     # S21
#     plt.subplot(2, 2, 1)
#     plt.plot(w, s21_origin_db_orig, label="S21 Origin", linewidth=2)
#     plt.plot(w, s21_opt_db, linestyle='--', label="S21 Optimized", linewidth=1.5)
#     for f_zero in f0_21:
#         plt.axvline(x=f_zero, color='red', linestyle=':', alpha=0.5, label='S21 zeros' if f_zero == f0_21[0] else "")
#     plt.legend()
#     plt.grid(True)
#     plt.title("S21 Transmission")
#     plt.ylim(-80, 5)
#
#     # S11
#     plt.subplot(2, 2, 2)
#     plt.plot(w, s11_origin_db_orig, label="S11 Origin", linewidth=2)
#     plt.plot(w, s11_opt_db, linestyle='--', label="S11 Optimized", linewidth=1.5)
#     for f_zero in f0_11:
#         plt.axvline(x=f_zero, color='blue', linestyle=':', alpha=0.5, label='S11 zeros' if f_zero == f0_11[0] else "")
#     plt.legend()
#     plt.grid(True)
#     plt.title("S11 Reflection")
#     plt.ylim(-30, 5)
#
#     # S22
#     plt.subplot(2, 2, 3)
#     plt.plot(w, s22_origin_db_orig, label="S22 Origin", linewidth=2)
#     plt.plot(w, s22_opt_db, linestyle='--', label="S22 Optimized", linewidth=1.5)
#     for f_zero in f0_22:
#         plt.axvline(x=f_zero, color='green', linestyle=':', alpha=0.5, label='S22 zeros' if f_zero == f0_22[0] else "")
#     plt.legend()
#     plt.grid(True)
#     plt.title("S22 Reflection")
#     plt.ylim(-30, 5)
#
#     # Все вместе
#     plt.subplot(2, 2, 4)
#     plt.plot(w, s11_origin_db_orig, label="S11 Origin", linewidth=2, alpha=0.7)
#     plt.plot(w, s21_origin_db_orig, label="S21 Origin", linewidth=2, alpha=0.7)
#     plt.plot(w, s22_origin_db_orig, label="S22 Origin", linewidth=2, alpha=0.7)
#     plt.plot(w, s11_opt_db, linestyle='--', label="S11 Opt", linewidth=1.5)
#     plt.plot(w, s21_opt_db, linestyle='--', label="S21 Opt", linewidth=1.5)
#     plt.plot(w, s22_opt_db, linestyle='--', label="S22 Opt", linewidth=1.5)
#     plt.legend()
#     plt.grid(True)
#     plt.title("All S-Parameters")
#     plt.ylim(-80, 5)
#
#     plt.tight_layout()
#
#     return CouplingMatrix(optim_matrix)
# def optimize_cm(pred_filter: DatasetMWFilter, orig_filter: DatasetMWFilter):
#     import time, numpy as np, torch
#     import matplotlib.pyplot as plt
#     from scipy.optimize import minimize
#
#     # ---------------- helpers ----------------
#     def smooth_conv1d(y: torch.Tensor, win: int = 7) -> torch.Tensor:
#         win = int(win) if win % 2 == 1 else int(win + 1)
#         k = torch.ones(win, dtype=y.dtype, device=y.device) / win
#         pad = win // 2
#         y_pad = torch.nn.functional.pad(y[None, None, :], (pad, pad), mode='replicate')
#         return torch.nn.functional.conv1d(y_pad, k[None, None, :]).squeeze()
#
#     def detect_zeros_generic(f_norm: torch.Tensor, s: torch.Tensor, search_mask: torch.Tensor,
#                              smooth_win: int = 7, min_prom_db: float = 12.0, min_dist_pts: int = 5):
#         """Локальные минимумы в dB с фильтрацией по prominence и разнесением."""
#         y_db = 20 * torch.log10(torch.abs(s) + 1e-8)
#         y_db_s = smooth_conv1d(y_db, smooth_win)
#         y = y_db_s
#         N = y.shape[0]
#         cond = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
#         cond = cond & search_mask[1:-1]
#         idx = torch.nonzero(cond).squeeze() + 1
#         if idx.numel() == 0:
#             return [], []
#         keep = []
#         for i in idx.tolist():
#             left, right = max(i - 10, 0), min(i + 11, N)
#             bg = torch.quantile(y[left:right], 0.9)
#             depth = float(bg - y[i])
#             if depth >= min_prom_db:
#                 keep.append(i)
#         keep = sorted(keep)
#         filtered, last = [], -10**9
#         for i in keep:
#             if i - last >= min_dist_pts:
#                 filtered.append(i); last = i
#         zeros_f, zeros_i = [], []
#         for i in filtered:
#             zeros_f.append(float(f_norm[i].item()))
#             zeros_i.append(int(i))
#         return zeros_f, zeros_i
#
#     def nearest_triplets_idx(f: torch.Tensor, centers: list[float]) -> torch.Tensor:
#         idx = []
#         for c in centers:
#             k = int(torch.argmin(torch.abs(f - float(c))))
#             k = max(1, min(f.numel()-2, k))
#             idx.append(k)
#         return torch.tensor(idx, dtype=torch.long, device=f.device)
#
#     def to_db_fast_from_abs(mag: torch.Tensor, eps: float = 1e-8):
#         return 20.0 * torch.log10(mag + eps)
#
#     def improved_bounds(cm: CouplingMatrix,
#                         abs_floor_main: float = 0.10,
#                         abs_floor_cross: float = 0.01,
#                         abs_floor_diag: float = 0.05,
#                         rel_main: float = 2.0,
#                         rel_cross: float = 3.0,
#                         rel_diag: float = 2.5):
#         """Границы, не зажимающие нулевые связи в [0,0]."""
#         M = cm.matrix.clone().detach().float()
#         Bmin = torch.zeros_like(M); Bmax = torch.zeros_like(M)
#         for (i, j) in cm.links:
#             v = float(M[i, j])
#             if i == j:
#                 width = max(abs_floor_diag, rel_diag * abs(v))
#             elif j == i + 1:
#                 width = max(abs_floor_main, rel_main * abs(v))
#             else:
#                 width = max(abs_floor_cross, rel_cross * abs(v))
#             Bmin[i, j] = v - width;  Bmax[i, j] = v + width
#             Bmin[j, i] = Bmin[i, j]; Bmax[j, i] = Bmax[i, j]
#         return CouplingMatrix(Bmin), CouplingMatrix(Bmax)
#
#     def parabolic_vertex(y_m1, y0, y_p1):
#         denom = (y_m1 - 2*y0 + y_p1) + 1e-18
#         delta = 0.5 * (y_m1 - y_p1) / denom
#         delta = torch.clamp(delta, -1.0, 1.0)
#         y_star = y0 - 0.25 * (y_m1 - y_p1) * delta
#         return delta, y_star
#
#     # -------------- prepare (full) --------------
#     print("Start optimize (L-BFGS-B)")
#     t0 = time.time()
#     matrix_order = pred_filter.coupling_matrix.matrix_order
#     links = pred_filter.coupling_matrix.links
#
#     # band = slice(:, :)
#     f_full = torch.tensor(pred_filter.f_norm, dtype=torch.float32)
#     pb = (-1.1, 1.1)
#     pass_mask_full = (f_full >= pb[0]) & (f_full <= pb[1])
#     stop_mask_full = ~pass_mask_full
#
#     margin = 2
#     stop_mask2 = stop_mask_full.clone()
#     stop_mask2[:margin] = False; stop_mask2[-margin:] = False
#
#     s11_full = torch.tensor(orig_filter.s[:, 0, 0], dtype=torch.complex64)
#     s21_full = torch.tensor(orig_filter.s[:, 1, 0], dtype=torch.complex64)
#     s22_full = torch.tensor(orig_filter.s[:, 1, 1], dtype=torch.complex64)
#
#     # нули на полной сетке
#     f0_21, _ = detect_zeros_generic(f_full, s21_full, stop_mask2)      # передача — в СП
#     f0_11, _ = detect_zeros_generic(f_full, s11_full, pass_mask_full)  # отражения — в ПП
#     f0_22, _ = detect_zeros_generic(f_full, s22_full, pass_mask_full)
#
#     # -------------- thin grid (speed-up) --------------
#     freq_weights_full = 1.0 / (torch.abs(f_full) + 1e-8)
#     freq_weights_full.clamp_(max=0.90)
#
#     keep = torch.zeros_like(f_full, dtype=torch.bool)
#     keep[0] = True; keep[-1] = True
#     pb_lo = torch.argmin(torch.abs(f_full - pb[0])); pb_hi = torch.argmin(torch.abs(f_full - pb[1]))
#     keep[pb_lo] = True; keep[pb_hi] = True
#
#     def mark_nearest(freqs, pad=2):
#         for c in freqs:
#             k = int(torch.argmin(torch.abs(f_full - float(c))))
#             k = max(2, min(f_full.numel()-3, k))
#             for t in range(-pad, pad+1):
#                 keep[k+t] = True
#     mark_nearest(f0_21, pad=2)
#     mark_nearest(f0_11, pad=2)
#     mark_nearest(f0_22, pad=2)
#
#     stride_pass, stride_stop = 2, 3
#     for i in range(f_full.numel()):
#         if keep[i]:
#             continue
#         if pass_mask_full[i] and (i % stride_pass == 0): keep[i] = True
#         if stop_mask_full[i] and (i % stride_stop == 0): keep[i] = True
#
#     # применяем маску
#     f = f_full[keep]
#     pass_mask = (f >= pb[0]) & (f <= pb[1])
#     stop_mask = ~pass_mask
#
#     freq_weights = freq_weights_full[keep]
#     s11_origin = s11_full[keep]; s21_origin = s21_full[keep]; s22_origin = s22_full[keep]
#
#     s11_origin_abs = torch.abs(s11_origin).detach()
#     s21_origin_abs = torch.abs(s21_origin).detach()
#     s22_origin_abs = torch.abs(s22_origin).detach()
#     s11_origin_db_w = (DatasetMWFilter.to_db(s11_origin) * freq_weights).detach()
#     s21_origin_db_w = (DatasetMWFilter.to_db(s21_origin) * freq_weights).detach()
#     s22_origin_db_w = (DatasetMWFilter.to_db(s22_origin) * freq_weights).detach()
#
#     # индексы триплетов на тонкой сетке и таргеты глубины/позиции
#     k21 = nearest_triplets_idx(f, f0_21)
#     k11 = nearest_triplets_idx(f, f0_11)
#     k22 = nearest_triplets_idx(f, f0_22)
#
#     def targets_from_origin(m_abs: torch.Tensor, f: torch.Tensor, k_idx: torch.Tensor):
#         if k_idx.numel() == 0:
#             z = torch.tensor(0.0, dtype=torch.float32, device=f.device)
#             return z, z
#         m2 = m_abs * m_abs
#         y_m1, y0, y_p1 = m2[k_idx-1], m2[k_idx], m2[k_idx+1]
#         delta, y_star = parabolic_vertex(y_m1, y0, y_p1)
#         df = (f[k_idx+1] - f[k_idx])
#         f_hat = f[k_idx] + delta * df
#         mag_star = torch.sqrt(torch.clamp(y_star, min=0.0) + 1e-16)
#         return mag_star.detach(), f_hat.detach()
#
#     tgt_mag21, tgt_f21 = targets_from_origin(torch.abs(s21_origin), f, k21)
#     tgt_mag11, tgt_f11 = targets_from_origin(torch.abs(s11_origin), f, k11)
#     tgt_mag22, tgt_f22 = targets_from_origin(torch.abs(s22_origin), f, k22)
#
#     # --- маска "стоп-полоса без окон нулей" для запрета ложных ям ---
#     R_ALLOW = 5  # ±точек вокруг целевых нулей
#     allow_21 = torch.zeros_like(f, dtype=torch.bool)
#     if k21.numel() > 0:
#         for t in range(-R_ALLOW, R_ALLOW + 1):
#             idx = torch.clamp(k21 + t, 0, f.numel() - 1)
#             allow_21[idx] = True
#     mask_sb_nozero = stop_mask & (~allow_21)
#     s21_abs_sb = torch.abs(s21_origin[mask_sb_nozero])
#     if s21_abs_sb.numel() > 0:
#         tgt_min_sb_db  = 20.0 * torch.log10(torch.quantile(s21_abs_sb, 0.05) + 1e-12)
#         tgt_min_sb_mag = (10.0 ** (tgt_min_sb_db / 20.0)).detach()
#     else:
#         tgt_min_sb_mag = torch.tensor(0.0, dtype=torch.float32, device=f.device)
#     mask_sb_nozero = mask_sb_nozero.detach()
#
#     # быстрый расчёт на тонкой сетке
#     fast_calc = FastMN2toSParamCalculation(matrix_order=matrix_order,
#                                            wlist=f.detach().cpu().numpy().astype(np.float32),
#                                            Q=pred_filter.Q,
#                                            fbw=pred_filter.fbw)
#
#     # быстрый «конструктор» матрицы
#     mat_shape = pred_filter.coupling_matrix.matrix.shape
#     base_mat = torch.zeros(mat_shape, dtype=torch.float32)
#     i_idx = torch.tensor([i for i, _ in links], dtype=torch.long)
#     j_idx = torch.tensor([j for _, j in links], dtype=torch.long)
#
#     # -------------- bounds + scaling (factors + Q) --------------
#     x0_factors = pred_filter.coupling_matrix.factors
#     Q0 = float(pred_filter.Q)
#
#     Mmin, Mmax = improved_bounds(
#         CouplingMatrix(CouplingMatrix.from_factors(torch.tensor(x0_factors, dtype=torch.float64),
#                                                    links, matrix_order))
#     )
#     lb = torch.tensor(Mmin.factors, dtype=torch.float32)
#     ub = torch.tensor(Mmax.factors, dtype=torch.float32)
#
#     q_low, q_high = 0.8 * Q0, 1.2 * Q0
#     lb = torch.cat([lb, torch.tensor([q_low], dtype=torch.float32)])
#     ub = torch.cat([ub, torch.tensor([q_high], dtype=torch.float32)])
#
#     x0 = torch.cat([x0_factors, torch.tensor([Q0], dtype=torch.float32)])
#     mid  = (lb + ub) / 2.0
#     half = torch.clamp((ub - lb) / 2.0, min=1e-3)
#
#     y0 = ((x0 - mid) / half).detach().cpu().numpy().astype(np.float64)
#     bounds_y = [(-1.0, 1.0)] * len(y0)
#
#     # -------------- loss builders --------------
#     def make_cost(lambda_zero: float, lambda_pos: float, lambda_sb: float):
#         ALPHA_SM = 800.0  # soft-min "temperature" для штрафа в СБ
#         def cost_with_grad(y_np, *args):
#             y = torch.from_numpy(y_np.astype(np.float32)).clone().requires_grad_(True)
#             x = mid + half * y
#             x_f = x[:-1]; q = x[-1]
#
#             # обновляем Q
#             fast_calc.update_Q(q)
#
#             # матрица из факторов
#             mat = base_mat.clone()
#             mat[i_idx, j_idx] = x_f
#             mat[j_idx, i_idx] = x_f
#
#             # _, s11_p, s21_p, s22_p = fast_calc.RespM2(mat, with_s22=True)
#             _, s11_p, s21_p = fast_calc.RespM2(mat, with_s22=False)
#
#             # базовый лосс (векторизовано)
#             s11a = torch.abs(s11_p); s21a = torch.abs(s21_p); # s22a = torch.abs(s22_p)
#             # diff_abs = torch.cat([s11a - s11_origin_abs, s21a - s21_origin_abs, s22a - s22_origin_abs], dim=0)
#             diff_abs = torch.cat([s11a - s11_origin_abs, s21a - s21_origin_abs], dim=0)
#             l1_abs  = diff_abs.abs().mean()
#             mse_abs = (diff_abs * diff_abs).mean()
#
#             s11d_w = to_db_fast_from_abs(s11a) * freq_weights
#             s21d_w = to_db_fast_from_abs(s21a) * freq_weights
#             # s22d_w = to_db_fast_from_abs(s22a) * freq_weights
#             # diff_db = torch.cat([s11d_w - s11_origin_db_w, s21d_w - s21_origin_db_w, s22d_w - s22_origin_db_w], dim=0)
#             diff_db = torch.cat([s11d_w - s11_origin_db_w, s21d_w - s21_origin_db_w], dim=0)
#             l1_db  = diff_db.abs().mean()
#             mse_db = (diff_db * diff_db).mean()
#
#             loss_core = torch.sqrt((l1_abs + mse_abs + l1_db + mse_db) * 3.0)
#
#             # прижим нулей к эталону (глубина+позиция)
#             def vertex_penalty_eq(m2, k_idx, tgt_mag, tgt_f):
#                 if k_idx.numel() == 0:
#                     z = torch.tensor(0.0, dtype=torch.float32, device=m2.device)
#                     return z, z
#                 y_m1, y0_, y_p1 = m2[k_idx-1], m2[k_idx], m2[k_idx+1]
#                 denom = (y_m1 - 2*y0_ + y_p1) + 1e-18
#                 delta = 0.5 * (y_m1 - y_p1) / denom
#                 delta = torch.clamp(delta, -1.0, 1.0)
#                 y_star = y0_ - 0.25 * (y_m1 - y_p1) * delta
#                 df_loc = (f[k_idx+1] - f[k_idx])
#                 f_hat = f[k_idx] + delta * df_loc
#                 mag_star = torch.sqrt(torch.clamp(y_star, min=0.0) + 1e-16)
#                 depth_pen = (20*torch.log10(mag_star + 1e-12) - 20*torch.log10(tgt_mag + 1e-12)).pow(2).mean()
#                 pos_pen   = (f_hat - tgt_f).pow(2).mean()
#                 return depth_pen, pos_pen
#
#             m11_2 = s11a * s11a; m21_2 = s21a * s21a; # m22_2 = s22a * s22a
#             d21, p21 = vertex_penalty_eq(m21_2, k21, tgt_mag21, tgt_f21)
#             d11, p11 = vertex_penalty_eq(m11_2, k11, tgt_mag11, tgt_f11)
#             # d22, p22 = vertex_penalty_eq(m22_2, k22, tgt_mag22, tgt_f22)
#             # loss_zero = lambda_zero * (d21 + d11 + d22) + lambda_pos * (p21 + p11 + p22)
#             loss_zero = lambda_zero * (d21 + d11) + lambda_pos * (p21 + p11)
#
#             # запрет на новые нули S21 в стоп-полосе (вне окон)
#             if mask_sb_nozero.any():
#                 m21 = s21a; m21_2 = m21 * m21
#                 logmask = torch.log(mask_sb_nozero.to(m21_2.dtype) + 1e-12)
#                 scores  = -ALPHA_SM * m21_2 + logmask
#                 softmin_m2  = -(1.0 / ALPHA_SM) * torch.logsumexp(scores, dim=0)
#                 softmin_mag = torch.sqrt(torch.clamp(softmin_m2, min=0.0) + 1e-16)
#                 spur_sb_pen = torch.relu(tgt_min_sb_mag - softmin_mag).pow(2)
#             else:
#                 spur_sb_pen = torch.tensor(0.0, dtype=torch.float32, device=m21_2.device)
#
#             loss = loss_core + loss_zero + lambda_sb * spur_sb_pen
#
#             loss.backward()
#             gy = (y.grad if y.grad is not None else torch.autograd.grad(loss, y)[0]).detach().cpu().numpy()
#             return float(loss.detach().cpu().item()), gy
#         return cost_with_grad
#
#     # -------------- two-phase L-BFGS-B --------------
#     res1 = minimize(make_cost(lambda_zero=0.0, lambda_pos=0.0, lambda_sb=0.0),
#                     y0, method='L-BFGS-B', jac=True, bounds=bounds_y,
#                     options={'disp': False, 'maxiter': 120, 'ftol': 3e-5, 'gtol': 3e-5, 'maxls': 25})
#     res2 = minimize(make_cost(lambda_zero=1.0, lambda_pos=0.3, lambda_sb=1.0),
#                     res1.x, method='L-BFGS-B', jac=True, bounds=bounds_y,
#                     options={'disp': True, 'maxiter': 1200, 'ftol': 1e-6, 'gtol': 1e-9, 'maxls': 50})
#
#     # -------------- build result --------------
#     y_opt = torch.tensor(res2.x, dtype=torch.float32)
#     x_opt = (mid + half * y_opt).detach()
#     x_opt_factors = x_opt[:-1]
#     Q_opt = float(x_opt[-1].item())
#
#     optim_matrix = CouplingMatrix.from_factors(x_opt_factors, links, matrix_order)
#     pred_filter._Q = Q_opt  # при желании фиксируем новое Q
#
#     print(f"Optimize time: {time.time() - t0:.3f} sec")
#
#     # -------------- plot on full grid (optional) --------------
#     fast_full = FastMN2toSParamCalculation(matrix_order=matrix_order,
#                                            wlist=pred_filter.f_norm,
#                                            Q=Q_opt, fbw=pred_filter.fbw)
#     w, s11_opt, s21_opt, s22_opt = fast_full.RespM2(optim_matrix, with_s22=True)
#
#     s11_origin_db = DatasetMWFilter.to_db(s11_full)
#     s21_origin_db = DatasetMWFilter.to_db(s21_full)
#     s22_origin_db = DatasetMWFilter.to_db(s22_full)
#     s11_opt_db = MWFilter.to_db(s11_opt)
#     s21_opt_db = MWFilter.to_db(s21_opt)
#     s22_opt_db = MWFilter.to_db(s22_opt)
#
#     plt.figure()
#     plt.title("S-параметры")
#     plt.plot(w, s11_origin_db, label="S11 Origin")
#     plt.plot(w, s11_opt_db, linestyle=':', label="S11 Optimized")
#     plt.plot(w, s21_origin_db, label="S21 Origin")
#     plt.plot(w, s21_opt_db, linestyle=':', label="S21 Optimized")
#     plt.plot(w, s22_origin_db, label="S22 Origin")
#     plt.plot(w, s22_opt_db, linestyle=':', label="S22 Optimized")
#     plt.legend(); plt.grid(True)
#
#     return CouplingMatrix(optim_matrix)
def optimize_cm(pred_filter: DatasetMWFilter,
                orig_filter: DatasetMWFilter,
                phase_init: tuple[float, float, float, float],
                plot: bool = True):
    import time, numpy as np, torch
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # ---------- helpers ----------
    def to_db_fast_from_abs(mag: torch.Tensor, eps: float = 1e-8):
        return 20.0 * torch.log10(mag + eps)

    def angle(z: torch.Tensor) -> torch.Tensor:
        return torch.atan2(z.imag, z.real)

    def phase_wrap(delta_phi: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))

    def _update_Q_safe(fast_calc, q_val):
        import torch
        # берем device из любой матрицы fast_calc
        dev = getattr(getattr(fast_calc, 'I', None), 'device', None)
        if dev is None:
            dev = getattr(getattr(fast_calc, 'R', None), 'device', torch.device('cpu'))
        q_t = q_val if isinstance(q_val, torch.Tensor) else torch.tensor(q_val)
        if q_t.dtype != torch.float32:
            q_t = q_t.to(torch.float32)
        if q_t.device != dev:
            q_t = q_t.to(dev)
        fast_calc.update_Q(q_t)

    def improved_bounds(cm: CouplingMatrix,
                        abs_floor_main: float = 0.05,
                        abs_floor_cross: float = 0.001,
                        abs_floor_diag: float = 0.05,
                        rel_main: float = 0.05,
                        rel_cross: float = 0.1,
                        rel_diag: float = 0.1):
        M = cm.matrix.clone().detach().float()
        Bmin = torch.zeros_like(M); Bmax = torch.zeros_like(M)
        for (i, j) in cm.links:
            v = float(M[i, j])
            # if i == j:        width = max(abs_floor_diag,  rel_diag  * abs(v))
            # elif j == i + 1:  width = max(abs_floor_main,  rel_main  * abs(v))
            # else:             width = max(abs_floor_cross, rel_cross * abs(v))
            if i == j:        width = rel_diag  * abs(v)
            elif j == i + 1:  width = rel_main  * abs(v)
            else:             width = rel_cross * abs(v)
            Bmin[i, j] = v - width;  Bmax[i, j] = v + width
            Bmin[j, i] = Bmin[i, j]; Bmax[j, i] = Bmax[i, j]
        return CouplingMatrix(Bmin), CouplingMatrix(Bmax)

    # веса компонентов лосса/отчёта
    W_MAG, W_REIM, W_PHASE = 1.0, 0.5, 0.5

    # ---------- prepare ----------
    print("Start optimize (L-BFGS-B) with phases")
    t0 = time.time()

    matrix_order = pred_filter.coupling_matrix.matrix_order
    links        = pred_filter.coupling_matrix.links

    # частоты и веса
    w = torch.tensor(pred_filter.f_norm, dtype=torch.float32)
    freq_weights = 1.0 / (torch.abs(w) + 1e-8)
    freq_weights = torch.clamp(freq_weights, max=0.90)

    # таргеты (комплекс)
    s11_true = torch.tensor(orig_filter.s[:, 0, 0], dtype=torch.complex64)
    s21_true = torch.tensor(orig_filter.s[:, 1, 0], dtype=torch.complex64)

    # предвычисления таргетов
    s11_abs_t = torch.abs(s11_true).detach()
    s21_abs_t = torch.abs(s21_true).detach()
    s11_db_wt = (DatasetMWFilter.to_db(s11_true) * freq_weights).detach()
    s21_db_wt = (DatasetMWFilter.to_db(s21_true) * freq_weights).detach()
    ph11_true = angle(s11_true).detach()
    ph21_true = angle(s21_true).detach()

    # быстрый расчёт на этой сетке
    fast_calc = FastMN2toSParamCalculation(matrix_order=matrix_order,
                                           wlist=w.detach().cpu().numpy().astype(np.float32),
                                           Q=pred_filter.Q,
                                           fbw=pred_filter.fbw)

    # быстрый «конструктор» матрицы
    mat_shape = pred_filter.coupling_matrix.matrix.shape
    base_mat  = torch.zeros(mat_shape, dtype=torch.float32)
    i_idx = torch.tensor([i for i, _ in links], dtype=torch.long)
    j_idx = torch.tensor([j for _, j in links], dtype=torch.long)

    # ---------- bounds + scaling (factors + Q + phases) ----------
    # матричные факторы
    x0_factors = pred_filter.coupling_matrix.factors

    # границы по факторам
    Mmin, Mmax = improved_bounds(
        CouplingMatrix(CouplingMatrix.from_factors(torch.tensor(x0_factors, dtype=torch.float64),
                                                   links, matrix_order))
    )
    lb_f = torch.tensor(Mmin.factors, dtype=torch.float32)
    ub_f = torch.tensor(Mmax.factors, dtype=torch.float32)

    # Q (±20%)
    Q0 = float(pred_filter.Q)
    q_lo, q_hi = 0.8 * Q0, 1.2 * Q0

    # фазы: (a11, a22, b11, b22) — максимум ±10% от старта, в пределах [-pi, pi]
    a11_0, a22_0, b11_0, b22_0 = map(float, phase_init)
    ph0  = torch.tensor([a11_0, a22_0, b11_0, b22_0], dtype=torch.float32)
    width = 0.1 * ph0.abs()  # ±10%
    lb_ph = torch.clamp(ph0 - width, min=-np.pi, max= np.pi)
    ub_ph = torch.clamp(ph0 + width, min=-np.pi, max= np.pi)

    # общий вектор x = [factors..., Q, phases(4)]
    lb = torch.cat([lb_f, torch.tensor([q_lo], dtype=torch.float32), lb_ph])
    ub = torch.cat([ub_f, torch.tensor([q_hi], dtype=torch.float32), ub_ph])

    x0 = torch.cat([x0_factors,
                    torch.tensor([Q0], dtype=torch.float32),
                    ph0])

    # нормировка к y∈[-1,1]
    mid  = (lb + ub) / 2.0
    half = torch.clamp((ub - lb) / 2.0, min=1e-6)
    y0   = ((x0 - mid) / half).detach().cpu().numpy().astype(np.float64)
    bounds_y = [(-1.0, 1.0)] * len(y0)

    # ---------- common simulate + metrics ----------
    def simulate_and_metrics(q_val: float, phases: torch.Tensor, factors: torch.Tensor):
        # обновляем Q
        _update_Q_safe(fast_calc, q_val)

        # сборка матрицы
        mat = base_mat.clone()
        mat[i_idx, j_idx] = factors
        mat[j_idx, i_idx] = factors

        # расчёт
        _, s11_p, s21_p = fast_calc.RespM2(mat, with_s22=False)

        # фаза
        a11, a22, b11, b22 = phases
        phi11 = 2.0 * (a11 + w * b11)
        phi21 = (a11 + a22 + w * (b11 + b22))

        s11_adj = s11_p * torch.exp(1j * phi11)
        s21_adj = (-s21_p) * torch.exp(1j * phi21)

        # --- метрики ---
        # амплитуды
        s11a = torch.abs(s11_adj); s21a = torch.abs(s21_adj)
        d_abs = torch.cat([s11a - s11_abs_t, s21a - s21_abs_t], dim=0)
        l1_abs   = d_abs.abs().mean()
        mse_abs  = (d_abs * d_abs).mean()
        rmse_abs = torch.sqrt(mse_abs + 1e-18)

        # dB (взвеш.)
        s11d_w = to_db_fast_from_abs(s11a) * freq_weights
        s21d_w = to_db_fast_from_abs(s21a) * freq_weights
        d_db   = torch.cat([s11d_w - s11_db_wt, s21d_w - s21_db_wt], dim=0)
        l1_db   = d_db.abs().mean()
        mse_db  = (d_db * d_db).mean()
        rmse_db = torch.sqrt(mse_db + 1e-18)

        # Re/Im
        reim = torch.cat([
            (s11_adj.real - s11_true.real),
            (s11_adj.imag - s11_true.imag),
            (s21_adj.real - s21_true.real),
            (s21_adj.imag - s21_true.imag),
        ], dim=0)
        mse_reim  = (reim * reim).mean()
        rmse_reim = torch.sqrt(mse_reim + 1e-18)

        # фаза
        ph11_pred = angle(s11_adj); ph21_pred = angle(s21_adj)
        dphi11 = phase_wrap(ph11_pred - ph11_true)
        dphi21 = phase_wrap(ph21_pred - ph21_true)
        mse_phase  = (dphi11 * dphi11).mean() + (dphi21 * dphi21).mean()
        rmse_phase = torch.sqrt(mse_phase + 1e-18)

        # целевая как в лоссе
        objective = torch.sqrt(
            W_MAG*(l1_abs + mse_abs + l1_db + mse_db) +
            W_REIM*mse_reim + W_PHASE*mse_phase
        )

        return {
            "L1_abs": float(l1_abs.item()),
            "RMSE_abs": float(rmse_abs.item()),
            "L1_dB_w": float(l1_db.item()),
            "RMSE_dB_w": float(rmse_db.item()),
            "RMSE_ReIm": float(rmse_reim.item()),
            "RMSE_phase": float(rmse_phase.item()),
            "Objective": float(objective.item())
        }

    # ---------- loss ----------
    def cost_with_grad(y_np, *args):
        y = torch.from_numpy(y_np.astype(np.float32)).clone().requires_grad_(True)
        x = mid + half * y

        nf = lb_f.numel()
        x_f = x[:nf]
        q   = x[nf]
        a11, a22, b11, b22 = x[nf+1:nf+5]

        # обновляем Q
        _update_Q_safe(fast_calc, q)

        # матрица
        mat = base_mat.clone()
        mat[i_idx, j_idx] = x_f
        mat[j_idx, i_idx] = x_f

        # расчёт
        _, s11_p, s21_p = fast_calc.RespM2(mat, with_s22=False)

        # фазы
        phi11 = 2.0 * (a11 + w * b11)
        phi21 = (a11 + a22 + w * (b11 + b22))
        s11_adj = s11_p * torch.exp(1j * phi11)
        s21_adj = (-s21_p) * torch.exp(1j * phi21)

        # модуль
        s11a = torch.abs(s11_adj); s21a = torch.abs(s21_adj)
        diff_abs = torch.cat([s11a - s11_abs_t, s21a - s21_abs_t], dim=0)
        l1_abs   = diff_abs.abs().mean()
        mse_abs  = (diff_abs * diff_abs).mean()

        # dB (взвеш.)
        s11d_w = to_db_fast_from_abs(s11a) * freq_weights
        s21d_w = to_db_fast_from_abs(s21a) * freq_weights
        diff_db = torch.cat([s11d_w - s11_db_wt, s21d_w - s21_db_wt], dim=0)
        l1_db   = diff_db.abs().mean()
        mse_db  = (diff_db * diff_db).mean()

        # Re/Im
        reim = torch.cat([
            (s11_adj.real - s11_true.real),
            (s11_adj.imag - s11_true.imag),
            (s21_adj.real - s21_true.real),
            (s21_adj.imag - s21_true.imag),
        ], dim=0)
        mse_reim = (reim * reim).mean()

        # фаза
        ph11_pred = angle(s11_adj); ph21_pred = angle(s21_adj)
        dphi11 = phase_wrap(ph11_pred - ph11_true)
        dphi21 = phase_wrap(ph21_pred - ph21_true)
        mse_phase = (dphi11 * dphi11).mean() + (dphi21 * dphi21).mean()

        loss_core = W_MAG*(l1_abs + mse_abs + l1_db + mse_db) + W_REIM*mse_reim + W_PHASE*mse_phase
        loss = torch.sqrt(loss_core)

        loss.backward()
        gy = y.grad.detach().cpu().numpy()
        return float(loss.detach().cpu().item()), gy

    # ---------- initial metrics (before optimization) ----------
    nf = lb_f.numel()
    init_metrics = simulate_and_metrics(
        q_val=Q0,
        phases=ph0,
        factors=x0_factors
    )
    print("[Initial] "
          f"L1_abs={init_metrics['L1_abs']:.6f} | RMSE_abs={init_metrics['RMSE_abs']:.6f} | "
          f"L1_dB_w={init_metrics['L1_dB_w']:.6f} | RMSE_dB_w={init_metrics['RMSE_dB_w']:.6f} | "
          f"RMSE_ReIm={init_metrics['RMSE_ReIm']:.6f} | RMSE_phase={init_metrics['RMSE_phase']:.6f} | "
          f"Objective={init_metrics['Objective']:.6f}")

    # ---------- optimize ----------
    res = minimize(
        fun=cost_with_grad,
        x0=y0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds_y,
        options={'disp': True, 'maxiter': 2500, 'ftol': 1e-6, 'gtol': 1e-9, 'maxls': 50}
    )

    y_opt = torch.tensor(res.x, dtype=torch.float32)
    x_opt = (mid + half * y_opt).detach()

    # распаковка результата
    x_opt_factors = x_opt[:nf]
    Q_opt  = float(x_opt[nf].item())
    a11_o, a22_o, b11_o, b22_o = map(float, x_opt[nf+1:nf+5].tolist())
    ph_opt = torch.tensor([a11_o, a22_o, b11_o, b22_o], dtype=torch.float32)

    # ---------- final metrics (after optimization) ----------
    final_metrics = simulate_and_metrics(
        q_val=Q_opt,
        phases=ph_opt,
        factors=x_opt_factors
    )
    print("[Final]   "
          f"L1_abs={final_metrics['L1_abs']:.6f} | RMSE_abs={final_metrics['RMSE_abs']:.6f} | "
          f"L1_dB_w={final_metrics['L1_dB_w']:.6f} | RMSE_dB_w={final_metrics['RMSE_dB_w']:.6f} | "
          f"RMSE_ReIm={final_metrics['RMSE_ReIm']:.6f} | RMSE_phase={final_metrics['RMSE_phase']:.6f} | "
          f"Objective={final_metrics['Objective']:.6f}")

    optim_matrix = CouplingMatrix.from_factors(x_opt_factors, links, matrix_order)
    pred_filter._Q = Q_opt

    print(f"Optimize time: {time.time() - t0:.3f} sec")

    # ---------- optional plots ----------
    if plot:
        fast_full = FastMN2toSParamCalculation(matrix_order=matrix_order,
                                               wlist=pred_filter.f_norm,
                                               Q=Q_opt, fbw=pred_filter.fbw)
        w_full, s11_p, s21_p = fast_full.RespM2(optim_matrix, with_s22=False)

        w_t = torch.tensor(pred_filter.f_norm, dtype=torch.float32)
        phi11 = 2.0 * (a11_o + w_t * b11_o)
        phi21 = (a11_o + a22_o + w_t * (b11_o + b22_o))
        s11_adj = s11_p * torch.exp(1j * phi11)
        s21_adj = (-s21_p) * torch.exp(1j * phi21)

        s11_opt_db = MWFilter.to_db(s11_adj)
        s21_opt_db = MWFilter.to_db(s21_adj)
        s11_origin_db = DatasetMWFilter.to_db(s11_true)
        s21_origin_db = DatasetMWFilter.to_db(s21_true)

        plt.figure(figsize=(5, 4))
        plt.title("S11/S21 (dB) with phase correction")
        plt.plot(w_full, s11_origin_db, label="S11 Origin")
        plt.plot(w_full, s11_opt_db, linestyle=':', label="S11 Optimized")
        plt.plot(w_full, s21_origin_db, label="S21 Origin")
        plt.plot(w_full, s21_opt_db, linestyle=':', label="S21 Optimized")
        plt.legend(); plt.grid(True)

    return CouplingMatrix(optim_matrix), Q_opt, (a11_o, a22_o, b11_o, b22_o)


