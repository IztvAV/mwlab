# from filters.filter import MWFilter
# from filters.filter.couplilng_matrix import CouplingMatrix
# from scipy.optimize import minimize
# from filters.mwfilter_optim.base import FastMN2toSParamCalculation
#
# import torch
# import time
# import matplotlib.pyplot as plt
#
#
# def optimize_cm(pred_filter:MWFilter, orig_filter: MWFilter):
#     @torch.no_grad()
#     def cost(x, *args):
#         """ х - элементы матрицы связи (сначала главная диагональ D, потом D+1, потом побочная d, потом d+1"""
#         fast_calc, orig_filter, s11_origin, s21_origin = args
#         matrix = CouplingMatrix.from_factors(x, orig_filter.coupling_matrix.links, orig_filter.coupling_matrix.matrix_order)
#         _, s11_pred, s21_pred = fast_calc.RespM2(matrix)
#         cost = torch.sum(torch.abs(s21_origin - MWFilter.to_db(s21_pred))) + torch.sum(torch.abs(s11_origin - MWFilter.to_db(s11_pred)))
#         return cost.item()
#
#     x0_real = pred_filter.coupling_matrix.factors
#     x0 = x0_real
#     x0 = torch.round(torch.tensor(x0), decimals=5)
#     print("Start optimize")
#     fast_calc = FastMN2toSParamCalculation(matrix_order=orig_filter.coupling_matrix.matrix_order, wlist=orig_filter.f_norm, Q=orig_filter.Q, fbw=orig_filter.fbw) # Q=torch.inf потому что мы предсказываем на фильтре с потерями
#     s11_origin_db = orig_filter.s_db[:, 0, 0]
#     s21_origin_db = orig_filter.s_db[:, 1, 0]
#     start_time = time.time_ns()
#     prev_cost = 0
#     for _ in range(15):
#         optim_res = minimize(fun=cost, x0=x0, jac="2-points", method="BFGS",
#                              args=(fast_calc, orig_filter, s11_origin_db, s21_origin_db),
#                              options={"disp": True, "maxiter": 50})
#         x0 = optim_res.x
#
#         if optim_res.nit == 0:
#             print("Number of iteration is 0. Break loop")
#             break
#         elif abs(optim_res.fun - prev_cost) < 5e-1:
#             print("Different between cost function values less than 5e-1. Break loop")
#             break
#         elif abs(optim_res.fun) < 1:
#             print("Cost function value less than 1. Break loop")
#             break
#         prev_cost = optim_res.fun
#     stop_time = time.time_ns()
#     print(f"Optimize time: {(stop_time - start_time) / 1e9} sec")
#
#     optim_matrix = CouplingMatrix.from_factors(optim_res.x, orig_filter.coupling_matrix.links,
#                                                orig_filter.coupling_matrix.matrix_order)
#     w, s11_optim_resp, s21_optim_resp = fast_calc.RespM2(optim_matrix)
#     s11_optim_db = MWFilter.to_db(s11_optim_resp)
#     s21_optim_db = MWFilter.to_db(s21_optim_resp)
#
#     plt.figure()
#     plt.title("S-параметры")
#     plt.plot(w, s11_origin_db)
#     plt.plot(w, s11_optim_db, linestyle=':')
#     plt.plot(w, s21_origin_db)
#     plt.plot(w, s21_optim_db, linestyle=':')
#     plt.legend(["Origin", "Optimized"])
#
#     return CouplingMatrix(optim_matrix)

import numpy as np
from scipy.optimize import minimize
import torch
import time
import matplotlib.pyplot as plt

def optimize_cm(pred_filter: MWFilter, orig_filter: MWFilter):
    def cost_with_grad(x_np, *args):
        """Функция стоимости + градиент для scipy"""
        fast_calc, orig_filter, s11_origin_db, s21_origin_db, links, matrix_order = args
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        M = CouplingMatrix.from_factors(x, links, matrix_order)
        _, s11_pred, s21_pred = fast_calc.RespM2(M)

        loss = torch.sum(torch.abs(s11_origin_db - MWFilter.to_db(s11_pred))) + \
               torch.sum(torch.abs(s21_origin_db - MWFilter.to_db(s21_pred)))

        loss.backward()
        return loss.item(), x.grad.detach().numpy()

    print("Start optimize (L-BFGS-B)")
    x0 = np.array(pred_filter.coupling_matrix.factors, dtype=np.float64)

    matrix_order = orig_filter.coupling_matrix.matrix_order
    links = orig_filter.coupling_matrix.links
    fast_calc = FastMN2toSParamCalculation(matrix_order=matrix_order,
                                           wlist=orig_filter.f_norm,
                                           Q=orig_filter.Q,
                                           fbw=orig_filter.fbw)

    s11_origin_db = orig_filter.s_db[:, 0, 0]
    s21_origin_db = orig_filter.s_db[:, 1, 0]

    # Преобразуем в тензоры один раз
    s11_origin_db = torch.tensor(s11_origin_db, dtype=torch.float32)
    s21_origin_db = torch.tensor(s21_origin_db, dtype=torch.float32)

    # Ограничения на x (например, от -3 до 3)
    bounds = [(0.5 * xi, 1.5 * xi) if xi != 0 else (-1.0, 1.0) for xi in x0]

    start_time = time.time()

    result = minimize(
        fun=cost_with_grad,
        x0=x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        args=(fast_calc, orig_filter, s11_origin_db, s21_origin_db, links, matrix_order),
        options={'disp': True, 'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6}
    )

    stop_time = time.time()
    print(f"Optimize time: {stop_time - start_time:.3f} sec")

    # Постобработка результата
    optim_matrix = CouplingMatrix.from_factors(
        torch.tensor(result.x, dtype=torch.float32),
        links,
        matrix_order
    )
    w, s11_opt, s21_opt = fast_calc.RespM2(optim_matrix)
    s11_opt_db = MWFilter.to_db(s11_opt)
    s21_opt_db = MWFilter.to_db(s21_opt)

    plt.figure()
    plt.title("S-параметры")
    plt.plot(w, s11_origin_db, label="S11 Origin")
    plt.plot(w, s11_opt_db, linestyle=':', label="S11 Optimized")
    plt.plot(w, s21_origin_db, label="S21 Origin")
    plt.plot(w, s21_opt_db, linestyle=':', label="S21 Optimized")
    plt.legend()
    plt.grid(True)

    return CouplingMatrix(optim_matrix)
