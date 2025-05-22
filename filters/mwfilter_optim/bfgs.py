from filters.filter import MWFilter
from filters.filter.couplilng_matrix import CouplingMatrix
from scipy.optimize import minimize
from filters.mwfilter_optim.base import FastMN2toSParamCalculation

import torch
import time
import matplotlib.pyplot as plt


@torch.no_grad()
def optimize_cm(pred_filter:MWFilter, orig_filter: MWFilter):
    def cost(x, *args):
        """ х - элементы матрицы связи (сначала главная диагональ D, потом D+1, потом побочная d, потом d+1"""
        fast_calc, orig_filter, s11_origin, s21_origin = args
        matrix = CouplingMatrix.from_factors(x, orig_filter.coupling_matrix.links, orig_filter.coupling_matrix.matrix_order)
        _, s11_pred, s21_pred = fast_calc.RespM2_gpu(matrix)
        cost = torch.sum(torch.abs(s21_origin - MWFilter.to_db(s21_pred))) + torch.sum(torch.abs(s11_origin - MWFilter.to_db(s11_pred)))
        return cost.item()

    x0_real = pred_filter.coupling_matrix.factors
    x0 = x0_real
    x0 = torch.round(torch.tensor(x0), decimals=5)
    print("Start optimize")
    fast_calc = FastMN2toSParamCalculation(matrix_order=orig_filter.coupling_matrix.matrix_order, wlist=orig_filter.f_norm, Q=orig_filter.Q, fbw=orig_filter.fbw) # Q=torch.inf потому что мы предсказываем на фильтре с потерями
    s11_origin_db = orig_filter.s_db[:, 0, 0]
    s21_origin_db = orig_filter.s_db[:, 1, 0]
    start_time = time.time_ns()
    prev_cost = 0
    for _ in range(15):
        optim_res = minimize(fun=cost, x0=x0, jac="2-points", method="BFGS",
                             args=(fast_calc, orig_filter, s11_origin_db, s21_origin_db),
                             options={"disp": True, "maxiter": 50})
        x0 = optim_res.x

        if optim_res.nit == 0:
            print("Number of iteration is 0. Break loop")
            break
        elif abs(optim_res.fun - prev_cost) < 1e-2:
            print("Different between cost function values less than 1e-2. Break loop")
            break
        elif abs(optim_res.fun) < 1:
            print("Cost function value less than 1. Break loop")
            break
        prev_cost = optim_res.fun
    stop_time = time.time_ns()
    print(f"Optimize time: {(stop_time - start_time) / 1e9} sec")

    optim_matrix = CouplingMatrix.from_factors(optim_res.x, orig_filter.coupling_matrix.links,
                                               orig_filter.coupling_matrix.matrix_order)
    w, s11_optim_resp, s21_optim_resp = fast_calc.RespM2_gpu(optim_matrix)
    s11_optim_db = MWFilter.to_db(s11_optim_resp)
    s21_optim_db = MWFilter.to_db(s21_optim_resp)

    plt.figure()
    plt.title("S11")
    plt.plot(w, s11_origin_db, w, s11_optim_db)
    plt.legend(["Origin", "Optimized"])

    plt.figure()
    plt.title("S21")
    plt.plot(w, s21_origin_db, w, s21_optim_db)
    plt.legend(["Origin", "Optimized"])

    return optim_res.x