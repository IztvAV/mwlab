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


def create_bounds(origin_matrix: CouplingMatrix):
    Mmin = origin_matrix.matrix
    Mmax = origin_matrix.matrix

    for (i, j) in origin_matrix.links:
        if i == j:
            Mmin[i][j] -= 2*Mmin[i][j]
            Mmax[i][j] += 2*Mmax[i][j]
        elif j == (i + 1):
            Mmin[i][j] -= 0.5*Mmin[i][j]
            Mmin[j][i] -= 0.5*Mmin[j][i]
            Mmax[i][j] += 0.5*Mmax[i][j]
            Mmax[j][i] += 0.5*Mmax[j][i]
        else:
            Mmin[i][j] -= 2*Mmin[i][j]
            Mmin[j][i] -= 2*Mmin[j][i]
            Mmax[i][j] += 2*Mmax[i][j]
            Mmax[j][i] += 2*Mmax[j][i]
    return CouplingMatrix(torch.tensor(Mmin, dtype=torch.float32)), CouplingMatrix(torch.tensor(Mmax, dtype=torch.float32))



def optimize_cm(pred_filter: DatasetMWFilter, orig_filter: DatasetMWFilter):
    def cost_with_grad(x_np, *args):
        """Функция стоимости + градиент для scipy"""
        fast_calc, orig_filter, s11_origin_db, s21_origin_db, s22_origin_db, links, matrix_order = args
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        M = CouplingMatrix.from_factors(x, links, matrix_order)
        _, s11_pred, s21_pred = fast_calc.RespM2(M)

        def normalize(tensor: torch.Tensor):
            mean = tensor.mean()
            std = tensor.std()
            return (tensor - mean) / (std + 1e-8)  # добавим epsilon, чтобы избежать деления на 0
            # return tensor
            # min = tensor.min()
            # max = tensor.max()
            # return (tensor - min) / (max - min)

        s11_pred_db = normalize(MWFilter.to_db(s11_pred))
        s21_pred_db = normalize(MWFilter.to_db(s21_pred))
        s11_origin_db = normalize(s11_origin_db)
        s21_origin_db = normalize(s21_origin_db)

        # Loss
        loss = (
                1.0*torch.nn.L1Loss()(s11_pred_db, s11_origin_db) +
                1.0*torch.nn.L1Loss()(s21_pred_db, s21_origin_db) +
                1.0*torch.nn.MSELoss()(s11_pred_db, s11_origin_db) +
                1.0*torch.nn.MSELoss()(s21_pred_db, s21_origin_db)
        )
        reg = 0.01 * torch.norm(x, p=2)  # L2-регуляризация
        loss += reg

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

    s11_origin = orig_filter.s[:, 0, 0]
    s21_origin = orig_filter.s[:, 1, 0]
    s22_origin = orig_filter.s[:, 1, 1]

    # Преобразуем в тензоры один раз
    s11_origin_db = DatasetMWFilter.to_db(torch.tensor(s11_origin, dtype=torch.complex64))
    s21_origin_db = DatasetMWFilter.to_db(torch.tensor(s21_origin, dtype=torch.complex64))
    s22_origin_db = DatasetMWFilter.to_db(torch.tensor(s22_origin, dtype=torch.complex64))

    # [(min(0.5*xi, 1*xi), max(0.5*xi, 1 * xi)) if xi != 0 else (-0.5, 0.5) for xi in x0]

    start_time = time.time()
    Mmin, Mmax = create_bounds(pred_filter.coupling_matrix)
    bounds = [(min(m_min, m_max), max(m_min, m_max)) for m_min, m_max in tuple(zip(Mmin.factors, Mmax.factors))]
    result = minimize(
        fun=cost_with_grad,
        x0=x0,
        method='l-bfgs-b',
        jac=True,
        bounds=bounds,
        args=(fast_calc, orig_filter, s11_origin_db, s21_origin_db, s22_origin_db, links, matrix_order),
        options={'disp': True, 'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6}
    )
    print(f"Cost after L-BFGS-B: {result.fun}")

    # result = minimize(
    #     fun=cost_with_grad,
    #     x0=result.x,
    #     method='BFGS',
    #     jac=True,
    #     args=(fast_calc, orig_filter, s11_origin_db, s21_origin_db, s22_origin_db, links, matrix_order),
    #     options={'disp': True, 'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6, 'return_all': True}
    # )

    print(f"Cost after BFGS: {result.fun}")
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