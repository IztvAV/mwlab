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


def create_bounds(origin_matrix: CouplingMatrix):
    Mmin = origin_matrix.matrix
    Mmax = origin_matrix.matrix

    for (i, j) in origin_matrix.links:
        if i == j:
            Mmin[i][j] -= 2*Mmin[i][j]
            Mmax[i][j] += 2*Mmax[i][j]
        elif j == (i + 1):
            Mmin[i][j] -= 2*Mmin[i][j]
            Mmin[j][i] -= 2*Mmin[j][i]
            Mmax[i][j] += 2*Mmax[i][j]
            Mmax[j][i] += 2*Mmax[j][i]
        else:
            Mmin[i][j] -= 2*Mmin[i][j]
            Mmin[j][i] -= 2*Mmin[j][i]
            Mmax[i][j] += 2*Mmax[i][j]
            Mmax[j][i] += 2*Mmax[j][i]
    return CouplingMatrix(torch.tensor(Mmin, dtype=torch.float32)), CouplingMatrix(torch.tensor(Mmax, dtype=torch.float32))



def mix_s_params(s_net0, s_data, lambda_):
    return (1.0 - lambda_) * s_net0 + lambda_ * s_data



def optimize_cm(pred_filter: DatasetMWFilter, orig_filter: DatasetMWFilter):
    def cost_for_partial_links(x_np, *args):
        """Функция стоимости + градиент для scipy"""
        fast_calc, orig_filter, s11_origin_db, s21_origin_db, s22_origin_db, links, matrix_order, matrix = args

        # Новый тензор с requires_grad
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        # Обновляем матрицу
        indices = torch.tensor(links, dtype=torch.long)
        # i_idx, j_idx = indices[:, 0], indices[:, 1]
        i_idx, j_idx = indices[0], indices[1]
        mat = matrix.clone().detach().requires_grad_(False)  # избежать inplace
        mat[i_idx, j_idx] = x
        mat[j_idx, i_idx] = x

        # Расчет S-параметров
        _, s11_pred, s21_pred, s22_pred = fast_calc.RespM2(mat, with_s22=True)

        def normalize(tensor: torch.Tensor):
            # mean = tensor.mean()
            # std = tensor.std()
            # return (tensor - mean) / (std + 1e-8)
            return tensor
            # min = tensor.min()
            # max = tensor.max()
            # return (tensor - min) / (max - min)

        # Приведение к dB и нормализация
        s11_pred_db = normalize(MWFilter.to_db(s11_pred))*freq_weights
        s21_pred_db = normalize(MWFilter.to_db(s21_pred))*freq_weights
        s22_pred_db = normalize(MWFilter.to_db(s22_pred))*freq_weights
        s11_origin_db = normalize(s11_origin_db)*freq_weights
        s21_origin_db = normalize(s21_origin_db)*freq_weights
        s22_origin_db = normalize(s22_origin_db)*freq_weights

        # Loss
        loss = (
                0.5 * torch.nn.functional.l1_loss(s11_pred_db, s11_origin_db) +
                1.0 * torch.nn.functional.l1_loss(s21_pred_db, s21_origin_db) +
                0.5 * torch.nn.functional.l1_loss(s22_pred_db, s22_origin_db) +
                0.5 * torch.nn.functional.mse_loss(s11_pred_db, s11_origin_db) +
                1.0 * torch.nn.functional.mse_loss(s21_pred_db, s21_origin_db) +
                0.5 * torch.nn.functional.mse_loss(s22_pred_db, s22_origin_db)
        )

        # Регуляризация
        reg = 1 * torch.norm(x, p=2)
        loss += reg
        reg = 0.1 * torch.nn.L1Loss()(s11_pred_db, s22_pred_db)
        loss += reg

        # backward
        loss.backward()

        # .detach() чтобы не тянуть граф дальше
        grad = x.grad.detach().cpu().numpy()
        return loss.item(), grad

    def cost_with_grad(x_np, *args):
        """Функция стоимости + градиент для scipy"""
        fast_calc, orig_filter, s11_origin_db, s21_origin_db, s22_origin_db, links, matrix_order = args
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        M = CouplingMatrix.from_factors(x, links, matrix_order)
        _, s11_pred, s21_pred, s22_pred = fast_calc.RespM2(M, with_s22=True)

        def normalize(tensor: torch.Tensor):
            # mean = tensor.mean()
            # std = tensor.std()
            # return (tensor - mean) / (std + 1e-8)  # добавим epsilon, чтобы избежать деления на 0
            return tensor
            # min = tensor.min()
            # max = tensor.max()
            # return (tensor - min) / (max - min)

        s11_pred_db = normalize(MWFilter.to_db(s11_pred))*freq_weights
        s21_pred_db = normalize(MWFilter.to_db(s21_pred))*freq_weights
        s22_pred_db = normalize(MWFilter.to_db(s22_pred))*freq_weights
        s11_origin_db = normalize(s11_origin_db)*freq_weights
        s21_origin_db = normalize(s21_origin_db)*freq_weights
        s22_origin_db = normalize(s22_origin_db)*freq_weights

        # Loss
        loss = (
                0.5*torch.nn.L1Loss()(s11_pred_db, s11_origin_db) +
                1.0*torch.nn.L1Loss()(s21_pred_db, s21_origin_db) +
                0.5*torch.nn.L1Loss()(s22_pred_db, s22_origin_db) +
                0.5*torch.nn.MSELoss()(s11_pred_db, s11_origin_db) +
                1.0*torch.nn.MSELoss()(s21_pred_db, s21_origin_db) +
                0.5*torch.nn.MSELoss()(s22_pred_db, s22_origin_db)
        )
        reg = 1 * torch.norm(x, p=2)  # L2-регуляризация
        loss += reg
        reg = 0.1 * torch.nn.L1Loss()(s11_pred_db, s22_pred_db)
        loss += reg


        loss.backward()
        return loss.item(), x.grad.detach().numpy()

    results = {}
    print("Start optimize (L-BFGS-B)")
    x0 = np.array(pred_filter.coupling_matrix.factors, dtype=np.float64)

    matrix_order = pred_filter.coupling_matrix.matrix_order
    links = pred_filter.coupling_matrix.links
    fast_calc = FastMN2toSParamCalculation(matrix_order=matrix_order,
                                           wlist=pred_filter.f_norm,
                                           Q=pred_filter.Q,
                                           fbw=pred_filter.fbw)
    freq_weights = torch.tensor([min(abs(1/f), 1) for f in pred_filter.f_norm], dtype=torch.float32)

    s11_origin = orig_filter.s[:, 0, 0]
    s21_origin = orig_filter.s[:, 1, 0]
    s22_origin = orig_filter.s[:, 1, 1]

    s11_pred = pred_filter.s[:, 0, 0]
    s21_pred = pred_filter.s[:, 1, 0]
    s22_pred = pred_filter.s[:, 1, 1]

    # Преобразуем в тензоры один раз
    s11_origin_db = DatasetMWFilter.to_db(torch.tensor(s11_origin, dtype=torch.complex64))
    s21_origin_db = DatasetMWFilter.to_db(torch.tensor(s21_origin, dtype=torch.complex64))
    s22_origin_db = DatasetMWFilter.to_db(torch.tensor(s22_origin, dtype=torch.complex64))

    s11_net0 = pred_filter.s[:, 0, 0]
    s21_net0 = pred_filter.s[:, 1, 0]
    s22_net0 = pred_filter.s[:, 1, 1]

    # [(min(0.5*xi, 1*xi), max(0.5*xi, 1 * xi)) if xi != 0 else (-0.5, 0.5) for xi in x0]

    start_time = time.time()

    lambda_values = np.linspace(0.05, 1.0, 20)  # Например, num_steps = 10
    r0, _ = cost_with_grad(x0, fast_calc, orig_filter, s11_origin_db, s21_origin_db, s22_origin_db, links, matrix_order)
    print("Initial cost value:", r0)

    # Миксуем S-данные: от нейросети к оригинальным
    s11_target = s11_origin_db
    s21_target = s21_origin_db
    s22_target = s22_origin_db

    # Частичная оптимизация параметров
    self_couplings = []
    main_couplings = []
    cross_couplings = []
    for i, j in pred_filter.coupling_matrix.links:
        if i == j:
            self_couplings.append((i, j))
        elif j == i+1:
            main_couplings.append((i, j))
        else:
            cross_couplings.append((i, j))

    optim_matrix = pred_filter.coupling_matrix.matrix
    for couplings in self_couplings:
        # Индексы связей
        indices = torch.tensor(couplings, dtype=torch.long)  # (L, 2)
        # i_idx, j_idx = indices[:, 0], indices[:, 1]
        i_idx, j_idx = indices[0], indices[1]
        x_current =  optim_matrix[i_idx, j_idx].numpy()
        # bounds = [(min(1.5*xi, 0.01*xi), max(1.5*xi, 0.01*xi)) for xi in x_current]
        bounds = [(min(1.5*x_current, 0.01*x_current), max(1.5*x_current, 0.01*x_current))]
        # bounds = [(-2, 2) for xi in x_current]

        result = minimize(
            fun=cost_for_partial_links,
            x0=x_current,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            args=(fast_calc, orig_filter, s11_target, s21_target, s22_target, couplings, matrix_order, optim_matrix),
            options={'disp': True, 'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6}
        )
        print(f"Cost after optim: {result.fun}")

        optim_matrix[i_idx, j_idx] = torch.tensor(result.x, dtype=torch.float32)
        optim_matrix[j_idx, i_idx] = torch.tensor(result.x, dtype=torch.float32)
        results.update({result.fun: optim_matrix})
        if result.fun < 0.5:
            break

    # Постобработка результата
    optim_matrix = CouplingMatrix(list(results.items())[-1][-1])
    x_current = optim_matrix.factors

    Mmin, Mmax = create_bounds(
        CouplingMatrix(CouplingMatrix.from_factors(torch.tensor(x_current, dtype=torch.float64),
                                                   pred_filter.coupling_matrix.links,
                                                   pred_filter.coupling_matrix.matrix_order)))
    bounds = [(min(m_min, m_max), max(m_min, m_max)) for m_min, m_max in tuple(zip(Mmin.factors, Mmax.factors))]
    # freq_weights = torch.ones_like(freq_weights)
    result = minimize(
        fun=cost_with_grad,
        x0=x_current,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        args=(fast_calc, orig_filter, s11_target, s21_target, s22_target, links, matrix_order),
        options={'disp': True, 'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6}
    )
    # for lambda_ in lambda_values:
    #     print(f"Current lambda={lambda_}")
    #     Mmin, Mmax = create_bounds(
    #         CouplingMatrix(CouplingMatrix.from_factors(torch.tensor(x_current, dtype=torch.float64),
    #                                                    pred_filter.coupling_matrix.links,
    #                                                    pred_filter.coupling_matrix.matrix_order)))
    #     bounds = [(min(m_min, m_max), max(m_min, m_max)) for m_min, m_max in tuple(zip(Mmin.factors, Mmax.factors))]
    #
    #     # Миксуем S-данные: от нейросети к оригинальным
    #     s11_target = DatasetMWFilter.to_db(
    #         torch.tensor(mix_s_params(s11_net0, s11_origin, lambda_), dtype=torch.complex64))
    #     s21_target = DatasetMWFilter.to_db(
    #         torch.tensor(mix_s_params(s21_net0, s21_origin, lambda_), dtype=torch.complex64))
    #     s22_target = DatasetMWFilter.to_db(
    #         torch.tensor(mix_s_params(s22_net0, s22_origin, lambda_), dtype=torch.complex64))
    #
    #     # Оптимизируем под текущее lambda
    #     result = minimize(
    #         fun=cost_with_grad,
    #         x0=x_current,
    #         method='L-BFGS-B',
    #         jac=True,
    #         bounds=bounds,
    #         args=(fast_calc, orig_filter, s11_target, s21_target, s22_target, links, matrix_order),
    #         options={'disp': True, 'maxiter': 10000, 'ftol': 1e-9, 'gtol': 1e-6}
    #     )
    #
    #     # Используем найденное решение как стартовое для следующего шага
    #     x_current = result.x

    # for delta_f in np.arange(3, 0, -0.5):
    #     print(f"Current delta={delta_f}")
    #     f_norm = np.linspace(-1-delta_f, 1+delta_f, 301)
    #     s = MWFilter.response_from_coupling_matrix(orig_filter.coupling_matrix.matrix,
    #                                                         pred_filter.f0, pred_filter.fbw,
    #                                                         pred_filter.Q,
    #                                                         MWFilter.nfreq_to_freq(f_norm, pred_filter.f0, pred_filter.bw),
    #                                                         NRNlist=[], Rs=1, Rl=1, PSs=None, device='cpu')
    #     s11_origin = s[:, 0, 0]
    #     s21_origin = s[:, 1, 0]
    #     s22_origin = s[:, 1, 1]
    #
    #     # Преобразуем в тензоры один раз
    #     s11_origin_db = DatasetMWFilter.to_db(torch.tensor(s11_origin, dtype=torch.complex64))
    #     s21_origin_db = DatasetMWFilter.to_db(torch.tensor(s21_origin, dtype=torch.complex64))
    #     s22_origin_db = DatasetMWFilter.to_db(torch.tensor(s22_origin, dtype=torch.complex64))
    #
    #     fast_calc = FastMN2toSParamCalculation(matrix_order=matrix_order,
    #                                            wlist=f_norm,
    #                                            Q=pred_filter.Q,
    #                                            fbw=pred_filter.fbw)
    #
    #     s_net0 = MWFilter.response_from_coupling_matrix(pred_filter.coupling_matrix.matrix,
    #                                                         pred_filter.f0, pred_filter.fbw,
    #                                                         pred_filter.Q,
    #                                                         MWFilter.nfreq_to_freq(f_norm, pred_filter.f0, pred_filter.bw),
    #                                                         NRNlist=[], Rs=1, Rl=1, PSs=None, device='cpu')
    #     s11_net0 = s_net0[:, 0, 0]
    #     s21_net0 = s_net0[:, 1, 0]
    #     s22_net0 = s_net0[:, 1, 1]
    #
    #     for lambda_ in lambda_values:
    #         print(f"Current lambda={lambda_}")
    #         Mmin, Mmax = create_bounds(
    #             CouplingMatrix(CouplingMatrix.from_factors(torch.tensor(x_current, dtype=torch.float64),
    #                                                        pred_filter.coupling_matrix.links,
    #                                                        pred_filter.coupling_matrix.matrix_order)))
    #         bounds = [(min(m_min, m_max), max(m_min, m_max)) for m_min, m_max in tuple(zip(Mmin.factors, Mmax.factors))]
    #
    #         # Миксуем S-данные: от нейросети к оригинальным
    #         s11_target = DatasetMWFilter.to_db(torch.tensor(mix_s_params(s11_net0, s11_origin, lambda_), dtype=torch.complex64))
    #         s21_target = DatasetMWFilter.to_db(torch.tensor(mix_s_params(s21_net0, s21_origin, lambda_), dtype=torch.complex64))
    #         s22_target = DatasetMWFilter.to_db(torch.tensor(mix_s_params(s22_net0, s22_origin, lambda_), dtype=torch.complex64))
    #
    #         # Оптимизируем под текущее lambda
    #         result = minimize(
    #             fun=cost_with_grad,
    #             x0=x_current,
    #             method='L-BFGS-B',
    #             jac=True,
    #             bounds=bounds,
    #             args=(fast_calc, orig_filter, s11_target, s21_target, s22_target, links, matrix_order),
    #             options={'disp': True, 'maxiter': 10000, 'ftol': 1e-9, 'gtol': 1e-6}
    #         )
    #
    #         # Используем найденное решение как стартовое для следующего шага
    #         x_current = result.x
    #     print(f"Cost after iteration: {result.fun}")

    print(f"Cost after homotopy L-BFGS-B: {result.fun}")
    # Mmin, Mmax = create_bounds(
    #     CouplingMatrix(CouplingMatrix.from_factors(torch.tensor(x0, dtype=torch.float64),
    #                                                pred_filter.coupling_matrix.links,
    #                                                pred_filter.coupling_matrix.matrix_order)))
    # bounds = [(min(m_min, m_max), max(m_min, m_max)) for m_min, m_max in tuple(zip(Mmin.factors, Mmax.factors))]
    #
    # result = minimize(
    #     fun=cost_with_grad,
    #     x0=x0,
    #     method='l-bfgs-b',
    #     jac=True,
    #     bounds=bounds,
    #     args=(fast_calc, orig_filter, s11_origin_db, s21_origin_db, s22_origin_db, links, matrix_order),
    #     options={'disp': True, 'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6}
    # )
    # print(f"Cost after classic L-BFGS-B: {result.fun}")

    stop_time = time.time()
    print(f"Optimize time: {stop_time - start_time:.3f} sec")

    # Постобработка результата
    optim_matrix = CouplingMatrix.from_factors(
        torch.tensor(result.x, dtype=torch.float32),
        links,
        matrix_order
    )
    results.update({result.fun: optim_matrix})
    optim_matrix = list(results.items())[-1][-1]
    w, s11_opt, s21_opt, s22_opt = fast_calc.RespM2(optim_matrix, with_s22=True)

    # plt.figure()
    # plt.plot(orig_filter.f_norm, s11_origin.real, orig_filter.f_norm, s11_pred.real, orig_filter.f_norm, s11_opt.real)
    # plt.legend(["S11 orig real", "S11 pred real", "S11 optim real"])
    # plt.figure()
    # plt.plot(orig_filter.f_norm, s11_origin.imag, orig_filter.f_norm, s11_pred.imag, orig_filter.f_norm, s11_opt.imag)
    # plt.legend(["S11 orig imag", "S11 pred imag", "S11 optim imag"])
    # plt.figure()
    # plt.plot(orig_filter.f_norm, s21_origin.real, orig_filter.f_norm, s21_pred.real, orig_filter.f_norm, s21_opt.real)
    # plt.legend(["S21 orig real", "S21 pred real", "S21 optim real"])
    # plt.figure()
    # plt.plot(orig_filter.f_norm, s21_origin.imag, orig_filter.f_norm, s21_pred.imag, orig_filter.f_norm, s21_opt.imag)
    # plt.legend(["S21 orig imag", "S21 pred imag", "S21 optim imag"])


    s11_opt_db = MWFilter.to_db(s11_opt)
    s21_opt_db = MWFilter.to_db(s21_opt)
    s22_opt_db = MWFilter.to_db(s22_opt)

    plt.figure()
    plt.title("S-параметры")
    plt.plot(w, s11_origin_db, label="S11 Origin")
    plt.plot(w, s11_opt_db, linestyle=':', label="S11 Optimized")
    plt.plot(w, s21_origin_db, label="S21 Origin")
    plt.plot(w, s21_opt_db, linestyle=':', label="S21 Optimized")
    plt.plot(w, s22_origin_db, label="S22 Origin")
    plt.plot(w, s22_opt_db, linestyle=':', label="S22 Optimized")
    plt.legend()
    plt.grid(True)

    return CouplingMatrix(optim_matrix)