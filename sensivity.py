from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
import torch

from filters import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers, SamplerTypes, MWFilter, CouplingMatrix
from filters.datasets.theoretical_dataset_generator import PSShift, CMShifts
from filters.mwfilter_optim.base import FastMN2toSParamCalculation


def transfer_function(fast_calc: FastMN2toSParamCalculation, orig_filter: MWFilter, x):
    M = CouplingMatrix.from_factors(torch.tensor(x, dtype=torch.float32), orig_filter.coupling_matrix.links, orig_filter.coupling_matrix.matrix_order)
    w, S11, S21, S22 = fast_calc.RespM2(M, with_s22=True)
    area_S11 = np.trapezoid(y=MWFilter.to_db(S11), x=orig_filter.f_norm)  # интеграл по каждому прогону
    area_S21 = np.trapezoid(y=MWFilter.to_db(S21), x=orig_filter.f_norm)  # интеграл по каждому прогону
    area_S22 = np.trapezoid(y=MWFilter.to_db(S22), x=orig_filter.f_norm)  # интеграл по каждому прогону
    return area_S11, area_S21, area_S22


def nonlinear_function(fast_calc: FastMN2toSParamCalculation, orig_filter: MWFilter, x):
    M = CouplingMatrix.from_factors(torch.tensor(x, dtype=torch.float32), orig_filter.coupling_matrix.links, orig_filter.coupling_matrix.matrix_order)
    w, S11, S21, S22 = fast_calc.RespM2(M, with_s22=True)

    min_S11 = np.min(np.array(MWFilter.to_db(S11)))  # интеграл по каждому прогону
    max_S21 = np.max(np.array(MWFilter.to_db(S21)))  # интеграл по каждому прогону
    min_S22 = np.min(np.array(MWFilter.to_db(S22)))  # интеграл по каждому прогону
    return min_S11, max_S21, min_S22


def run(orig_filter: MWFilter):
    # 1. Определите пространство параметров
    cm = orig_filter.coupling_matrix
    d = len(cm.factors)
    cm_shifts_delta = CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=5e-3, parasitic_coupling=5e-3)
    m_min, m_max = CMTheoreticalDatasetGeneratorSamplers.create_min_max_matrices(
        origin_matrix=cm,
        deltas=cm_shifts_delta
    )
    m_min = CouplingMatrix(m_min)
    m_max = CouplingMatrix(m_max)
    bounds = [ [min(m), max(m)] for m in zip(m_min.factors, m_max.factors)]

    problem = {
        'num_vars': d,  # число переменных, например, элементов матрицы связи
        'names': [f'm_{i}_{j}' for i, j in cm.links],
        'bounds': bounds
    }
    # 2. Сэмплирование
    param_values = saltelli.sample(problem, N=128)

    fast_calc = FastMN2toSParamCalculation(matrix_order=cm.matrix_order,
                                           wlist=orig_filter.f_norm,
                                           Q=orig_filter.Q,
                                           fbw=orig_filter.fbw)
    # 3. Вычисление выходов (АЧХ на фиксированных частотах)
    Y = np.array([nonlinear_function(fast_calc, orig_filter, x) for x in param_values])

    # Предварительно инициализируем накопители для суммирования
    total_S1 = np.zeros(d)
    total_S2 = np.zeros(d)

    for y, title in zip(Y.T, ["S11", "S21", "S22"]):
        print(f"Start Sobol analyze for {title}")
        Si = sobol.analyze(problem, y, calc_second_order=True)

        S1 = np.clip(Si['S1'], 0, 1)
        S2 = Si['S2']

        # Считаем вклады второго порядка
        S2_contributions = np.zeros(d)
        for i in range(d):
            for j in range(i + 1, d):
                contrib = S2[i, j]
                S2_contributions[i] += contrib
                S2_contributions[j] += contrib

        S2_contributions = np.clip(S2_contributions, 0, 1)

        # Накапливаем вклад
        total_S1 += S1
        total_S2 += S2_contributions

        # Отдельный график для текущего параметра
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(problem['names'], S1, label='First-order (S1)')
        ax.bar(problem['names'], S2_contributions, bottom=S1, label='Second-order interaction')
        ax.set_ylabel('Sobol Index')
        ax.set_title(f'Соболь-анализ: вклад параметров для {title}')
        ax.legend()
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

    # ===== Суммарный график =====

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(problem['names'], total_S1, label='Суммарный S1')
    ax.bar(problem['names'], total_S2, bottom=total_S1, label='Суммарный S2')
    ax.set_ylabel('Суммарный индекс Соболя (по S11+S21+S22)')
    ax.set_title('Соболь-анализ: общий вклад параметров (S1 + S2)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    importance = total_S1 + total_S2
    importance /= importance.sum()  # нормализация
    return importance
