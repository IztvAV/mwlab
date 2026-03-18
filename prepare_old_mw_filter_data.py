import argparse
import os
import skrf as rf

from filters.filter import couplilng_matrix as cm
import mwlab
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_filter_name(path_to_s_parameters: str) -> str:
    components = path_to_s_parameters.split('\\')
    filename = components[-1]
    filter_name = filename.split('.s2p')[0]
    return filter_name


def build_coupling_mask(
    n: int,
    keep_main_diag: bool = True,
    keep_adjacent_to_main: int = 1,
    keep_secondary_diag: bool = True,
    keep_adjacent_to_secondary: int = 0,
) -> np.ndarray:
    """
    Строит универсальную маску для матрицы связи размера n x n.

    Сохраняет:
    - главную диагональ (опционально),
    - несколько диагоналей рядом с главной,
    - побочную диагональ (опционально),
    - несколько диагоналей рядом с побочной (если нужно).

    Параметры
    ---------
    n : int
        Размер квадратной матрицы.
    keep_main_diag : bool
        Сохранять ли главную диагональ.
    keep_adjacent_to_main : int
        Сколько диагоналей по обе стороны от главной сохранить.
        0 -> только главная
        1 -> главная + ближайшие к ней
    keep_secondary_diag : bool
        Сохранять ли побочную диагональ.
    keep_adjacent_to_secondary : int
        Сколько диагоналей рядом с побочной сохранить.
        Для твоего случая лучше 0, чтобы смещённые побочные диагонали занулялись.

    Возвращает
    ----------
    mask : np.ndarray
        Бинарная маска shape=(n, n), dtype=bool
    """
    i, j = np.indices((n, n))

    mask = np.zeros((n, n), dtype=bool)

    # Главная диагональ и соседние к ней
    # |i - j| = 0 -> главная
    # |i - j| = 1 -> соседние к главной
    if keep_main_diag:
        mask |= (np.abs(i - j) <= keep_adjacent_to_main)
    else:
        if keep_adjacent_to_main > 0:
            mask |= (np.abs(i - j) >= 1) & (np.abs(i - j) <= keep_adjacent_to_main)

    # Побочная диагональ и соседние к ней
    # i + j = n - 1 -> побочная
    if keep_secondary_diag:
        mask |= (np.abs((i + j) - (n - 1)) <= keep_adjacent_to_secondary)

    return mask


def apply_coupling_mask(
    M: np.ndarray,
    keep_main_diag: bool = True,
    keep_adjacent_to_main: int = 1,
    keep_secondary_diag: bool = True,
    keep_adjacent_to_secondary: int = 0,
    preserve_symmetry: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Накладывает маску на матрицу связи.

    Для твоего случая рекомендуемые параметры по умолчанию:
    - keep_main_diag=True
    - keep_adjacent_to_main=1
    - keep_secondary_diag=True
    - keep_adjacent_to_secondary=0

    Это оставит:
    - главную диагональ,
    - ближайшие к ней диагонали,
    - только саму побочную диагональ,
    а смещённые побочные диагонали занулит.
    """
    M = np.asarray(M)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Ожидается квадратная матрица.")

    n = M.shape[0]
    mask = build_coupling_mask(
        n=n,
        keep_main_diag=keep_main_diag,
        keep_adjacent_to_main=keep_adjacent_to_main,
        keep_secondary_diag=keep_secondary_diag,
        keep_adjacent_to_secondary=keep_adjacent_to_secondary,
    )

    M_masked = np.where(mask, M, 0)

    if preserve_symmetry:
        M_masked = 0.5 * (M_masked + M_masked.T)

    return M_masked, mask


def save_matrix_to_s2p_single_comment(matrix_indices, matrix_values):
    """
    Сохраняет .s2p файл с матрицей, записанной в один комментарий.
    """

    # Формируем строку вида: ! matrix = [m_0_1 = 1.23, m_1_2 = 0.56]
    matrix_dict = {}
    [matrix_dict.update({f"m_{i}_{j}": val}) for (i, j), val in zip(matrix_indices, matrix_values)]
    return matrix_dict

def save_phase_to_s2p_single_comment(a11=0, a22=0, b11=0, b22=0):
    phase_dict = {}
    phase_dict.update({"a11": a11, "a22": a22, "b11": b11, "b22": b22})
    return phase_dict


FILTER_NAME = "ERV-KuCMUXT1-BPFC1"
MANIFEST_PATH = os.path.join("filters", "FilterData", FILTER_NAME)


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Генератор .s2p файлов для добавления информации о МС и параметрах фильтра. Корректирует старые данные для использования их в обучении ИИ")
    parser.add_argument("-fr", "--freq_resp", type=str, default=os.path.join(os.getcwd(), "filters", "origins", f"{FILTER_NAME}", f"{FILTER_NAME}.s2p"),
                        help="Путь к .s2p файлу с частотными характеристиками фильтра")
    parser.add_argument("-m", "--matrix", type=str, default=os.path.join(os.getcwd(), "filters", "origins", f"{FILTER_NAME}", "ERV-KuCMUXT1-BPFC1 Rev.0.3-1.0#УФМ. Матрица связи.txt"), help="Путь к файлу .txt с матрицей связи фильтра")
    parser.add_argument("-f0", "--center_freq", type=str, default="10720.8", help="Центральная частота фильтра в МГц")
    parser.add_argument("-bw", "--bandwidth", type=str, default="40", help="Ширина полосы пропускания фильтра в МГц")
    parser.add_argument("-q", "--quality_factor", type=str, default="6500", help="Значение добротности")
    parser.add_argument("-p", "--path_to_save", type=str, default=os.path.join(os.getcwd(), "filters", "origins", f"{FILTER_NAME}"), help="Путь для сохранения измененного файла")

    args = parser.parse_args()
    freq_resp_filename = get_filter_name(path_to_s_parameters=args.freq_resp)
    matrix = cm.CouplingMatrix.from_file(args.matrix)
    mod_matrix, _ = apply_coupling_mask(matrix.matrix)
    matrix = cm.CouplingMatrix(torch.tensor(mod_matrix))
    net = rf.Network(args.freq_resp)

    path_to_save = os.path.join(args.path_to_save, freq_resp_filename+"_modify.s2p")
    new_td = mwlab.TouchstoneData(
        network=rf.Network(frequency=net.frequency, s=net.s, z0=net.z0),
        params={"f0": args.center_freq, "bw": args.bandwidth, "Q": args.quality_factor, "N": matrix.matrix_order-2}
    )
    matrix_dict = save_matrix_to_s2p_single_comment(matrix_indices=matrix.links, matrix_values=matrix.factors)
    new_td.params.update(matrix_dict)
    phase_dict = save_phase_to_s2p_single_comment()
    new_td.params.update(phase_dict)
    print(f"Path to save file: {path_to_save}")
    new_td.network.write_touchstone(filename=path_to_save)
    new_td.save(path_to_save)
    path_to_save = os.path.join(MANIFEST_PATH, "origins_data", f"{FILTER_NAME}_modify.s2p")
    print(f"Path to save file: {path_to_save}")
    new_td.network.write_touchstone(path_to_save)
    new_td.save(path_to_save)
    new_td.network.plot_s_db(m=0, n=0, label='S11 origin')
    new_td.network.plot_s_db(m=1, n=0, label='S21 origin')
    new_td.network.plot_s_db(m=1, n=1, label='S22 origin')
    plt.show()


if __name__ == "__main__":
    main()
