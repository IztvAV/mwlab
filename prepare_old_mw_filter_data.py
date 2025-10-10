import argparse
import os
import skrf as rf
from filters.filter import couplilng_matrix as cm
import copy
from filters.filter.mwfilter import MWFilter
import mwlab


def get_filter_name(path_to_s_parameters: str) -> str:
    components = path_to_s_parameters.split('\\')
    filename = components[-1]
    filter_name = filename.split('.s2p')[0]
    return filter_name


def save_matrix_to_s2p_single_comment(matrix_indices, matrix_values):
    """
    Сохраняет .s2p файл с матрицей, записанной в один комментарий.
    """

    # Формируем строку вида: ! matrix = [m_0_1 = 1.23, m_1_2 = 0.56]
    matrix_dict = {}
    [matrix_dict.update({f"m_{i}_{j}": val}) for (i, j), val in zip(matrix_indices, matrix_values)]
    return matrix_dict

def save_phase_to_s2p_single_comment(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017):
    phase_dict = {}
    phase_dict.update({"phi11": phi11, "phi21": phi21, "theta11": theta11, "theta21": theta21})
    return phase_dict


FILTER_NAME = "EAMU4-KuIMUXT2-BPFC2"


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Генератор .s2p файлов для добавления информации о МС и параметрах фильтра. Корректирует старые данные для использования их в обучении ИИ")
    parser.add_argument("-fr", "--freq_resp", type=str, default=os.path.join(os.getcwd(), "filters", "origins", f"{FILTER_NAME}", f"{FILTER_NAME}.s2p"),
                        help="Путь к .s2p файлу с частотными характеристиками фильтра")
    parser.add_argument("-m", "--matrix", type=str, default=os.path.join(os.getcwd(), "filters", "origins", f"{FILTER_NAME}", "cst_matrix_with_parasitic_couplings.txt"), help="Путь к файлу .txt с матрицей связи фильтра")
    parser.add_argument("-f0", "--center_freq", type=str, default="11543.4", help="Центральная частота фильтра в МГц")
    parser.add_argument("-bw", "--bandwidth", type=str, default="65", help="Ширина полосы пропускания фильтра в МГц")
    parser.add_argument("-q", "--quality_factor", type=str, default="6000", help="Значение добротности")
    parser.add_argument("-p", "--path_to_save", type=str, default=os.path.join(os.getcwd(), "filters", "origins", f"{FILTER_NAME}"), help="Путь для сохранения измененного файла")

    args = parser.parse_args()
    filter_name = get_filter_name(path_to_s_parameters=args.freq_resp)
    matrix = cm.CouplingMatrix.from_file(args.matrix)
    net = rf.Network(args.freq_resp)

    path_to_save = os.path.join(args.path_to_save, filter_name+"_modify.s2p")
    new_td = mwlab.TouchstoneData(
        network=rf.Network(frequency=net.frequency, s=net.s, z0=net.z0),
        params={"f0": args.center_freq, "bw": args.bandwidth, "Q": args.quality_factor, "N": matrix.matrix_order-2}
    )
    matrix_dict = save_matrix_to_s2p_single_comment(matrix_indices=matrix.links, matrix_values=matrix.factors)
    new_td.params.update(matrix_dict)
    print(f"Path to save file: {path_to_save}")
    new_td.network.write_touchstone(filename=path_to_save)
    new_td.save(path_to_save)


if __name__ == "__main__":
    main()
