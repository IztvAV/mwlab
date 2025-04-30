import argparse
import os
import skrf as rf
import couplilng_matrix as cm
import copy
from mwfilter import MWFilter


def get_filter_name(path_to_s_parameters: str) -> str:
    components = path_to_s_parameters.split('\\')
    filename = components[-1]
    filter_name = filename.split('.s2p')[0]
    return filter_name


def save_matrix_to_s2p_single_comment(matrix_indices, matrix_values):
    """
    Сохраняет .s2p файл с матрицей, записанной в один комментарий.
    """
    import numpy as np

    # Формируем строку вида: ! matrix = [m_0_1 = 1.23, m_1_2 = 0.56]
    elements = [f"m_{i}_{j} = {val:.6f}" for (i, j), val in zip(matrix_indices, matrix_values)]
    matrix_comment = " matrix: [" + ", ".join(elements) + "]\n"
    return matrix_comment


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Генератор .s2p файлов для добавления информации о МС и параметрах фильтра. Корректирует старые данные для использования их в обучении ИИ")
    parser.add_argument("-fr", "--freq_resp", type=str, default=os.getcwd()+"\\SCYA501-KuIMUXT5-BPFC3/SCYA501-KuIMUXT5-BPFC3 Rev.0.3-0.0#УФМ.12.44-12.8 ГГц. RI.s2p",
                        help="Путь к .s2p файлу с частотными характеристиками фильтра")
    parser.add_argument("-m", "--matrix", type=str, default=os.getcwd()+"\\SCYA501-KuIMUXT5-BPFC3/SCYA501-KuIMUXT5-BPFC3 Rev.0.3-0.0#Матрица связи.txt", help="Путь к файлу .txt с матрицей связи фильтра")
    parser.add_argument("-f0", "--center_freq", type=str, default="12000", help="Центральная частота фильтра в МГц")
    parser.add_argument("-bw", "--bandwidth", type=str, default="36", help="Ширина полосы пропускания фильтра в МГц")
    parser.add_argument("-q", "--quality_factor", type=str, default="6100", help="Значение добротности")
    parser.add_argument("-p", "--path_to_save", type=str, default=os.getcwd(), help="Путь для сохранения измененного файла")

    args = parser.parse_args()
    filter_name = get_filter_name(path_to_s_parameters=args.freq_resp)
    matrix = cm.CouplingMatrix.from_file(args.matrix)
    matrix_comment = save_matrix_to_s2p_single_comment(matrix_indices=matrix.links, matrix_values=matrix.factors)
    net = copy.deepcopy(rf.Network(args.freq_resp))
    net.comments += " f0: " + args.center_freq + " MHz\n bw: " + args.bandwidth + " MHz\n Q: " + args.quality_factor + "\n N: " + str(matrix.matrix_order-2) + "\n" + matrix_comment
    path_to_save = args.path_to_save + "\\" + filter_name+"_modify.s2p"
    print(f"Path to save file: {path_to_save}")
    net.write_touchstone(filename=path_to_save)
    mwfilter = MWFilter(path_to_save)
    pass


if __name__ == "__main__":
    main()
