#
#   100% работает с
#   python 3.11
#   numpy 2.2.4
#   pyDOE3 1.0.4
#   scikit-rf (skrf) 1.6.2
#

import random
import string

import numpy as np
import skrf as rf


def get_random_n_bytes_ascii_string(string_length: int):
    random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(string_length))

    return random_string


def get_db_array_from_complex_one(complex_array: np.array):
    db_array = 20 * np.log10(np.abs(complex_array))

    return db_array


def get_converted_params_comment_string(params_array: np.array):
    params_string = (
            '! Parameters = {dim1_delta='+str(params_array[0]) + '; dim2_delta='+str(params_array[1])
            + '; dim3_delta='+str(params_array[2]) + '; dim4_delta='+str(params_array[3])
            + '; dim5_delta='+str(params_array[4]) + '; dim6_delta='+str(params_array[5])
            + '; dim7_delta='+str(params_array[6]) + '; dim8_delta='+str(params_array[7])
            + '; dim9_delta='+str(params_array[8]) + '; dim10_delta='+str(params_array[9])
            + '; dim11_delta='+str(params_array[10]) + '; dim12_delta='+str(params_array[11])
            + '; dim13_delta=' + str(params_array[12]) + '; dim14_delta=' + str(params_array[13])
            + '; dim15_delta=' + str(params_array[14]) + '; dim16_delta=' + str(params_array[15])
            + '; dim17_delta='+str(params_array[16]) + '; dim18_delta='+str(params_array[17])
            + '; dim19_delta='+str(params_array[18]) + '; dim20_delta='+str(params_array[19]) + '}')

    return params_string


class TouchstoneGenerator:
    def __init__(self, file_path: str, params_array: np.array):
        self.file_path = file_path

        if self.file_path.endswith('.cst'):
            self.file_path = self.file_path.replace('.cst', '')

        if not self.file_path.endswith('.s4p'):
            self.file_path = self.file_path + '.s4p'

        self.params_array = params_array
        self.freqs_amount = None


    def convert_and_save_these_sparams(self, s_params_set):
        s11_vals = [x[1] for x in s_params_set[0]]
        s21_vals = [x[1] for x in s_params_set[1]]
        #   первая и вторая мода одного третьего, порта
        s31_1_vals = [x[1] for x in s_params_set[2]]
        s31_2_vals = [x[1] for x in s_params_set[3]]

        self.freqs_amount = len(s11_vals)

        snp_np = np.zeros((self.freqs_amount, 4, 4), dtype=complex)

        snp_np[:, 0, 0] = np.array(s11_vals)
        snp_np[:, 1, 0] = np.array(s21_vals)
        #   моды превращаются в 3 и 4 порт
        snp_np[:, 2, 0] = np.array(s31_1_vals)
        snp_np[:, 3, 0] = np.array(s31_2_vals)

        frequencies = np.linspace(17.3e9, 20.2e9, self.freqs_amount)

        freq_obj = rf.Frequency.from_f(frequencies, unit='Hz')

        ntw = rf.Network(frequency=freq_obj, s=snp_np, z0=0)
        ntw.write_touchstone(self.file_path)

        #   добавить с-параметры в заголовок файла
        params_as_str = get_converted_params_comment_string(self.params_array)
        self._insert_str_into_snp_file(params_as_str)


    def _insert_str_into_snp_file(self, str_to_insert: str):
        with open(self.file_path, 'r+') as f:
            lines = f.readlines()
            lines.insert(0, str_to_insert + '\n')

            f.seek(0)
            f.writelines(lines)


if __name__ == '__main__':
    pass
