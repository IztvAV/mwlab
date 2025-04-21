#
#   100% работает с
#   python 3.11
#   numpy 2.2.4
#   pyDOE3 1.0.4
#   scikit-rf (skrf) 1.6.2
#

import datetime
import os

from pyDOE3 import lhs

#   местные библиотеки
from cst_funcs import CSTWrapper, CSTResultsWrapper
from utils import TouchstoneGenerator, get_random_n_bytes_ascii_string

#   настройки
PATH_ROOT = os.path.abspath(os.path.dirname(__file__))
print('рабочий путь', [PATH_ROOT])


class DatasetsGenerator:
    def __init__(self, cst_project_path):
        self.cst_project_path = cst_project_path

    def generate_this_many_datasets(self, iters: int):
        #   кол-во итераций - (n генераций датасетов выходных данных)
        iters_number = iters

        #   набор входных данных, используя latin-hypercube
        #   A элементов, Б наборов всего, 17 входов у нас на модели
        size_errors_np_arrays = lhs(17, iters_number)
        # print(f"size_errors_np_arrays shape: {size_errors_np_arrays.shape}, type: {type(size_errors_np_arrays)}")
        #   перевод в область от -100 до 100 микрон, 0 = -100, 1 = 100
        #   микрон это 10 ^ -6, у нас ** 3 потомучто единица измерений в модели = мм
        scale = 10 ** 3
        size_errors_np_arrays = (size_errors_np_arrays * 200 - 100) / scale

        for i in range(iters_number):
            start_datetime = datetime.datetime.now()
            print('итерация:', i + 1, '@', start_datetime, end='...')
            #   модель
            cst_project = CSTWrapper(self.cst_project_path)

            #   изменение параметров в модели, расчёт и закрытие cst
            size_errors_set = size_errors_np_arrays[i]
            # print(errors_set.shape, errors_set)

            cst_project.update_buro_params_list_with_this_np_set(size_errors_set)
            cst_project.run_solver()
            cst_project.close()

            #   получить вывод
            s_params_set_complex = CSTResultsWrapper(self.cst_project_path).raw_complex_sparams

            #   и сгенерировать тачстоун
            if not os.path.exists('sparams'):
                os.makedirs('sparams')
                        
            ts_file_path = 'sparams\\sparams_'  + get_random_n_bytes_ascii_string(8)
            TouchstoneGenerator(ts_file_path, size_errors_set).convert_and_save_these_sparams(s_params_set_complex)

            print(' заняло', datetime.datetime.now() - start_datetime)


if __name__ == '__main__':
    print('start @', datetime.datetime.now())
    main_start_time = datetime.datetime.now()

    cst_project_path = r'C:\Users\admin\PycharmProjects\NNWithCST\cst_projects\buro.cst'
    DatasetsGenerator(cst_project_path).generate_this_many_datasets(2)

    print('генерация датасета закончена, это заняло', datetime.datetime.now() - main_start_time)
