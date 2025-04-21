#
#   обёртки для функций из CST python
#   подхват либы добавлением в paths через поиск по регистру установочного пути CST
#


import os
import sys
import winreg

import numpy as np

import utils


#   подхват из регистра пути к библиотеке
def get_cst_api_path_via_registry_search():
    #   если CST установлена правильно то в регистре должно быть
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"SOFTWARE\Classes\CST.FD3Dfile.2024\DefaultIcon") as key:
        value, _ = winreg.QueryValueEx(key, "")
        exe_path = value.split(",")[0].strip('"')

        #   должно быть в C:\Soft\CST_Studio_Suite\AMD64\python_cst_libraries
        if os.path.isfile(exe_path):
            root_dir = os.path.dirname(exe_path)
            python_api_dir = os.path.join(root_dir, "python_cst_libraries")

            return python_api_dir


cst_lib_path = get_cst_api_path_via_registry_search()
sys.path.insert(0, cst_lib_path)
#   теперь можно импорт
import cst
from cst.interface import DesignEnvironment
import cst.results


class CSTWrapper:
    def __init__(self, cst_project_file_path):
        self.de = DesignEnvironment()
        self.cst_file_path = cst_project_file_path
        self.cst_project = self.de.open_project(self.cst_file_path)
        self.s_params = None


    def print_attributes_raw(self):
        #   кол-во параметров проекта
        num_params = self.cst_project.model3d.GetNumberOfParameters()
        print('всего в модели', num_params, 'параметров, их значения:')

        #   вывод параметров и их значения
        for i in range(num_params):
            param_name = self.cst_project.model3d.GetParameterName(i)
            param_value = self.cst_project.model3d.GetParameterNValue(i)
            print(param_name, ':', param_value)


    def update_buro_params_list_with_this_np_set(self, np_set: np.array):
        vba_command = '''Sub Main ()
        DeleteResults                        
        StoreDoubleParameter("dim1_delta", ''' + str(np_set[0]) + ''')
        StoreDoubleParameter("dim2_delta", ''' + str(np_set[1]) + ''')
        StoreDoubleParameter("dim3_delta", ''' + str(np_set[2]) + ''')
        StoreDoubleParameter("dim4_delta", ''' + str(np_set[3]) + ''')
        StoreDoubleParameter("dim5_delta", ''' + str(np_set[4]) + ''')
        StoreDoubleParameter("dim6_delta", ''' + str(np_set[5]) + ''')
        StoreDoubleParameter("dim7_delta", ''' + str(np_set[6]) + ''')
        StoreDoubleParameter("dim8_delta", ''' + str(np_set[7]) + ''')
        StoreDoubleParameter("dim9_delta", ''' + str(np_set[8]) + ''')
        StoreDoubleParameter("dim10_delta", ''' + str(np_set[9]) + ''')
        StoreDoubleParameter("dim11_delta", ''' + str(np_set[10]) + ''')
        StoreDoubleParameter("dim12_delta", ''' + str(np_set[11]) + ''')
        StoreDoubleParameter("dim14_delta", ''' + str(np_set[12]) + ''')
        StoreDoubleParameter("dim15_delta", ''' + str(np_set[13]) + ''')
        StoreDoubleParameter("dim16_delta", ''' + str(np_set[14]) + ''')
        StoreDoubleParameter("dim17_delta", ''' + str(np_set[15]) + ''')
        StoreDoubleParameter("dim20_delta", ''' + str(np_set[16]) + ''')
        RebuildOnParametricChange (bfullRebuild, bShowErrorMsgBox)
        End Sub'''
        self.cst_project.schematic.execute_vba_code(vba_command, timeout=None)


    def run_solver(self):
        self.cst_project.model3d.run_solver()


    def save_as(self, new_project_file_path: str):
        #   метод model3d
        #self.cst_project.model3d.Save() можно так
        #self.cst_project.save(new_project_file_path) можно так но он не перезаписывает вроде
        #   это перезапишет
        self.cst_project.model3d.SaveAs(new_project_file_path + '.cst', True)


    def close(self):
        #   это закрыло проект
        self.cst_project.close()
        #   это закрыло cst
        self.de.close()


class CSTResultsWrapper:
    def __init__(self, cst_file_path: str):
        self.cst_file_path = cst_file_path

        if not self.cst_file_path.endswith('.cst'):
            self.cst_file_path = cst_file_path + '.cst'

        self.project = cst.results.ProjectFile(self.cst_file_path)

        #   временное
        self.raw_complex_sparams = self.get_four_sets_of_s_parameters_data()
        # self.db_sets = self.get_sets_of_db()


    #   параметры именно для одной специфической модели
    def get_four_sets_of_s_parameters_data(self):
        s11_data_xy = self.project.get_3d().get_result_item(r"1D Results\S-Parameters\S1(1),1(1)")
        s21_data_xy = self.project.get_3d().get_result_item(r"1D Results\S-Parameters\S2(1),1(1)")
        s31_data_xy = self.project.get_3d().get_result_item(r"1D Results\S-Parameters\S3(1),1(1)")
        s32_data_xy = self.project.get_3d().get_result_item(r"1D Results\S-Parameters\S3(2),1(1)")

        return s11_data_xy.get_data(), s21_data_xy.get_data(), s31_data_xy.get_data(), s32_data_xy.get_data()


    def get_sets_of_db(self):
        #   s params
        s11_complex = [x[1] for x in self.raw_complex_sparams[0]]
        s21_complex = [x[1] for x in self.raw_complex_sparams[1]]
        s31_complex = [x[1] for x in self.raw_complex_sparams[2]]
        s32_complex = [x[1] for x in self.raw_complex_sparams[3]]

        s11_db = utils.get_db_array_from_complex_one(s11_complex)
        s21_db = utils.get_db_array_from_complex_one(s21_complex)
        s31_db = utils.get_db_array_from_complex_one(s31_complex)
        s32_db = utils.get_db_array_from_complex_one(s32_complex)

        return s11_db, s21_db, s31_db, s32_db


#   EXAMPLES
if __name__ == '__main__':
    #   get results (if they exist)
    file_path = r'C:\Users\admin\PycharmProjects\NNWithCST\cst_projects\buro_zero.cst'
    #   results
    cst_results = CSTResultsWrapper(file_path)
    print(cst_results.raw_complex_sparams)

    exit()
    #   print params list of a CST project
    cst_project = CSTWrapper(file_path)
    cst_project.print_attributes_raw()
    cst_project.close()
