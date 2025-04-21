#
#
#
from cst_funcs import CSTWrapper, CSTResultsWrapper

#   пример 1. - открытие проекта CST и вывод его параметров
#   абсолютный путь к проекту
cst_project_path = r'C:\Users\admin\PycharmProjects\NNWithCST\cst_projects\buro_zero.cst'

cst_project = CSTWrapper(cst_project_path)
cst_project.print_attributes_raw()
cst_project.close()


#   пример 2 - получение с параметров из определённого results tree
cst_file_path = r'C:\Users\admin\PycharmProjects\NNWithCST\cst_projects\buro_zero.cst'
cst_results = CSTResultsWrapper(cst_file_path)
print(cst_results.raw_complex_sparams)