#
#
#
from cst_funcs import CSTWrapper, CSTResultsWrapper

#   пример 1. - открытие проекта CST и вывод его параметров
#   абсолютный путь к проекту
cst_project_path = 'asdasd'

cst_project = CSTWrapper(cst_project_path)
cst_project.print_attributes_raw()
cst_project.close()


#   пример 2 - получение с-параметров из определённого results tree
cst_file_path = 'asdasd'
cst_results = CSTResultsWrapper(cst_file_path)
print(cst_results.raw_complex_sparams)