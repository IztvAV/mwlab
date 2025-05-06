import numpy as np
from edmwe_cst import CSTProject, Result1DComplex, ResultTree
import skrf as rf
from skrf import Network

from mwlab import TouchstoneData

import re

_S_PARAMS_TREE_ITEM = "1D Results\\S-Parameters"
_SP_PATTERN = "[sS][0-9]+[.,][0-9]+"



def run_solver_with_params(prj: CSTProject, params: dict[str, str | float]) -> TouchstoneData:
    """
        Заливает параметры в проект, перестраивает проект и запускает солвер.
        В случае успеха собирает S-параметры и параметры проекта в TouchstoneData.
        S-параметры извлекаются из объектов дерева "1D Results\\S-Parameters\\SX,Y"
        в 3D:RunID:0 (последние рассчитанные результаты).

        Пример использования:

        >>> from edmwe_cst import CST
        >>> from mwlab.utils import cst_utils

        >>> prj = CST.connect(2022).active3d()
        >>> ts_data = cst_utils.run_solver_with_params(prj, params={"pin_L": "20"})
        >>> ts_data.network.write_touchstone('my_network_20.s2p', form="ri")
    """

    if params:
        store_parameters(prj, params)

    if not prj.call("RebuildOnParametricChange", False, False): # добавить этот метод в edmwe_cst
        raise RuntimeError("rebuild failed")

    if not prj.call("RunSolver"): # добавить этот метод в edmwe_cst
        raise RuntimeError("solver failed")

    network = get_last_s_params_as_network(prj, consider_run_id_0=True)
    params = list_param_values(prj)

    return TouchstoneData(network=network, params=params)


def get_last_s_params_as_network(prj: CSTProject, consider_run_id_0: bool = True, run_id: str = "") -> rf.Network:
    """
        Достает S-параметры из проекта.

        Параметром run_id задаётся 3D:RunID:x для извлечения результатов.
        Если run_id не задан (пустая строка), берется либо 3D:RunID:0 (если consider_run_id_0=True),
        либо 3D:RunID:MAX (MAX это самый большой индекс run_id, доступный в проекте).

        По умолчанию используется 3D:RunID:0 (последний расчет)

        3D:RunID:0 не обязательно ссылается на последний расчет, т.к. можно запустить расчет с параметрами,
        которые до этого уже использовались. В этом случае cst просто пересчитает какой-то runID из середины списка,
        но 3D:RunID:0 будет ссылаться на него.

        Пример использования:

        >>> from edmwe_cst import CST
        >>> from mwlab.utils import cst_utils

        >>> prj = CST.connect(2022).active3d()
        >>> nw = cst_utils.get_last_s_params_as_network(prj)
        >>> nw.write_touchstone('my_network.s2p', form="ri")

    """

    rt = prj.result_tree()
    units = prj.units()

    f_unit_mult = units.get_frequency_unit_to_si()

    sp_items = [item for item in list_children(rt, _S_PARAMS_TREE_ITEM)
                if _is_s_param(item)]

    if not sp_items:
        return Network()

    if not run_id:
        ids = rt.get_result_ids_from_tree_item(sp_items[0])

        run_id = ids[0] if consider_run_id_0 and ids[0].lower() == "3d:runid:0" else ids[-1]

    results = [rt.get_result_from_tree_item(item, run_id) for item in sp_items]

    return _pack_network(f_unit_mult, sp_items, results)


def _pack_network(f_unit_mult: float, sp_items: list[str], results: list[Result1DComplex]) -> Network:
    f = _x_to_numpy(results[0]) * f_unit_mult

    sp_names = _get_item_names(sp_items)
    sp_y = [_y_to_numpy(result) for result in results]

    s = _to_numpy_data(sp_names, sp_y)

    freq = rf.Frequency.from_f(f, unit="Hz")

    return Network(frequency=freq, s = s)


def _to_numpy_data(sp_names: list[str], sp_data: list[np.ndarray]) -> np.ndarray:
    ports = _get_matrix_size(_get_item_indices(sp_names))
    size = len(sp_data[0])

    params = dict(zip(sp_names, sp_data))

    s_data = np.zeros((size, ports, ports), dtype="complex128")
    for p_out in range(ports):
        for p_in in range(ports):
            key = f"s{p_out+1},{p_in+1}"
            y = params[key]
            s_data[:, p_out, p_in] = y

    return s_data


def _get_matrix_size(sp_indices: list[(int, int)]) -> int:
    min = 0
    for ind_out, ind_in in sp_indices:
        if min < ind_out:
            min = ind_out
        if min < ind_in:
            min = ind_in
    return min


def _x_to_numpy(r1dc: Result1DComplex) -> np.ndarray:
    return _arr_to_numpy(r1dc, "x")


def _y_to_numpy(r1dc: Result1DComplex) -> np.ndarray:
    yre = _arr_to_numpy(r1dc, "yre")
    yim = _arr_to_numpy(r1dc, "yim")
    return yre + 1j*yim


def _arr_to_numpy(r1dc: Result1DComplex, arr_name: str) -> np.ndarray:
    return np.array(r1dc.get_array(arr_name))


def _get_port_indices(sp_name: str) -> (int, int):
    separator = sp_name.index(',')
    return int(sp_name[1: separator]), int(sp_name[separator+1:])


def _get_item_indices(tree_items: list[str]) -> list[(int, int)]:
    return [_get_port_indices(_get_item_name(item)) for item in tree_items]


def _get_item_names(tree_items: list[str]):
    return [_get_item_name(item) for item in tree_items]


def _get_item_name(tree_item: str):
    return tree_item.split("\\")[-1].lower()


def _is_s_param(item: str) -> bool:
    return re.fullmatch(_SP_PATTERN, _get_item_name(item))


# --------------------------------- #
# Добавить всё что ниже в edmwe_cst #
# --------------------------------- #

def store_parameters(prj: CSTProject, params: dict[str, str|float]):
    for key, value in params.items():
        prj.store_parameter(key, value)


def list_children(rt: ResultTree, item: str) -> list[str]:
    if not rt.does_tree_item_exist(item):
        return []

    children = []
    child = rt.get_first_child_name(item)
    while child:
        children.append(child)
        child = rt.get_next_item_name(child)

    return children


def list_param_values(prj: CSTProject) -> dict[str, str|float]:
    num_params = prj.get_number_of_parameters()

    params = {}

    for i in range(num_params):
        params[prj.get_parameter_name(i)] = prj.get_parameter_n_value(i)

    return params