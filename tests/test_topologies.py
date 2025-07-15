#tests/test_topologies.py
"""
Юнит-тесты для mwlab.filters.topologies
--------------------------------------

Запуск:
    pytest -q tests/test_topologies.py
"""

import pytest
import numpy as np
import json

from mwlab.filters.topologies import (
    Topology,
    TopologyError,
    register_topology,
    get_topology,
    list_topologies,
    folded,
    transversal,
)

# --------------------------------------------------------------------------
# 1. Registry & factory
# --------------------------------------------------------------------------
def test_registry_list_contains_defaults():
    """folded / transversal должны присутствовать в реестре сразу после импорта."""
    names = list_topologies()
    assert "folded" in names and "transversal" in names


def test_get_topology_returns_instance():
    topo = get_topology("folded", order=4)
    assert isinstance(topo, Topology)
    assert topo.order == 4
    assert topo.ports == 2


def test_register_topology_duplicate_alias():
    """Повторная регистрация alias-а должна бросать исключение."""
    with pytest.raises(KeyError):
        @register_topology("folded")        # alias уже есть
        def dummy(order, ports=2): ...      # pragma: no cover
        # регистрировать не будем – тест упадёт до сюда
        pass


# --------------------------------------------------------------------------
# 2. Topology.__post_init__  (валидность links, name, порты)
# --------------------------------------------------------------------------
def test_manual_topology_ok():
    topo = Topology(
        order=3,
        ports=2,
        links=[(1, 2), (2, 3), (1, 3), (3, 5), (5, 4)],   # с дублями и зеркалом
        name="manual"
    )
    # (3 резонатора + 2 порта) → K=5
    assert topo.size == 5
    # зеркальная пара (2,1) должна исчезнуть, (links)==8 верх. треугольник
    assert len(topo) == 5
    # name сохраняется
    assert topo.name == "manual"


def test_port_without_edge_raises():
    """Если хотя бы один порт не подключён, ожидается TopologyError."""
    with pytest.raises(TopologyError):
        Topology(order=2, ports=1, links=[(1, 2)])  # порт = 3 без связи


# --------------------------------------------------------------------------
# 3. validate_mvals()
# --------------------------------------------------------------------------
@pytest.fixture
def topo_folded4():
    return folded(order=4)        # (1-2,2-3,3-4,1-4) + порты


def test_validate_mvals_ok(topo_folded4):
    mvals = {
        "M1_2": 1, "M2_3": 1, "M3_4": 1, "M1_4": 0.1,  # резонатор-резонатор
        "M1_5": 0.8, "M4_6": 0.9  # связи с портами
    }
    # не должно бросаться
    topo_folded4.validate_mvals(mvals)


@pytest.mark.parametrize(
    "bad_keys",
    [
        {"M1_2": 1},                                 # недостаёт связей
        {"M1_2": 1, "M2_3": 1, "M3_4": 1, "M2_4": 0.2},  # лишняя M2_4
        {"M1_5": 1},                                 # индекс за диапазоном
    ],
)
def test_validate_mvals_errors(topo_folded4, bad_keys):
    with pytest.raises(TopologyError):
        topo_folded4.validate_mvals(bad_keys)


# --------------------------------------------------------------------------
# 4. Шаблоны folded / transversal
# --------------------------------------------------------------------------
def test_folded_links_correct():
    topo = folded(order=5)
    # chain (n-1) + corner + 2 port-links  = (order) +1
    assert len(topo.links) == 7        # 5-звенный ⇒ 4+1+2 = 7
    assert (1, 5) in topo.links


def test_transversal_alternating_ports():
    topo = transversal(order=4)
    p_in, p_out = 5, 6
    # (1,p_in),(2,p_out),(3,p_in),(4,p_out)
    expected = {(1, p_in), (2, p_out), (3, p_in), (4, p_out)}
    assert set(topo.links) == expected


def test_transversal_order_one_fails():
    """order=1 → порт p2 не связан, ожидаем TopologyError."""
    with pytest.raises(TopologyError):
        transversal(order=1)


def test_transversal_order_three_ok():
    topo = transversal(order=3)
    # каждый из двух портов должен появиться хотя бы один раз
    ports = [p for link in topo.links for p in link if p > topo.order]
    assert set(ports) == {topo.order + 1, topo.order + 2}

def test_transversal_wrong_ports_error():
    with pytest.raises(TopologyError):
        transversal(order=4, ports=3)     # пока поддерживаются только 2 порта

# --------------------------------------------------------------------------
# 5. Новый функционал (serialization, utils)
# --------------------------------------------------------------------------
def test_res_port_indices_and_adjacency():
    topo = folded(order=4)
    assert list(topo.res_indices)  == [1, 2, 3, 4]
    assert list(topo.port_indices) == [5, 6]

    A = topo.adjacency_matrix()
    # форма (K,K), симметричность и bool-dtype
    K = topo.size
    assert A.shape == (K, K)
    assert A.dtype == bool
    assert np.allclose(A, A.T)

def test_to_from_dict_roundtrip():
    topo = transversal(order=5, name="tr5")
    blob = topo.to_dict()
    # JSON-safe?
    json.dumps(blob)
    restored = Topology.from_dict(blob)
    assert topo == restored            # __eq__
    assert hash(topo) == hash(restored)

def test_validate_mvals_non_strict_allows_missing():
    topo = folded(order=4)
    partial = {"M1_2": 1.0}            # лишь одна связь из нужных
    # strict=False → допускаем отсутствие остальных
    topo.validate_mvals(partial, strict=False)

def test_validate_mvals_non_strict_still_blocks_extra():
    topo = folded(order=4)
    extra = {"M1_2": 1.0, "M2_4": 0.1}     # M2_4 лишняя
    with pytest.raises(TopologyError):
        topo.validate_mvals(extra, strict=False)

def test_hash_eq_ignore_order_of_links():
    t1 = Topology(order=3, ports=1, links=[(1,2), (1,4), (2,3)])
    # переставили порядок links
    t2 = Topology(order=3, ports=1, links=[(2,3), (1,2), (1,4)])
    assert t1 == t2
    assert hash(t1) == hash(t2)

