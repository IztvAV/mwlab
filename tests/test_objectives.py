# tests/test_objectives.py
"""
Тесты для mwlab.opt.objectives
==============================

Покрывают работу всего слоя целей и спецификаций для MWLab-DOE/оптимизации:

1. **S-критерии:** Проверка критерия типа «|S11| ≤ –22 dB» через цепочку
   Selector → Aggregator → Comparator.
2. **Specification:** Проверка объединения критериев, расчёта penalty,
   корректности работы AND-логики.
3. **Stub-surrogate:** Используем суррогат-заглушку, чтобы контролировать
   поведение цели (pass/fail).
4. **YieldObjective:** Проверка Монте-Карло-оценки yield (P(pass)=1 и P(pass)=0).
5. **Отдельные edge-case проверки:** (см. ниже, для расширения набора).

Примеры кейсов:
---------------
- S-критерий на S-параметре (S11)
- Суммарный штраф (penalty) с учетом веса
- Сценарии, когда все точки проходят/заваливают ТЗ (yield=1.0 / yield=0.0)
- Проверка отчёта (`report`) и пустых спецификаций (ошибка)

Если добавляете новые агрегаторы, компараторы или селекторы —
добавьте отдельный тест!
"""

import numpy as np
import pytest
import skrf as rf

from mwlab.opt.objectives.selectors import SMagSelector
from mwlab.opt.objectives.aggregators import MaxAgg
from mwlab.opt.objectives.comparators import LEComparator, SoftLEComparator
from mwlab.opt.objectives.base import BaseCriterion
from mwlab.opt.objectives.specification import Specification
from mwlab.opt.objectives.yield_max import YieldObjective

from mwlab.opt.design.space import DesignSpace, ContinuousVar
from mwlab.opt.design.samplers import get_sampler
from mwlab.opt.surrogates.base import BaseSurrogate


# ────────────────────────── helpers ──────────────────────────
def make_network(s11_db: float) -> rf.Network:
    """Создаём 1-портовый скрэф-Network с заданным |S11| в дБ."""
    f = rf.Frequency.from_f([2.2e9], unit="hz")
    mag = 10 ** (s11_db / 20)         # линейная амплитуда
    s = np.array([[[mag + 0j]]])      # shape (nf, nports, nports)
    return rf.Network(frequency=f, s=s)


class SurStub(BaseSurrogate):
    """Surrogate-заглушка: возвращает заранее заданный Network."""

    def __init__(self, net: rf.Network):
        self._net = net

    def predict(self, x, *, return_std=False):
        return self._net

    def batch_predict(self, xs, *, return_std=False):
        return [self._net for _ in xs]


# ────────────────────────── tests ──────────────────────────
@pytest.fixture(scope="module")
def specification():
    crit = BaseCriterion(
        selector   = SMagSelector(1, 1, band=None, db=True),
        aggregator = MaxAgg(),
        comparator = LEComparator(-22),
        name="S11max",
        weight=2.0,                 # проверим вес
    )
    return Specification([crit])


def test_spec_pass_fail(specification):
    net_pass = make_network(-25)    # лучше порога
    net_fail = make_network(-15)    # хуже порога

    assert specification.is_ok(net_pass)
    assert not specification.is_ok(net_fail)

    # penalty с весом 2.0
    assert specification.penalty(net_pass) == 0
    assert specification.penalty(net_fail) == pytest.approx(2.0)  # 2*1


def test_yield_objective_pass():
    space   = DesignSpace({"x": ContinuousVar(-1, 1)})
    sampler = get_sampler("sobol", rng=123)
    sur     = SurStub(make_network(-30))        # всегда pass
    spec    = Specification([
        BaseCriterion(SMagSelector(1,1, db=True),
                      MaxAgg(), LEComparator(-22), name="S11")
    ])
    yobj = YieldObjective(surrogate=sur,
                          spec=spec,
                          design_space=space,
                          sampler=sampler,
                          n_mc=256)
    assert yobj() == pytest.approx(1.0, abs=1e-9)


def test_yield_objective_fail():
    space   = DesignSpace({"x": ContinuousVar(-1, 1)})
    sampler = get_sampler("sobol", rng=321)
    sur     = SurStub(make_network(-10))        # всегда fail
    spec    = Specification([
        BaseCriterion(SMagSelector(1,1, db=True),
                      MaxAgg(), LEComparator(-22), name="S11")
    ])
    yobj = YieldObjective(surrogate=sur,
                          spec=spec,
                          design_space=space,
                          sampler=sampler,
                          n_mc=256)
    assert yobj() == pytest.approx(0.0, abs=1e-9)


def test_spec_report_structure(specification):
    net = make_network(-23)
    rep = specification.report(net)
    assert "S11max" in rep
    assert "__all_ok__" in rep and "__penalty__" in rep
    assert isinstance(rep["S11max"]["value"], float)
    assert rep["S11max"]["ok"] is bool(net.s_db[0, 0, 0] <= -22)


def test_soft_le_penalty():
    c = SoftLEComparator(limit=10, margin=2, power=2)
    assert c.penalty(9) == 0.0
    assert c.penalty(12) == ((12-10)/2)**2


def test_specification_empty():
    with pytest.raises(ValueError):
        Specification([])

