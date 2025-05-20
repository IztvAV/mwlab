#mwlab/opt/loops/tolerance.py
"""
mwlab.opt.loops.tolerance
=========================
**ToleranceLoop** – управляющий цикл расчета технологических допусков.

Алгоритм (MVP)
--------------
1. Пользователь задаёт:
   * **EM-backend** – функция-обертка для вызова CST/HFSS/…
     (любой САПР, главное – метод `simulate(params) -> rf.Network`);
   * **DesignSpace** – переменные, их диапазоны;
   * **Specification** – набор S-критериев pass/fail;
   * **готовый surrogate** (*NNSurrogate*) – эмуль прямой задачи;
   * опции `BoxFinder` (`strategy`, `mode`, `target_yield`, …).

2. `ToleranceLoop.run(center)`
   * запускает `BoxFinder` внутри начального куба `center±δ₀`;
   * возвращает словарь допусков `{param: (-δ_left, +δ_right)}`
     **(для `mode='sym'` δ_left = δ_right).**

> **TODO на будущее:**
> – активное пополнение датасета точками CST/HFSS на границе бокса;
> – дообучение surrogate;
> – многократные итерации до сходимости.

Пример
------
>>> loop = ToleranceLoop(
...         em_backend   = cst_wrap,            # класс-прокси
...         design_space = space,
...         specification= spec,
...         surrogate    = nn_surr,
...         delta_init   = 1e-4,
...         box_args     = dict(strategy="lhs_global",
...                             mode="asym",
...                             target_yield=0.9,
...                             n_lhs=4096)
... )
>>> cube = loop.run(center_params)
>>> print(cube["w"])      # (-8.2e-5, 7.8e-5)
"""
from __future__ import annotations

from typing import Protocol, Dict, Tuple, Mapping, Any, Callable

import skrf as rf

from ..design.space import DesignSpace
from ..surrogates import BaseSurrogate
from ..objectives.specification import Specification
from ..analysis.box_finder import BoxFinder


# ────────────────────────────────────────────────────────────────────────────
#                   Легкий протокол EM-backend (CST / HFSS / …)
# ────────────────────────────────────────────────────────────────────────────
class EMBackend(Protocol):
    """
    Мини-интерфейс к любому полноволновому решателю.
    Пользователь должен реализовать *один* метод `simulate`.
    """
    def simulate(self, params: Mapping[str, float]) -> rf.Network: ...


# ────────────────────────────────────────────────────────────────────────────
class ToleranceLoop:
    """
    Parameters
    ----------
    em_backend : EMBackend
        Объект-обертка над CST/HFSS/…, обладающий методом
        `simulate(params) -> rf.Network`.
    design_space : DesignSpace
        Описание проектных переменных.
    specification : Specification
        Техзадание (список Criterion-ов).
    surrogate : BaseSurrogate
        Уже обученная surrogate-модель прямой задачи X→S.
        *(На следующих версиях будет дообучаться внутри цикла.)*
    delta_init : float
        Начальный симметричный δ для первой исследуемой зоны.
    box_args : dict, optional
        Параметры, прокидываемые в `BoxFinder(**box_args)`.
    """

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        *,
        em_backend: EMBackend,
        design_space: DesignSpace,
        specification: Specification,
        surrogate: BaseSurrogate,
        delta_init: float = 1e-4,
        box_args: Dict[str, Any] | None = None,
    ):
        self.backend = em_backend
        self.space = design_space
        self.spec = specification
        self.surrogate = surrogate
        self.delta_init = float(delta_init)
        self.box_args = box_args or {}

    # ======================================================================
    #                                RUN
    # ======================================================================
    def run(self, center: Mapping[str, float]) -> Dict[str, Tuple[float, float]]:
        """
        Запускает **одну** итерацию tolerance-анализа и
        возвращает гипер-куб допусков.

        Notes
        -----
        • В текущем MVP surrogate *не* переобучается.
        • Backend вызывается только при дальнейшем расширении
          (см. TODO-блоки).
        """
        # 1) запускаем BoxFinder
        box_finder = BoxFinder(**self.box_args)
        cube = box_finder.find(center,
                               self.space,
                               self.surrogate,
                               self.spec,
                               delta_init=self.delta_init)

        # TODO: 2) валидация точек на границе куба через em_backend.simulate
        # TODO: 3) active sampling + дообучение surrogate + повтор цикла

        return cube

    # ---------------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return (f"ToleranceLoop(box={self.box_args}, "
                f"δ0={self.delta_init})")
