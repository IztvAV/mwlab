# mwlab/filters/cm_gen.py
# -*- coding: utf-8 -*-
"""
cm_gen.py — генерация батчей параметров и S‑параметров
=====================================================

Модуль связывает подсистему матриц связи **MWLab.filters** с инфраструктурой
проектных пространств **MWLab.opt.design** и предоставляет готовые инструменты
для обучения нейронных сетей и проведения численных экспериментов.

Содержимое
----------

* **schema_to_space(schema, …)**
    Автоматически формирует `DesignSpace`, в котором каждая переменная
    соответствует одному элементу вектора параметров, описанного
    `ParamSchema`.

* **CMDataset**
    Класс `torch.utils.data.Dataset`, который
      1. использует любой сэмплер `mwlab.opt.design.samplers`
         (Sobol, Halton, LHS, …)
      2. пакует точки в векторы параметров (`ParamSchema.pack`)
      3. по желанию рассчитывает S‑параметры на фиксированной сетке Ω
         через ядро `cm_core.solve_sparams`.

Пример
-----------------
```python
from mwlab.filters.topologies import get_topology
from mwlab.filters.cm_schema import ParamSchema
from mwlab.filters.cm_gen import CMDataset
import numpy as np

topo   = get_topology("folded", order=5)
schema = ParamSchema.from_topology(topo, include_qu="vec", include_phase=("a",))
omega  = np.linspace(-3, 3, 801)

ds = CMDataset(schema, 50_000, omega=omega, return_s=True)
vec, S = ds[0]             # param‑vector и S‑матрица
```

Таким образом модуль позволяет получать как «сырые» батчи параметров,
так и готовые обучающие пары «параметры → S(Ω)» без лишнего кода.
"""

from __future__ import annotations

from typing import Callable, List, Mapping, Sequence, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from mwlab.opt.design.samplers import BaseSampler, get_sampler
from mwlab.opt.design.space import (
    DesignSpace,
    ContinuousVar,
)

from .cm_schema import ParamSchema
from .cm_core import (
    solve_sparams,
    CoreSpec,
    DT_R,
    DEFAULT_DEVICE,
)

__all__ = ["schema_to_space", "CMDataset"]


# ────────────────────────────────────────────────────────────────────────────
#        1. schema_to_space, space_to_schema – мост ParamSchema <-> DesignSpace
# ────────────────────────────────────────────────────────────────────────────
def schema_to_space(
    schema: ParamSchema,
    *,
    m_range: tuple[float, float] = (-1.0, 1.0),
    qu_range: tuple[float, float] = (300.0, 1200.0),
    phase_range: tuple[float, float] = (-0.25, 0.25),
) -> DesignSpace:
    """
    Создаёт *DesignSpace*, в котором каждая переменная соответствует ОДНОМУ
    скалярному элементу вектора параметров.

    Диапазоны выбираются по блокам:

    * ``M``       → ``m_range``
    * ``qu``      → ``qu_range``  (игнорируется, если вектор не содержит qu)
    * ``phase_a``/``phase_b`` → ``phase_range``  (если соответствующий блок есть)
    """
    vars: Dict[str, ContinuousVar] = {}

    # --- M блок ---
    for k in schema.keys[schema.slices["M"]]:
        vars[k] = ContinuousVar(lower=m_range[0], upper=m_range[1])

    # --- qu блок ---
    if schema.include_qu != "none":
        if schema.include_qu == "scalar":
            vars["qu"] = ContinuousVar(lower=qu_range[0], upper=qu_range[1])
        else:
            for i in range(1, schema.topo.order + 1):
                vars[f"qu_{i}"] = ContinuousVar(lower=qu_range[0], upper=qu_range[1])

    # --- phase ---
    if "a" in schema.include_phase:
        for i in range(1, schema.topo.ports + 1):
            vars[f"phase_a{i}"] = ContinuousVar(lower=phase_range[0], upper=phase_range[1])

    if "b" in schema.include_phase:
        for i in range(1, schema.topo.ports + 1):
            vars[f"phase_b{i}"] = ContinuousVar(lower=phase_range[0], upper=phase_range[1])

    return DesignSpace(vars)


def space_to_schema(
    space,
    topo,
    *,
    include_qu: str | None = None,
    include_phase: Sequence[str] | None = None,
) -> "ParamSchema":
    """
    Построить **ParamSchema** из уже существующего `DesignSpace`.

    Функция удобна, когда пользователь сначала описал пространство DOE
    (например — YAML‑файлом под оптимизатор), а затем хочет получить
    схему параметров для паковки векторов и расчёта S‑параметров.

    Параметры
    ---------
    space : DesignSpace
        Пространство переменных.   Имена переменных должны
        *точно* совпадать с ключами, используемыми в ParamSchema:
        ``"M1_1", "M1_2", …, "qu" / "qu_3", "phase_a1" …``.
    topo : Topology
        Топология фильтра, необходима для построения Schema.
    include_qu : {"none","scalar","vec"} | None
        * None  → попытаться определить автоматически:
            * `"qu"`      → `"scalar"`
            * `"qu_1"`    → `"vec"`
            * иначе       → `"none"`
        * Строка – жёстко задать режим блока qu.
    include_phase : (),("a",),("a","b") | None
        * None → авто‑детект:
            присутствие имён, начинающихся на `"phase_a"`/`"phase_b"`.
        * Tuple/Sequence – задать явно.

    Возвращает
    ----------
    ParamSchema
        Схема в точности соответствующая переменным пространства.

    Исключения
    ----------
    ValueError
        • если не удаётся сделать авто‑детект,
        • если в пространстве есть «лишние» переменные,
        • если каких‑то нужных переменных не хватает.
    """
    from mwlab.filters.cm_schema import ParamSchema  # локальный импорт во избежание циклов

    var_names = set(space)          # множество имён переменных DesignSpace

    # ----------- блок include_qu -----------------------------------------
    if include_qu is None:          # авто‑детект
        if "qu" in var_names:
            include_qu = "scalar"
        elif any(name.startswith("qu_") for name in var_names):
            include_qu = "vec"
        else:
            include_qu = "none"

    # ----------- блок include_phase --------------------------------------
    if include_phase is None:       # авто‑детект
        blocks: list[str] = []
        if any(n.startswith("phase_a") for n in var_names):
            blocks.append("a")
        if any(n.startswith("phase_b") for n in var_names):
            blocks.append("b")
        include_phase = tuple(blocks)

    # ----------- строим Schema -------------------------------------------
    schema = ParamSchema.from_topology(
        topo,
        include_qu=include_qu,           # type: ignore[arg-type]
        include_phase=include_phase,
    )

    # ----------- валидация -----------------------------------------------
    # 1) Все ключи, ожидаемые схемой, должны присутствовать в space
    missing = [k for k in schema.keys if k not in var_names]
    # 2) В space не должно быть «лишних» переменных
    extra   = [n for n in var_names if n not in schema.key_set]

    if missing or extra:
        raise ValueError(
            "space_to_schema: несовпадение переменных.\n"
            f"  Не хватает: {missing}\n"
            f"  Лишние   : {extra}"
        )

    return schema

# ────────────────────────────────────────────────────────────────────────────
#                                 CMDataset
# ────────────────────────────────────────────────────────────────────────────
class CMDataset(Dataset):
    """
    Dataset, работающий поверх DesignSpace‑сэмплеров.

    Форматы выдачи
    --------------
    * ``return_s=False``  →  ``Tensor param_vec``           shape (L,)
    * ``return_s=True``   →  ``(param_vec, S)``             shape (F,P,P)

    Parameters
    ----------
    schema : ParamSchema
        Схема параметров (определяет порядок и размерность L).
    N : int
        Количество выборок в датасете.
    sampler : str | BaseSampler | Callable[[DesignSpace,int],list[dict]] | None
        Источник точек дизайна.  Если None — автоматически:
        ``schema_to_space → SobolSampler``.
    omega : Sequence[float] | torch.Tensor | None
        Сетка Ω (нормированная). Обязательна, если `return_s=True`.
    return_s : bool
        Генерировать ли сразу S‑матрицу.
    device : torch.device | str
        Куда создавать тензоры.
    method / fix_sign
        Параметры ядра `solve_sparams`.
    """

    def __init__(
        self,
        schema: ParamSchema,
        N: int,
        *,
        sampler: str | BaseSampler | Callable[[DesignSpace, int], List[Mapping[str, float]]] | None = None,
        omega: Sequence[float] | torch.Tensor | None = None,
        return_s: bool = False,
        method: str = "auto",
        fix_sign: bool = False,
        device: str | torch.device = DEFAULT_DEVICE,
        # диапазоны по умолчанию (если auto‑space)
        m_range: tuple[float, float] = (-1.0, 1.0),
        qu_range: tuple[float, float] = (300.0, 1200.0),
        phase_range: tuple[float, float] = (-0.25, 0.25),
    ):
        self.schema = schema
        self.N = int(N)
        self.device = torch.device(device)
        self.return_s = bool(return_s)

        # 1) DesignSpace & sampler
        if sampler is None or isinstance(sampler, str):
            # авто‑создание пространства и Sobol‑сэмплера
            self.space = schema_to_space(schema, m_range=m_range, qu_range=qu_range, phase_range=phase_range)
            self.sampler = get_sampler(sampler or "sobol", rng=0)  # type: ignore[arg-type]
        elif isinstance(sampler, BaseSampler):
            self.sampler = sampler
            # пользователь сам знает своё пространство → берём из самого сэмплера вызовом позже
            self.space = None
        else:
            # произвольный call‑back sampler(space, n)
            self.sampler = sampler
            self.space = schema_to_space(schema, m_range=m_range, qu_range=qu_range, phase_range=phase_range)

        # 2) Omega / CoreSpec
        if self.return_s:
            if omega is None:
                raise ValueError("omega must be provided when return_s=True")
            self.omega = torch.as_tensor(omega, dtype=DT_R, device=self.device)
            self.spec = CoreSpec(
                order=schema.topo.order,
                ports=schema.topo.ports,
                method=method,
                fix_sign=fix_sign,
            )

    # ------------------------------------------------ Dataset interface
    def __len__(self):  # noqa: D401
        return self.N

    def _sample_point(self) -> Mapping[str, float]:
        """Запрашивает одну точку у выбранного сэмплера."""
        if isinstance(self.sampler, BaseSampler):
            # s = None означает, что сэмплер сам знает / принимает space
            pts = self.sampler.sample(self.space, 1) if self.space is not None else self.sampler.sample(None, 1)
        else:  # произвольный коллбэк‑функционал
            pts = self.sampler(self.space, 1)
        return pts[0]

    def __getitem__(self, idx: int):  # noqa: D401
        # 1) точка дизайна → dict
        d = self._sample_point()

        # 2) pack → vector
        vec = self.schema.pack(d, strict=False, device=self.device)

        if not self.return_s:
            return vec

        # 3) S‑матрица
        M_real, qu, phase_a, phase_b = self.schema.assemble(vec, device=self.device)
        S = solve_sparams(
            self.spec,
            M_real,
            self.omega,
            qu=qu,
            phase_a=phase_a,
            phase_b=phase_b,
            device=self.device,
        )
        return vec, S

    # ------------------------------------------------ helpers
    def extra_repr(self) -> str:  # noqa: D401
        return f"N={self.N}, return_s={self.return_s}, device={self.device}"
