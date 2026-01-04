# mwlab/filters/cm_nn.py
# -*- coding: utf-8 -*-
"""
cm_nn.py — слои PyTorch для аналитического расчёта S‑параметров
===============================================================

Зачем
-----
Инкапсулировать вызов ядра `solve_sparams` в класс `torch.nn.Module`,
чтобы его можно было вставлять в большие модели: оптимизаторы матриц
связи, автоэнкодеры, GAN‑ы и т.д.

Ключевой объект
---------------
**CMLayer** (Coupling‑Matrix Layer)

* принимает **батч** параметр‑векторов, сформированных согласно `ParamSchema`;
* собирает `(M_real, qu, phase_a, phase_b)` с помощью `schema.assemble`;
* вычисляет `S(ω)` через `cm_core.solve_sparams`, сохраняя автоград;
* все операции выполняются на том же `device`, что и входной батч.

Особенности реализации
----------------------
* В конструкторе кэшируются `rows/cols` (индексы верхнего треугольника) в
  виде Buffer‑ов, чтобы избежать лишнего выделения памяти.
* `omega` хранится как Buffer (частотная сетка фиксирована для слоя).
* `CoreSpec` формируется однажды; метод/фиксация знака задаются в `__init__`.
* Дополнительные опции — `return_intermediate` (будущая доработка)
  можно легко добавить.

```python
import torch
from mwlab.filters.topologies import get_topology
from mwlab.filters.cm_schema import ParamSchema
from mwlab.filters.cm_nn import CMLayer

# 1) топология и схема
topo = get_topology("folded", order=6)
schema = ParamSchema.from_topology(topo, include_qu="scalar", include_phase=("a", "b"))

# 2) слой
omega = torch.linspace(-3, 3, 801)
layer = CMLayer(schema, omega)

# 3) случайный батч параметров
batch_vec = torch.randn(32, schema.size, requires_grad=True)

# 4) прямой проход
S = layer(batch_vec)          # (32, 801, 2, 2)
loss = S.abs().mean()
loss.backward()               # градиент по batch_vec

```

"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from mwlab.filters.cm_schema import ParamSchema
from mwlab.filters.cm_core import (
    solve_sparams,
    CoreSpec,
    DEFAULT_DEVICE,
    DT_R,
)

__all__ = ["CMLayer"]


class CMLayer(nn.Module):
    """
    Слой «Расчёт S‑параметров» для батча параметр‑векторов.

    Parameters
    ----------
    schema : ParamSchema
        Схема параметров (задаёт порядок и срезы).
    omega : Sequence[float] | torch.Tensor
        Частотная сетка Ω (нормированная), shape (F,).
    method : {"auto","solve","inv"}
        Алгоритм обращения матрицы (как в `solve_sparams`).
    fix_sign : bool
        Инвертировать ли знак S12/S21 (IEEE).
    device : str | torch.device | None
        Устройство, на котором будут храниться буферы (`omega`, `rows`, `cols`).
        Если None — определяется как CPU, а при первом вызове переносится
        на устройство входного тензора.
    """

    def __init__(
        self,
        schema: ParamSchema,
        omega: Sequence[float] | torch.Tensor,
        *,
        method: str = "auto",
        fix_sign: bool = False,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()

        self.schema = schema
        self.spec = CoreSpec(
            order=schema.topo.order,
            ports=schema.topo.ports,
            method=method,
            fix_sign=fix_sign,
        )

        dev = torch.device(device) if device is not None else torch.device("cpu")

        # Частотная сетка (F,)
        self.register_buffer(
            "omega",
            torch.as_tensor(omega, dtype=DT_R, device=dev).view(-1),
            persistent=False,
        )

        # Индексные массивы верхнего треугольника (1D long)
        self.register_buffer(
            "rows",
            torch.as_tensor(schema.m_rows, dtype=torch.long, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "cols",
            torch.as_tensor(schema.m_cols, dtype=torch.long, device=dev),
            persistent=False,
        )

        self.K = schema.topo.size  # порядок полной матрицы

    # ---------------------------------------------------------------- forward
    def forward(self, param_vec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        param_vec : torch.Tensor
            Батч‑тензор формы (..., L), где L == `schema.size`.

        Returns
        -------
        torch.Tensor
            S‑матрица размера (..., F, P, P) (complex64).
        """
        if param_vec.shape[-1] != self.schema.size:
            raise ValueError(
                f"CMLayer: последняя размерность входа должна быть {self.schema.size}"
            )

        dev = param_vec.device
        # Переносим buffers, если они ещё на CPU и вход на CUDA
        if self.omega.device != dev:
            self.omega = self.omega.to(dev)
            self.rows = self.rows.to(dev)
            self.cols = self.cols.to(dev)

        # --- сборка блоков ---
        M_real, qu, phase_a, phase_b = self.schema.assemble(param_vec, device=dev)

        # --- расчёт S ---
        S = solve_sparams(
            self.spec,
            M_real,
            self.omega,
            qu=qu,
            phase_a=phase_a,
            phase_b=phase_b,
            device=dev,
        )
        return S

    # ---------------------------------------------------------------- helpers
    def extra_repr(self) -> str:
        return (
            f"order={self.schema.topo.order}, ports={self.schema.topo.ports}, "
            f"method={self.spec.method}, fix_sign={self.spec.fix_sign}, "
            f"L={self.schema.size}"
        )
