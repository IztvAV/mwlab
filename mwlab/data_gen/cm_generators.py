# mwlab/data_gen/cm_generators.py
# -*- coding: utf-8 -*-
"""
Синтетические генераторы на основе Coupling-Matrix (CM)
=======================================================

Назначение
----------
Данный модуль реализует два генератора, совместимых с инфраструктурой
``mwlab.data_gen``:

1) :class:`CMGenerator` — «сырое» ядро: на вход принимает батч словарей с
   параметрами матрицы связи (ключи вида ``M1_2``, ``qu``, ``phase_a1`` и т.п.),
   собирает батч тензоров (``M_real``, ``qu``, ``phase_a``, ``phase_b``)
   по схеме :class:`mwlab.filters.cm_schema.ParamSchema`, выполняет один
   векторизованный расчёт :func:`mwlab.filters.cm_core.solve_sparams` на
   **нормированной** частотной сетке Ω и возвращает список объектов
   :class:`mwlab.io.touchstone.TouchstoneData` (либо при желании — список
   тензоров с S-параметрами).

2) :class:`DeviceCMGenerator` — thin-wrapper над CMGenerator, у которого вместо
   нормированной сетки Ω задаётся *реальная* частотная сетка **f** (Гц), а
   преобразование f→Ω выполняется методом конкретного устройства
   (:class:`mwlab.filters.devices.Device`). В метаданные также добавляются
   параметры устройства.

Ключевые свойства
-----------------
* Полностью векторизованный расчёт — один вызов ядра на целый батч.
* Поддержка GPU/CPU (device ``"auto"`` выбирает CUDA при наличии).
* Возвращаемый формат данных настраивается: Touchstone или тензоры
  (см. ``output_mode``).
* Совместимость с хуками :meth:`DataGenerator.preprocess` /
  :meth:`DataGenerator.preprocess_batch`, которые вызываются внутри
  :func:`mwlab.data_gen.runner.run_pipeline`.

Мини-пример
-----------
>>> from mwlab.filters.topologies import get_topology
>>> from mwlab.filters.cm_schema import ParamSchema
>>> from mwlab.data_gen.cm_generators import CMGenerator
>>> topo = get_topology("folded", order=4)
>>> schema = ParamSchema.from_topology(topo, include_qu="vec", include_phase=("a",))
>>> omega = np.linspace(-3, 3, 801)
>>> gen = CMGenerator(topo, omega, schema=schema, output_mode="touchstone")
>>> batch = [
...     {"__id":"p0","M1_1":0.0,"M2_2":0.0,"M3_3":0.0,"M4_4":0.0,
...      "M1_2":0.8,"M2_3":0.8,"M3_4":0.8,"M1_4":0.2,"M1_5":1.0,"M4_6":1.0,
...      "qu_1":1000,"qu_2":1000,"qu_3":1000,"qu_4":1000,"phase_a1":0.0,"phase_a2":0.0},
... ]
>>> outs, meta = gen.generate(batch)
>>> type(outs[0]).__name__
'TouchstoneData'

Примечания
----------
* В нормированном режиме ``CMGenerator`` отмечает метадату флагом
  ``"freq_normalized": True`` и создаёт объект :class:`skrf.Frequency`,
  численно равный Ω (единица измерения — Hz для совместимости с ``skrf``).
* В режиме устройства (:class:`DeviceCMGenerator`) флаг равен ``False``, а
  :class:`skrf.Frequency` создаётся из реальной сетки f (единицы — как на входе).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import torch

from mwlab.data_gen.base import Batch, DataGenerator, MetaBatch, Outputs
from mwlab.filters.cm_core import (
    CoreSpec,
    DEFAULT_DEVICE,
    DT_R,
    solve_sparams,
)
from mwlab.filters.cm_schema import ParamSchema
from mwlab.filters.topologies import Topology
from mwlab.io.touchstone import TouchstoneData

# Для DeviceCMGenerator — подключаем отдельно, чтобы избежать циклов импортов
from mwlab.filters.devices import Device

from typing import TYPE_CHECKING
try:
    import skrf as _rf  # только для TYPE_CHECKING/исключений
except Exception:
    _rf = None

def _require_skrf():
    try:
        import skrf as rf
        return rf
    except ModuleNotFoundError as err:
        raise ImportError(
            "Для output_mode='touchstone' требуется пакет 'scikit-rf'. "
            "Либо установите его: pip install scikit-rf, "
            "либо используйте output_mode='tensor'."
        ) from err


# ─────────────────────────────────────────────────────────────────────────────
#                             Вспомогательные утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _to_tensor_1d(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Любой 1-D массив → torch.Tensor(dtype) на устройстве.

    * Вход: скаляр/список/NumPy/torch.Tensor.
    * Всегда возвращает именно 1-D тензор (через `.flatten()`).
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).flatten()
    return torch.as_tensor(x, dtype=dtype, device=device).flatten()


# ─────────────────────────────────────────────────────────────────────────────
#                                  CMGenerator
# ─────────────────────────────────────────────────────────────────────────────

class CMGenerator(DataGenerator):
    """Генератор S-параметров в нормированной частоте Ω (ядро CM).

    Параметры
    ---------
    topology : Topology
        Граф связей (порядок, число портов, набор ссылок).
    omega_grid : array-like | torch.Tensor
        Нормированная сетка Ω (обычно в диапазоне ~[-3, 3]); форма (F,).
    schema : ParamSchema | None, default None
        Схема параметров. Если None — строится автоматически по ``topology``
        с настройками по умолчанию (``include_qu="vec"``, ``include_phase=("a",)``).
    device : str | torch.device, default "auto"
        Устройство для вычислений. "auto" → CUDA при наличии, иначе CPU.
    method : {"auto","solve","inv"}, default "auto"
        Алгоритм решения в :func:`solve_sparams`.
    output_mode : {"touchstone","tensor"}, default "touchstone"
        Формат выходных объектов:
            * "touchstone" → :class:`TouchstoneData` (рекомендовано для I/O);
            * "tensor"     → :class:`torch.Tensor` формы (F, P, P) для каждого образца.
        В обоих случаях метаданные возвращаются отдельно вторым списком.

    Замечание про метаданные
    ------------------------
    В ``meta`` для каждого образца дублируются исходные параметры батча
    (включая ``"__id"``), а также добавляются поля:

    * ``"topo_order"`` / ``"topo_ports"`` / ``"topo_name"`` — сведения о топологии;
    * ``"freq_normalized": True`` — признак нормированной частоты;
    * ``"grid_size": F`` — число частотных точек.

    Совместимость с хуками
    ----------------------
    :func:`mwlab.data_gen.runner.run_pipeline` перед вызовом :meth:`generate`
    выполнит :meth:`DataGenerator.preprocess_batch` и :meth:`DataGenerator.preprocess`
    над входным батчем — здесь лишь минимальные дополнительные проверки.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        topology: Topology,
        omega_grid: Any,
        *,
        schema: Optional[ParamSchema] = None,
        device: str | torch.device = "auto",
        method: str = "auto",
        output_mode: Literal["touchstone", "tensor"] = "touchstone",
    ) -> None:
        # --- устройство вычислений ---
        if device == "auto":
            self.device = torch.device(DEFAULT_DEVICE)
        else:
            self.device = torch.device(device)

        # --- сетка Ω ---
        self.omega: torch.Tensor = _to_tensor_1d(omega_grid, device=self.device, dtype=DT_R)
        self.F: int = int(self.omega.numel())

        # --- схема параметров ---
        self.topology: Topology = topology
        self.schema: ParamSchema = schema or ParamSchema.from_topology(topology)

        # --- спецификация ядра ---
        self._spec = CoreSpec(self.topology.order, self.topology.ports, method)

        # --- формат выхода ---
        self.output_mode: Literal["touchstone", "tensor"] = output_mode

        # --- частотный объект для скриптов/экспорта (Ω численно → Hz) ---
        # В нормализованном режиме это просто удобный ненулевой rf.Frequency
        # с той же числовой сеткой; downstream-код ориентируется по meta["freq_normalized"].
        self._freq_obj = None
        self._freq_is_normalized = True
        if self.output_mode == "touchstone":
            rf = _require_skrf()
            self._freq_obj = rf.Frequency.from_f(
                self.omega.detach().cpu().numpy(), unit="Hz"
            )
    # ----------------------------------------------------------------- hooks
    def preprocess(self, params: Mapping[str, Any]) -> Mapping[str, Any]:  # noqa: D401
        """Мини-проверки входного словаря.

        Логика:
        * Если `schema.keys` задаёт *фиксированный набор* (множество и т.п.) —
          отклоняем любые неизвестные ключи, кроме служебных `__*` и сахара фаз.
        * Если `schema.keys` «разрешает всё» — мягкий режим: не мешаем метаданным,
          но продолжаем ловить явные опечатки в «CM-похожих» ключах.
        """
        service_keys = {"__id", "__path"}
        sugar_ok = {"phase_a", "phase_b"}

        # Признак «строгости» набора ключей
        is_restricted = isinstance(getattr(self.schema, "keys", None), (set, frozenset))

        if is_restricted:
            unknown = [
                k for k in params.keys()
                if k not in self.schema.keys
                and k not in service_keys
                and k not in sugar_ok
            ]
            if unknown:
                raise KeyError(f"CMGenerator: неизвестные ключи параметров: {unknown}")
            return params

        # Мягкая проверка: рассматриваем только «CM-похожие» ключи
        cm_like_prefixes = ("M", "qu", "phase_a", "phase_b")
        bad = []
        for k in params.keys():
            if k in service_keys or k in sugar_ok:
                continue
            if not k.startswith(cm_like_prefixes):
                continue
            if k not in self.schema.keys:  # AllowAllKeys → всегда True; здесь сработает только реальная несостыковка
                bad.append(k)
        if bad:
            raise KeyError(f"CMGenerator: неизвестные ключи параметров: {bad}")
        return params

    # ---------------------------------------------------------------- generate
    def generate(self, params_batch: Batch) -> Tuple[Outputs, MetaBatch]:
        """Векторизованный расчёт батча S-параметров.

        Контракт
        --------
        Возвращает:
            * `outputs` — список длины `len(params_batch)`,
              каждый элемент — либо TouchstoneData, либо torch.Tensor(F,P,P);
            * `meta` — список словарей такой же длины.
        """
        B = len(params_batch)
        if B == 0:
            return [], []

        # -------- 1) Пакуем батч словарей в (B, L) по схеме --------
        # strict=True: отсутствие ключей будет считаться ошибкой генерации
        vecs: torch.Tensor = torch.stack([
            self.schema.pack(p, device=self.device, dtype=DT_R, strict=True)
            for p in params_batch
        ])  # -> (B, L)

        # -------- 2) Собираем блоки ядра --------
        M_real, qu, phase_a, phase_b = self.schema.assemble(vecs, device=self.device)
        # M_real: (B,K,K), qu: (B,order)|None, phase_*: (B,P)|None

        # -------- 3) Один вызов solve_sparams на весь батч --------
        # Результат: (B, F, P, P) complex64
        with torch.no_grad():
            S_all: torch.Tensor = solve_sparams(
                self._spec,
                M_real,
                self.omega,
                qu=qu,
                phase_a=phase_a,
                phase_b=phase_b,
                device=self.device,
            )
        print("S_all:", tuple(S_all.shape))  # → (B, F, P, P)

        # Проверка формы (для dev-режима):
        B, P = len(params_batch), self.topology.ports
        exp = (B, self.F, P, P)
        if tuple(S_all.shape) != exp:
            raise ValueError(f"solve_sparams returned {tuple(S_all.shape)}; expected {exp}")

        # -------- 4) Готовим выходы и метаданные --------
        outputs: List[Any] = []
        meta_list: List[Dict[str, Any]] = []

        # Общая часть метаданных про топологию и сетку
        topo_meta = {
            "topo_order": self.topology.order,
            "topo_ports": self.topology.ports,
            "topo_name": self.topology.name or "",
            "freq_normalized": True,
            "grid_size": self.F,
        }

        if self.output_mode == "touchstone":
            # Создаём список TouchstoneData: Network берётся из соответствующей срезки S_all[i]
            # skrf ожидает NumPy-массив complex, переводим с устройства на CPU
            rf = _require_skrf()
            freq_obj = self._freq_obj
            assert freq_obj is not None

            for i, p_map in enumerate(params_batch):
                net = rf.Network(frequency=freq_obj, s=S_all[i].detach().cpu().numpy())
                meta = {**p_map, **topo_meta}  # исходные поля + сервисная инфо
                outputs.append(TouchstoneData(net, params=meta))
                meta_list.append(meta)
        else:
            # Режим «tensor»: возвращаем для каждого элемента (F, P, P)
            for i, p_map in enumerate(params_batch):
                s_item = S_all[i]  # (F,P,P), на device генератора
                meta = {**p_map, **topo_meta}
                outputs.append(s_item)
                meta_list.append(meta)

        return outputs, meta_list


# ─────────────────────────────────────────────────────────────────────────────
#                              DeviceCMGenerator
# ─────────────────────────────────────────────────────────────────────────────

class DeviceCMGenerator(CMGenerator):
    """Генератор S-параметров для конкретного устройства (реальная частота f).

    Отличия от :class:`CMGenerator`:
    * На вход подаётся сетка **f** (NumPy/torch или :class:`skrf.Frequency`);
      преобразование f→Ω выполняет сам объект :class:`~mwlab.filters.devices.Device`.
    * В метаданные добавляются параметры устройства (``device._device_params()``).
    * Флаг ``"freq_normalized"`` устанавливается в **False**, а
      :class:`skrf.Frequency` создаётся из *реальной* сетки.

    Параметры
    ---------
    device : :class:`mwlab.filters.devices.Device`
        Экземпляр устройства (Filter и т.п.) с привязанной матрицей связи (cm).
    f_grid : array-like | torch.Tensor | :class:`skrf.Frequency`
        Сетка реальных частот. Если передан :class:`skrf.Frequency`, её единицы
        сохраняются в выходных Touchstone-файлах.
    schema : ParamSchema | None, default None
        Схема параметров. Если None — строится по топологии устройства.
    device_str : str | torch.device, default "auto"
        Устройство для вычислений (аналог `device` в базовом генераторе).
    method : {"auto","solve","inv"}, default "auto"
        Алгоритм решения.
    output_mode : {"touchstone","tensor"}, default "touchstone"
        Формат выходных объектов (как в базовом генераторе).
    """

    def __init__(
        self,
        device: Device,
        f_grid: Any,
        *,
        schema: Optional[ParamSchema] = None,
        device_str: str | torch.device = "auto",
        method: str = "auto",
        output_mode: Literal["touchstone", "tensor"] = "touchstone",
    ) -> None:
        self.device_obj: Device = device
        topology = device.cm.topo

        # ---- 1) Преобразуем входную сетку частот к NumPy; Frequency строим ТОЛЬКО для touchstone ----
        f_arr = np.asarray(f_grid, dtype=float).ravel()
        freq_obj = None
        if output_mode == "touchstone":  # используем ПАРАМЕТР, self.output_mode ещё не инициализирован
            rf = _require_skrf()
            if isinstance(f_grid, rf.Frequency):
                f_arr = f_grid.f.copy()
                freq_obj = f_grid
            else:
                freq_obj = rf.Frequency.from_f(f_arr, unit="Hz")

        # ---- 2) Реализуем f → Ω через устройство ----
        omega = device._omega(f_arr)

        # ---- 3) Схема по топологии устройства, если не передана ----
        schema = schema or ParamSchema.from_topology(topology)

        # ---- 4) Инициализируем базовый генератор на Ω-сетке ----
        super().__init__(
            topology=topology,
            omega_grid=omega,
            schema=schema,
            device=device_str,
            method=method,
            output_mode=output_mode,
        )

        # ---- 5) Переопределяем частотный объект и флаг нормировки ----
        self._freq_obj = freq_obj  # None в режиме 'tensor'
        self._freq_is_normalized = False

    # Переопределять generate() **не требуется**, но мы расширим метаданные.
    def generate(self, params_batch: Batch) -> Tuple[Outputs, MetaBatch]:
        outputs, meta_list = super().generate(params_batch)

        # device-specific поля + вводим явный признак реальной частоты
        dev_meta = self.device_obj._device_params()
        for m in meta_list:  # type: ignore[assignment]
            m["freq_normalized"] = False
            # Информативные параметры устройства (без дублирования order/ports)
            m.update(dev_meta)

        return outputs, meta_list


# ─────────────────────────────────────────────────────────────────────────────
#                                       __all__
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "CMGenerator",
    "DeviceCMGenerator",
]
