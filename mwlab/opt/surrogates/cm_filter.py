# mwlab/opt/surrogates/cm_filter.py
"""
CMFilter — суррогат «coupling matrix → S-параметры» на базе mwlab.filters.

Что улучшено / исправлено относительно исходной версии
------------------------------------------------------
1) **Ускорение через ParamSchema**
   Вместо того, чтобы на каждом вызове:
     - собирать CouplingMatrix из dict,
     - проходить Python-циклом по mvals,
     - заново раскладывать индексы/значения,
   мы один раз строим ParamSchema и работаем с параметрами как с **вектором**.
   Это:
     - заметно уменьшает Python-overhead,
     - позволяет считать батч точек одним вызовом solve_sparams (быстрее).

2) **Корректная Ω-сетка через частоты в Гц**
   Ω всегда вычисляется из частот в **Гц** (Hz), независимо от того,
   в каких единицах пользователь передал f_grid (Hz/kHz/MHz/GHz).
   Это убирает типовые ошибки единиц и делает кэширование Ω корректным.

3) **Lazy import scikit-rf**
   Модуль НЕ требует scikit-rf для расчётов:
     - output="torch"/"numpy" работает без skrf.
     - skrf нужен только когда реально требуется rf.Network / rf.Frequency.
   Если skrf не установлен и нужен network-выход — бросаем понятный ImportError.

Примечание про Filter._omega(...)
--------------------------------
Мы используем приватный метод Filter._omega(f_hz) как «источник истины»
формул Ω↔f (LP/HP/BP/BR). В текущем mwlab.filters.devices.Filter он принимает
частоты **в Гц** и возвращает numpy-массив Ω. Это удобно и не дублирует формулы.

Так как модуль ещё нигде не использовался, обратная совместимость не нужна.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, List, Optional, Tuple, Union, Literal, TYPE_CHECKING

import numpy as np
import torch

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.surrogates import register

from mwlab.filters.topologies import Topology
from mwlab.filters.cm_schema import ParamSchema
from mwlab.filters.cm_core import CoreSpec, solve_sparams
from mwlab.filters.cm import CouplingMatrix  # нужен для создания Filter и для from_filter()
from mwlab.filters.devices import Filter


if TYPE_CHECKING:  # чтобы типы были, но skrf не был обязательной зависимостью
    import skrf as rf


# -----------------------------------------------------------------------------
#                                  Типы
# -----------------------------------------------------------------------------

OutputKind = Literal["network", "numpy", "torch"]


@dataclass(frozen=True, slots=True)
class CMFilterMeta:
    """Лёгкие метаданные для удобной диагностики/логов."""
    kind: str
    unit: str
    f0: Optional[float] = None
    bw: Optional[float] = None
    fbw: Optional[float] = None
    f_edges: Optional[Tuple[Optional[float], Optional[float]]] = None


# -----------------------------------------------------------------------------
#                      Локальные утилиты (единицы, skrf)
# -----------------------------------------------------------------------------

# Мини-таблица множителей единиц. Держим локально, чтобы не зависеть
# от приватных функций других модулей.
_UNIT_MULT: Dict[str, float] = {
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}


def _to_hz(vals: np.ndarray, unit: str) -> np.ndarray:
    """
    Перевод массива частот в Гц.
    unit: 'Hz'/'kHz'/'MHz'/'GHz' (регистр не важен)
    """
    key = str(unit).lower()
    if key not in _UNIT_MULT:
        valid = ", ".join(sorted(_UNIT_MULT))
        raise ValueError(f"Неизвестная единица частоты: {unit!r}. Допустимые: {valid}")
    return np.asarray(vals, dtype=float) * _UNIT_MULT[key]


def _try_import_skrf():
    """
    Пытаемся импортировать scikit-rf.
    Возвращаем модуль или None (без исключения).
    """
    try:
        import skrf as rf  # type: ignore
        return rf
    except ModuleNotFoundError:
        return None


def _require_skrf():
    """
    Жёсткий импорт scikit-rf: если нет — кидаем понятный ImportError.
    """
    rf = _try_import_skrf()
    if rf is None:
        raise ImportError(
            "Эта операция требует установленный пакет 'scikit-rf'. "
            "Установите его, например: pip install scikit-rf"
        )
    return rf


def _is_rf_frequency(obj: Any) -> bool:
    """
    Проверка "obj является rf.Frequency" через isinstance, но с lazy import.
    Если skrf не установлен — всегда False.
    """
    rf = _try_import_skrf()
    if rf is None:
        return False
    return isinstance(obj, rf.Frequency)  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
#                                  CMFilter
# -----------------------------------------------------------------------------

@register("cm_filter", "cm", "coupling_matrix_filter")
class CMFilter(BaseSurrogate):
    """
    Суррогат для 2-портового фильтра:
        x (параметры матрицы связи / потерь / фаз) -> S-параметры на фиксированной сетке.

    Ключевые принципы реализации:
    - На входе x — словарь {str: float}.
    - Внутри всё упаковывается в вектор ParamSchema (быстро и батчево).
    - Для расчёта используется solve_sparams (torch).
    - Выход выбирается параметром output:
        * "torch"   -> torch.Tensor complex64
        * "numpy"   -> np.ndarray complex64
        * "network" -> skrf.Network (требует scikit-rf)
    """

    supports_uncertainty: bool = False

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        topo: Topology,
        # частотная постановка (как в Filter)
        kind: str,
        f_edges: Tuple[Optional[float], Optional[float]] | None = None,
        f0: Optional[float] = None,
        bw: Optional[float] = None,
        fbw: Optional[float] = None,
        # сетка частот (числа или rf.Frequency)
        f_grid: Any,
        unit: str = "Hz",
        # политика добротности
        include_qu: str = "none",  # "none" | "scalar" | "vec"
        Q_fixed: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        # базовые значения
        base_mvals: Optional[Mapping[str, float]] = None,
        base_qu: Optional[Union[float, Sequence[float], np.ndarray]] = None,  # это НОРМИРОВАННЫЙ qu
        base_phase_a: Optional[Union[float, Sequence[float], Mapping[int, float]]] = None,
        base_phase_b: Optional[Union[float, Sequence[float], Mapping[int, float]]] = None,
        # torch/device параметры расчёта
        device: str = "cpu",
        method: str = "auto",
        fix_sign: bool = False,
        # формат выхода
        output: OutputKind = "network",
    ):
        if topo.ports != 2:
            raise ValueError("CMFilter поддерживает только 2-портовые устройства (topo.ports == 2)")

        # -------------------- сохраняем базовые настройки --------------------
        self.topo = topo
        self.device = str(device)
        self.method = str(method)
        self.fix_sign = bool(fix_sign)
        self.output: OutputKind = output

        # Политика добротности:
        #   - "none": qu НЕ является параметром оптимизации, берём фиксированный Q_fixed (физический)
        #   - "scalar"/"vec": qu входит в ParamSchema и может браться из x
        self.include_qu = str(include_qu).lower()
        if self.include_qu not in {"none", "scalar", "vec"}:
            raise ValueError("include_qu должен быть 'none' | 'scalar' | 'vec'")

        # -------------------- частотная сетка: user-unit и Hz --------------------
        # Мы храним одновременно:
        #   - f_grid_user: числа в единицах self.unit (для формирования rf.Frequency при network-выходе)
        #   - f_grid_hz:   те же частоты в Гц (для вычисления Ω)
        self.unit = str(unit)

        self._rf_freq_obj: Optional[Any] = None  # если пользователь дал rf.Frequency — используем его как есть
        if _is_rf_frequency(f_grid):
            # rf.Frequency всегда хранит f в Гц
            self._rf_freq_obj = f_grid
            self.f_grid_hz = np.asarray(f_grid.f, dtype=float)
            # единица для "красивого" вывода — то, что внутри rf.Frequency (если есть)
            self.unit = str(getattr(f_grid, "unit", self.unit))
            # при этом числовой "user" массив удобнее хранить в тех же единицах, что и rf.Frequency
            # но rf.Frequency.from_f ожидает значения в unit, а f_grid.f уже в Hz,
            # поэтому для объекта rf.Frequency мы НЕ пересоздаём сетку — используем f_grid напрямую.
            self.f_grid_user = None
        else:
            arr = np.asarray(f_grid, dtype=float)
            if arr.ndim != 1:
                raise ValueError("f_grid должен быть 1-D массивом частот (или rf.Frequency)")
            self.f_grid_user = arr  # в единицах self.unit
            self.f_grid_hz = _to_hz(arr, self.unit)

        # -------------------- создаём Filter только для Ω↔f и qu_scale --------------------
        # Filter требует cm, но мы НЕ будем использовать cm для расчёта S (чтобы не мутировать и не тратить overhead).
        cm_dummy = CouplingMatrix(topo=self.topo, mvals={})
        self._filter = Filter(
            cm_dummy,
            kind=kind,
            f_edges=f_edges,
            f0=f0,
            bw=bw,
            fbw=fbw,
            name="CMFilter(mapping-only)",
        )

        # qu_scale (LP/HP:1, BP/BR:FBW) нужен для перевода Q -> qu
        self._qu_scale = float(self._filter._qu_scale())

        self.meta = CMFilterMeta(
            kind=self._filter.kind,
            unit=self.unit,
            f0=self._filter.f0,
            bw=self._filter.bw,
            fbw=self._filter.fbw,
            f_edges=self._filter.f_edges,
        )

        # -------------------- фазы: включаем в схему только если заданы базовые значения --------------------
        # Идея:
        #   - если base_phase_a/base_phase_b = None -> фаз нет (и в x их не ждём)
        #   - если base_phase_* задано -> соответствующий блок добавим в ParamSchema,
        #     и пользователь может переопределять phase_* через x.
        include_phase: Tuple[Literal["a", "b"], ...] = ()
        if base_phase_a is not None:
            include_phase = (*include_phase, "a")
        if base_phase_b is not None:
            include_phase = (*include_phase, "b")

        # -------------------- ParamSchema: "единственный источник истины" по параметрам --------------------
        schema_include_qu = "none" if self.include_qu == "none" else self.include_qu  # type: ignore[assignment]
        self.schema: ParamSchema = ParamSchema.from_topology(
            self.topo,
            include_qu=schema_include_qu,  # "none" | "scalar" | "vec"
            include_phase=include_phase,
        )

        # Быстрый мэп ключ->индекс в параметр-векторе.
        self._idx: Dict[str, int] = {k: i for i, k in enumerate(self.schema.keys)}

        # Для удобства: набор ключей M* (чтобы быстро валидировать и применять overrides).
        # Замечание: schema.keys включает также qu и фазы, но нам часто нужно проверять именно блок M.
        self._m_keys = set(self.schema.keys[self.schema.slices["M"].start : self.schema.slices["M"].stop])

        # -------------------- базовый параметр-вектор (CPU), который мы потом клонируем и правим --------------------
        base_params: Dict[str, float] = {}

        # 1) Базовые mvals: разрешаем задавать не все (остальные станут 0).
        #    Важно: запрещаем ключи, которых нет в схеме (например, портовые диагонали).
        if base_mvals:
            for k, v in base_mvals.items():
                if not isinstance(k, str) or not k.startswith("M"):
                    raise ValueError(f"base_mvals: ожидается ключ вида 'M<i>_<j>', получено {k!r}")
                if k not in self._idx:
                    raise ValueError(
                        f"base_mvals: ключ {k!r} отсутствует в ParamSchema. "
                        f"Проверьте, что индексы в пределах topo.size={self.topo.size}, "
                        f"и что это либо резонаторная диагональ, либо связь из topo.links."
                    )
                base_params[k] = float(v)

        # 2) Базовый qu (НОРМИРОВАННЫЙ) — только если include_qu != "none"
        if self.include_qu == "none":
            # Фиксированные потери задаются через физический Q_fixed.
            if Q_fixed is None:
                raise ValueError("include_qu='none' требует Q_fixed (скаляр или вектор длиной order)")
            self._qu_fixed_cpu = self._compute_qu_fixed_cpu(Q_fixed)
        else:
            # base_qu — нормированный qu (то, что напрямую идёт в solve_sparams)
            # Разрешаем:
            #   - scalar -> повторить на все резонаторы (если include_qu='vec') или записать как 'qu' (если scalar)
            #   - vector длиной order -> записать по qu_i
            if base_qu is not None:
                self._apply_base_qu_to_params(base_params, base_qu)

            # Для include_qu != "none" фиксированный Q_fixed не обязателен.
            self._qu_fixed_cpu = None

        # 3) Базовые фазы (если включены)
        if "a" in include_phase:
            self._apply_base_phase_to_params(base_params, "a", base_phase_a)
        if "b" in include_phase:
            self._apply_base_phase_to_params(base_params, "b", base_phase_b)

        # Собираем базовый вектор.
        # strict=False: не требуем заполнения всех ключей (незаполненные -> 0).
        self._base_vec_cpu: torch.Tensor = self.schema.pack(
            base_params,
            device="cpu",
            dtype=torch.float32,
            strict=False,
            default=0.0,
        ).detach()

        # Кэш переноса base_vec на GPU/другие девайсы (чтобы не делать .to(device) постоянно).
        self._base_vec_by_device: Dict[str, torch.Tensor] = {"cpu": self._base_vec_cpu}

        # -------------------- кэш omega (CPU + перенос по устройствам) --------------------
        self._omega_cpu: torch.Tensor = self._compute_omega_cpu()
        self._omega_by_device: Dict[str, torch.Tensor] = {"cpu": self._omega_cpu}

        # -------------------- CoreSpec: можно создать один раз --------------------
        self._core_spec = CoreSpec(
            order=self.topo.order,
            ports=self.topo.ports,
            method=self.method,
            fix_sign=self.fix_sign,
        )

        # -------------------- кэши для network-выхода --------------------
        self._rf_frequency_cached: Optional[Any] = None

    # ----------------------------------------------------------------- helpers: fixed Q -> qu
    def _compute_qu_fixed_cpu(self, Q_fixed: Union[float, Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Преобразуем фиксированный физический Q_fixed в нормированный qu:
            qu = Q * qu_scale
        Возвращаем либо float (скаляр), либо np.ndarray формы (order,).
        """
        q = np.asarray(Q_fixed, dtype=float)
        scale = self._qu_scale

        if q.ndim == 0 or q.size == 1:
            return float(q.reshape(-1)[0]) * scale

        if q.size != self.topo.order:
            raise ValueError(f"Q_fixed: ожидалось {self.topo.order} значений, получено {q.size}")

        return q.reshape(self.topo.order) * scale

    def _qu_fixed(self, device: str) -> Union[float, torch.Tensor, None]:
        """
        Возвращает fixed qu на нужном устройстве:
          - если qu_fixed скаляр -> float (solve_sparams сам расширит)
          - если qu_fixed вектор -> torch.Tensor (order,)
        """
        if self._qu_fixed_cpu is None:
            return None

        if isinstance(self._qu_fixed_cpu, float):
            return self._qu_fixed_cpu

        # вектор -> перенос на device
        dev = str(device)
        arr = self._qu_fixed_cpu
        t_cpu = torch.as_tensor(arr, dtype=torch.float32, device="cpu")
        if dev == "cpu":
            return t_cpu
        return t_cpu.to(dev)

    # ----------------------------------------------------------------- helpers: base qu/phases -> params dict
    def _apply_base_qu_to_params(self, params: Dict[str, float], base_qu: Union[float, Sequence[float], np.ndarray]) -> None:
        """
        Записываем base_qu (НОРМИРОВАННЫЙ) в словарь параметров под ключи схемы.

        Для include_qu='scalar': ожидаем скаляр (или массив из 1 элемента) -> 'qu'
        Для include_qu='vec'   : допускаем скаляр (размножим) или вектор длиной order -> 'qu_i'
        """
        n = self.topo.order
        arr = np.asarray(base_qu, dtype=float)

        if self.include_qu == "scalar":
            if arr.ndim == 0 or arr.size == 1:
                params["qu"] = float(arr.reshape(-1)[0])
                return
            raise ValueError("base_qu: при include_qu='scalar' ожидается скаляр (или 1 элемент), а не вектор")

        # include_qu == "vec"
        if arr.ndim == 0 or arr.size == 1:
            val = float(arr.reshape(-1)[0])
            for i in range(1, n + 1):
                params[f"qu_{i}"] = val
            return

        if arr.size != n:
            raise ValueError(f"base_qu: ожидалось 1 или {n} значений, получено {arr.size}")
        arr = arr.reshape(n)
        for i in range(1, n + 1):
            params[f"qu_{i}"] = float(arr[i - 1])

    def _phase_to_list(self, ph: Any, *, name: str) -> Optional[List[float]]:
        """
        Приводим фазу к списку длиной ports=2 или None.

        Поддерживаем:
          - None
          - скаляр -> одинаковое значение для всех портов
          - последовательность длиной ports
          - mapping {port_index(1-based): value}
        """
        if ph is None:
            return None
        p = self.topo.ports

        if isinstance(ph, (int, float, np.floating)):
            return [float(ph)] * p

        if isinstance(ph, Mapping):
            out = [0.0] * p
            for k, v in ph.items():
                idx = int(k)
                if not (1 <= idx <= p):
                    raise ValueError(f"{name}: индекс {idx} вне диапазона 1…{p}")
                out[idx - 1] = float(v)
            return out

        # sequence-like
        arr = np.asarray(ph, dtype=float).reshape(-1)
        if arr.size != p:
            raise ValueError(f"{name}: ожидалось {p} значений, получено {arr.size}")
        return [float(x) for x in arr.tolist()]

    def _apply_base_phase_to_params(self, params: Dict[str, float], which: Literal["a", "b"], base_phase: Any) -> None:
        """
        Записываем базовые фазы в словарь параметров:
          which='a' -> phase_a1, phase_a2
          which='b' -> phase_b1, phase_b2
        """
        if base_phase is None:
            return
        pref = "phase_a" if which == "a" else "phase_b"
        lst = self._phase_to_list(base_phase, name=pref)
        if lst is None:
            return
        for i, v in enumerate(lst, 1):
            params[f"{pref}{i}"] = float(v)

    # ----------------------------------------------------------------- base_vec + omega caching
    def _base_vec(self, device: str) -> torch.Tensor:
        """
        Возвращает базовый параметр-вектор на заданном устройстве.
        (Кэшируем .to(device), но в predict всё равно делаем clone() для правок.)
        """
        dev = str(device)
        if dev in self._base_vec_by_device:
            return self._base_vec_by_device[dev]
        self._base_vec_by_device[dev] = self._base_vec_cpu.to(dev)
        return self._base_vec_by_device[dev]

    def _compute_omega_cpu(self) -> torch.Tensor:
        """
        Предвычисляет Ω на CPU.

        Важно:
          - частоты на вход Filter._omega должны быть в Гц.
          - Filter._omega возвращает numpy-массив.
        """
        # Частоты в Гц -> Ω
        omega_np = self._filter._omega(self.f_grid_hz)  # приватный метод, но формулы "истина"
        omega_np = np.asarray(omega_np, dtype=float).reshape(-1)

        # Переводим в torch.float32 на CPU
        w = torch.as_tensor(omega_np, dtype=torch.float32, device="cpu").detach()
        return w

    def _omega(self, device: str) -> torch.Tensor:
        """
        Возвращает omega-тензор на заданном устройстве.
        Кэшируем перенос, чтобы не делать .to(device) на каждом вызове.
        """
        dev = str(device)
        if dev in self._omega_by_device:
            return self._omega_by_device[dev]
        self._omega_by_device[dev] = self._omega_cpu.to(dev)
        return self._omega_by_device[dev]

    # ----------------------------------------------------------------- x -> vec (single/batch)
    def _vector_from_x(self, x: Mapping[str, float], device: str) -> torch.Tensor:
        """
        Собираем параметр-вектор (L,) для одной точки x:
          vec = base_vec + overrides(x)

        Важное правило:
          - все ключи вида "M..." должны присутствовать в schema (иначе ошибка),
            чтобы не оптимизировать параметры вне топологии.
        """
        vec = self._base_vec(device).clone()

        # 1) M-ключи: строго должны быть в схеме
        for k, v in x.items():
            if not isinstance(k, str):
                continue
            if k.startswith("M"):
                if k not in self._m_keys:
                    raise ValueError(f"Неизвестный/запрещённый параметр матрицы связи: {k!r}")
                vec[self._idx[k]] = float(v)

        # 2) qu/Q (если разрешено)
        if self.include_qu == "none":
            # При фиксированном Q мы запрещаем qu/Q в x, чтобы не было двусмысленности.
            for bad in ("qu", "Q"):
                if bad in x:
                    raise ValueError(f"include_qu='none': параметр {bad!r} в x запрещён (используйте Q_fixed)")
            if any(str(k).startswith(("qu_", "Q_")) for k in x.keys()):
                raise ValueError("include_qu='none': параметры qu_*/Q_* в x запрещены (используйте Q_fixed)")
        else:
            self._apply_qu_overrides_inplace(vec, x)

        # 3) фазы (если включены в схему)
        self._apply_phase_overrides_inplace(vec, x)

        return vec

    def _vectors_from_xs(self, xs: Sequence[Mapping[str, float]], device: str) -> torch.Tensor:
        """
        Батчевая версия: собираем vec_batch формы (B,L).
        Делается через:
          base_vec.expand(B,L).clone() + in-place overrides по каждой точке.
        """
        if not xs:
            raise ValueError("_vectors_from_xs: пустой batch")

        base = self._base_vec(device)  # (L,)
        B = len(xs)
        vecs = base.expand(B, base.shape[0]).clone()

        for i, x in enumerate(xs):
            # M
            for k, v in x.items():
                if not isinstance(k, str):
                    continue
                if k.startswith("M"):
                    if k not in self._m_keys:
                        raise ValueError(f"Неизвестный/запрещённый параметр матрицы связи: {k!r}")
                    vecs[i, self._idx[k]] = float(v)

            # qu/Q
            if self.include_qu == "none":
                for bad in ("qu", "Q"):
                    if bad in x:
                        raise ValueError(f"include_qu='none': параметр {bad!r} в x запрещён (используйте Q_fixed)")
                if any(str(k).startswith(("qu_", "Q_")) for k in x.keys()):
                    raise ValueError("include_qu='none': параметры qu_*/Q_* в x запрещены (используйте Q_fixed)")
            else:
                self._apply_qu_overrides_inplace(vecs[i], x)

            # phases
            self._apply_phase_overrides_inplace(vecs[i], x)

        return vecs

    # ----------------------------------------------------------------- overrides: qu/Q
    def _apply_qu_overrides_inplace(self, vec: torch.Tensor, x: Mapping[str, float]) -> None:
        """
        Применяем параметры добротности из x к vec (in-place).

        Поддерживаем:
          - qu (скаляр)
          - qu_1..qu_n (вектор)
          - Q (скаляр) или Q_1..Q_n (вектор) -> перевод в qu через qu_scale

        Правила при include_qu='scalar':
          - используем только скаляр (qu или Q)
          - векторные варианты (qu_i/Q_i) запрещаем, чтобы не было "тихих" ошибок.

        Правила при include_qu='vec':
          - qu (скаляр) -> размножаем на все qu_i
          - Q (скаляр)  -> размножаем на все qu_i (и домножаем на scale)
          - qu_i/Q_i -> если встретился хотя бы один, требуем наличие всех i=1..n
        """
        n = self.topo.order
        scale = self._qu_scale

        has_qu_scalar = "qu" in x
        has_Q_scalar = "Q" in x

        has_qu_vec = any(f"qu_{i}" in x for i in range(1, n + 1))
        has_Q_vec = any(f"Q_{i}" in x for i in range(1, n + 1))

        # Запрещаем одновременно Q и qu (любых форм)
        if (has_qu_scalar or has_qu_vec) and (has_Q_scalar or has_Q_vec):
            raise ValueError("Нельзя одновременно задавать 'qu*' и 'Q*' в одной точке x")

        if self.include_qu == "scalar":
            if has_qu_vec or has_Q_vec:
                raise ValueError("include_qu='scalar': векторные qu_i/Q_i запрещены, используйте 'qu' или 'Q'")

            if has_qu_scalar:
                vec[self._idx["qu"]] = float(x["qu"])
            elif has_Q_scalar:
                vec[self._idx["qu"]] = float(x["Q"]) * scale
            return

        # include_qu == "vec"
        if has_qu_scalar:
            val = float(x["qu"])
            for i in range(1, n + 1):
                vec[self._idx[f"qu_{i}"]] = val
            return

        if has_Q_scalar:
            val = float(x["Q"]) * scale
            for i in range(1, n + 1):
                vec[self._idx[f"qu_{i}"]] = val
            return

        if has_qu_vec:
            for i in range(1, n + 1):
                k = f"qu_{i}"
                if k not in x:
                    raise KeyError(f"Ожидался параметр {k} для qu-вектора (order={n})")
                vec[self._idx[k]] = float(x[k])
            return

        if has_Q_vec:
            for i in range(1, n + 1):
                k = f"Q_{i}"
                if k not in x:
                    raise KeyError(f"Ожидался параметр {k} для Q-вектора (order={n})")
                vec[self._idx[f"qu_{i}"]] = float(x[k]) * scale
            return

        # Если ничего про qu/Q в x нет — оставляем базовые значения.

    # ----------------------------------------------------------------- overrides: phases
    def _apply_phase_overrides_inplace(self, vec: torch.Tensor, x: Mapping[str, float]) -> None:
        """
        Применяем фазовые параметры из x к vec (in-place), если фазы включены в схему.

        Поддерживаем сахар:
          - phase_a (скаляр) -> phase_a1..phase_aP
          - phase_b (скаляр) -> phase_b1..phase_bP
          - phase_a1, phase_a2 ... (можно задавать частично)
        """
        ports = self.topo.ports

        # Если фазовые блоки не включены в схему, но пользователь пытается их задать — это ошибка.
        schema_has_a = self.schema.slices["phase_a"].stop > self.schema.slices["phase_a"].start
        schema_has_b = self.schema.slices["phase_b"].stop > self.schema.slices["phase_b"].start

        wants_a = ("phase_a" in x) or any(f"phase_a{i}" in x for i in range(1, ports + 1))
        wants_b = ("phase_b" in x) or any(f"phase_b{i}" in x for i in range(1, ports + 1))

        if wants_a and not schema_has_a:
            raise ValueError("phase_a задан в x, но фазы 'a' не включены (base_phase_a=None в конструкторе)")
        if wants_b and not schema_has_b:
            raise ValueError("phase_b задан в x, но фазы 'b' не включены (base_phase_b=None в конструкторе)")

        # phase_a: сахар скаляром
        if schema_has_a and "phase_a" in x:
            val = float(x["phase_a"])
            for i in range(1, ports + 1):
                vec[self._idx[f"phase_a{i}"]] = val

        # phase_a: покомпонентно (можно частично)
        if schema_has_a:
            for i in range(1, ports + 1):
                k = f"phase_a{i}"
                if k in x:
                    vec[self._idx[k]] = float(x[k])

        # phase_b: сахар скаляром
        if schema_has_b and "phase_b" in x:
            val = float(x["phase_b"])
            for i in range(1, ports + 1):
                vec[self._idx[f"phase_b{i}"]] = val

        # phase_b: покомпонентно
        if schema_has_b:
            for i in range(1, ports + 1):
                k = f"phase_b{i}"
                if k in x:
                    vec[self._idx[k]] = float(x[k])

    # ----------------------------------------------------------------- rf helpers
    def _rf_frequency(self):
        """
        Возвращает объект rf.Frequency для создания rf.Network.
        Ленивая зависимость: импортим skrf только здесь.
        """
        rf = _require_skrf()

        if self._rf_freq_obj is not None:
            return self._rf_freq_obj

        if self._rf_frequency_cached is None:
            # Здесь важно: rf.Frequency.from_f ожидает числа в единицах `unit`.
            # Поэтому используем f_grid_user (если сетка была численной).
            if self.f_grid_user is None:
                # Теоретически сюда не должны попасть (rf_freq_obj задан выше),
                # но оставим защиту.
                self._rf_frequency_cached = rf.Frequency.from_f(self.f_grid_hz, unit="hz")
            else:
                self._rf_frequency_cached = rf.Frequency.from_f(
                    np.asarray(self.f_grid_user, dtype=float),
                    unit=str(self.unit).lower(),
                )

        return self._rf_frequency_cached

    def _to_numpy(self, S_t: torch.Tensor) -> np.ndarray:
        """torch.Tensor -> np.ndarray complex64"""
        return S_t.detach().cpu().numpy()

    def _to_network(self, S_t: torch.Tensor):
        """torch.Tensor -> rf.Network (требует scikit-rf)"""
        rf = _require_skrf()
        S_np = self._to_numpy(S_t)
        return rf.Network(frequency=self._rf_frequency(), s=S_np)

    # ----------------------------------------------------------------- low-level compute (torch)
    def _predict_s_torch_single(self, x: Mapping[str, float]) -> torch.Tensor:
        """
        Расчёт S(Ω) для одной точки -> torch.Tensor complex64 формы (F,2,2).
        """
        dev = self.device
        vec = self._vector_from_x(x, dev)
        M_real, qu, phase_a, phase_b = self.schema.assemble(vec, device=dev)

        omega = self._omega(dev)  # (F,)
        if self.include_qu == "none":
            qu = self._qu_fixed(dev)

        S = solve_sparams(
            self._core_spec,
            M_real,
            omega,
            qu=qu,
            phase_a=phase_a,
            phase_b=phase_b,
            device=dev,
        )
        return S

    def _predict_s_torch_batch(self, xs: Sequence[Mapping[str, float]]) -> torch.Tensor:
        """
        Батчевый расчёт:
          xs (B) -> torch.Tensor complex64 формы (B,F,2,2)

        Здесь выигрываем за счёт того, что solve_sparams работает батчево.
        """
        dev = self.device
        vecs = self._vectors_from_xs(xs, dev)
        M_real, qu, phase_a, phase_b = self.schema.assemble(vecs, device=dev)

        omega = self._omega(dev)  # (F,)
        if self.include_qu == "none":
            qu = self._qu_fixed(dev)

        S = solve_sparams(
            self._core_spec,
            M_real,
            omega,
            qu=qu,
            phase_a=phase_a,
            phase_b=phase_b,
            device=dev,
        )
        return S  # (B,F,2,2)

    # ----------------------------------------------------------------- public convenience: network
    def predict_network(self, x: Mapping[str, float]):
        """
        Всегда возвращает rf.Network.
        Требует установленный scikit-rf.
        """
        S_t = self._predict_s_torch_single(x)
        return self._to_network(S_t)

    def batch_predict_network(self, xs: Sequence[Mapping[str, float]]):
        """
        Всегда возвращает list[rf.Network].
        Требует установленный scikit-rf.
        """
        rf = _require_skrf()
        if not xs:
            return []

        S_b = self._predict_s_torch_batch(xs)  # (B,F,2,2)
        S_np = self._to_numpy(S_b)             # (B,F,2,2)
        freq = self._rf_frequency()

        return [rf.Network(frequency=freq, s=S_np[i]) for i in range(S_np.shape[0])]

    # ----------------------------------------------------------------- BaseSurrogate API
    def predict(self, x: Mapping[str, float], *, return_std: bool = False):
        """
        Основной контракт BaseSurrogate: predict(x).

        return_std не поддерживается (нет σ).
        """
        if return_std:
            raise NotImplementedError("CMFilter does not support uncertainty (σ).")

        S_t = self._predict_s_torch_single(x)

        if self.output == "torch":
            return S_t
        if self.output == "numpy":
            return self._to_numpy(S_t)
        if self.output == "network":
            return self._to_network(S_t)

        raise ValueError(f"Unknown output kind: {self.output!r}")

    def batch_predict(self, xs: Sequence[Mapping[str, float]], *, return_std: bool = False):
        """
        Батчевый API.

        - output="torch"  -> torch.Tensor (B,F,2,2)
        - output="numpy"  -> np.ndarray   (B,F,2,2)
        - output="network"-> list[rf.Network]
        """
        if return_std:
            raise NotImplementedError("CMFilter does not support uncertainty (σ).")

        if not xs:
            if self.output == "network":
                return []
            if self.output == "numpy":
                return np.zeros((0, self.f_grid_hz.size, 2, 2), dtype=np.complex64)
            if self.output == "torch":
                return torch.zeros((0, self.f_grid_hz.size, 2, 2), dtype=torch.complex64)
            raise ValueError(f"Unknown output kind: {self.output!r}")

        if self.output == "network":
            return self.batch_predict_network(xs)

        S_b = self._predict_s_torch_batch(xs)  # (B,F,2,2)

        if self.output == "torch":
            return S_b
        if self.output == "numpy":
            return self._to_numpy(S_b)

        raise ValueError(f"Unknown output kind: {self.output!r}")

    def passes_spec(self, xs, spec):
        """
        Проверка спецификации по предсказаниям.

        Для совместимости со старым стилем Specification предполагаем,
        что spec.is_ok(...) ожидает rf.Network, поэтому здесь всегда строим Network.
        """
        from collections.abc import Mapping as _Mapping

        # одиночная точка
        if isinstance(xs, _Mapping):
            net = self.predict_network(xs)
            return bool(spec.is_ok(net))

        # батч
        nets = self.batch_predict_network(xs)
        return np.fromiter((spec.is_ok(net) for net in nets), dtype=bool)

    # -------------------------------------------------------------------- property filter
    @property
    def filter(self) -> Filter:
        """
        Доступ к «внутреннему» объекту mwlab.filters.devices.Filter.

        Он используется CMFilter как источник истины для:
          - преобразования частот f -> Ω,
          - вычисления qu_scale.

        ВАЖНО:
          - CMFilter НЕ использует self.filter.cm при расчёте S (мы НЕ мутируем cm).
          - Не рассчитывайте, что изменения self.filter.cm повлияют на predict().
            Для изменения матрицы связи используйте x (M.../Q/qu) или base_* параметры.
        """
        return self._filter

    # ----------------------------------------------------------------- factory from_filter
    @classmethod
    def from_filter(
            cls,
            flt: Filter,
            *,
            f_grid: Any,
            unit: str = "Hz",
            include_qu: str = "none",
            Q_fixed: Optional[Union[float, Sequence[float], np.ndarray]] = None,
            base_mvals: Optional[Mapping[str, float]] = None,
            device: str = "cpu",
            method: str = "auto",
            fix_sign: bool = False,
            output: OutputKind = "network",
    ) -> "CMFilter":
        """
        Создать CMFilter из существующего Filter.

        Берём:
          - topo, kind, f_edges/f0/bw/fbw из flt
          - base значения из flt.cm (плюс опциональный override base_mvals)

        Важно про потери:
          - include_qu="none": qu НЕ оптимизируем, берём физический Q_fixed.
            Если Q_fixed не передан, пытаемся восстановить его из flt.cm.qu:
                Q_fixed = flt.cm.qu / flt._qu_scale()
            (то же самое, что flt.Q)
        """
        cm = flt.cm

        # --- базовые mvals ---
        m0 = dict(cm.mvals)
        if base_mvals:
            m0.update({k: float(v) for k, v in base_mvals.items()})

        include_qu_l = str(include_qu).lower()

        # --- авто-восстановление Q_fixed из flt.cm.qu ---
        if include_qu_l == "none" and Q_fixed is None:
            if cm.qu is None:
                raise ValueError(
                    "from_filter(..., include_qu='none'): нужно Q_fixed, "
                    "но flt.cm.qu is None, восстановить физический Q невозможно."
                )

            scale = float(flt._qu_scale())  # LP/HP:1, BP/BR:FBW
            if scale == 0.0:
                raise ValueError("flt._qu_scale() вернул 0 — невозможно восстановить Q_fixed из qu.")

            # cm.qu может быть float/list/np/torch — аккуратно приводим к numpy
            import numpy as _np
            import torch as _torch

            if _torch.is_tensor(cm.qu):
                qarr = cm.qu.detach().cpu().numpy().astype(float, copy=False)
            else:
                qarr = _np.asarray(cm.qu, dtype=float)

            # скаляр или вектор
            if qarr.ndim == 0 or qarr.size == 1:
                Q_fixed = float(qarr.reshape(-1)[0] / scale)
            else:
                Q_fixed = (qarr.reshape(-1) / scale)

        # Для include_qu="none" base_qu не нужен (и даже нежелателен семантически)
        base_qu = None if include_qu_l == "none" else cm.qu

        return cls(
            topo=cm.topo,
            kind=flt.kind,
            f_edges=flt.f_edges,
            f0=flt.f0,
            bw=flt.bw,
            fbw=flt.fbw,
            f_grid=f_grid,
            unit=unit,
            include_qu=include_qu,
            Q_fixed=Q_fixed,
            base_mvals=m0,
            base_qu=base_qu,
            base_phase_a=cm.phase_a,
            base_phase_b=cm.phase_b,
            device=device,
            method=method,
            fix_sign=fix_sign,
            output=output,
        )


    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CMFilter(order={self.topo.order}, kind={self.meta.kind}, "
            f"include_qu={self.include_qu!r}, output={self.output!r}, device={self.device!r})"
        )


__all__ = ["CMFilter", "CMFilterMeta"]
