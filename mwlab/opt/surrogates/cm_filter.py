# mwlab/opt/surrogates/cm_filter.py
"""
CMFilter — суррогат «матрица связи → S-параметры» на базе mwlab.filters.

Ключевая цель этого модуля — быть *безопасным* и *совместимым* с текущими
Objective/Specification из mwlab.opt.objectives:
  - predict(x) по умолчанию возвращает skrf.Network;
  - batch_predict(xs) по умолчанию возвращает list[skrf.Network];
  - passes_spec(xs, spec) работает без трюков вида временного self.output = ...

Минимальные изменения относительно предыдущей версии (по твоим пунктам):
  1) Добавлены predict_network(x) и batch_predict_network(xs) — всегда rf.Network.
  2) Убрано мутирование self._filter.cm = ...
     Расчёт S выполняется напрямую через cm.sparams(omega_cached, ...).
  3) Добавлен кэш omega (нормированная частота Ω), вычисляемая через Filter._omega(...).
     Это ускоряет многократные вызовы (особенно в оптимизации).
  4) passes_spec использует predict_network/batch_predict_network напрямую.

ВАЖНО ПРО _omega:
  Сейчас мы сознательно используем приватный метод Filter._omega(...) как
  «источник истины» для Ω↔f отображения (LP/HP/BP/BR), чтобы не дублировать формулы.
  Если позже появится публичный эквивалент — достаточно заменить _compute_omega_cpu().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, List, Optional, Tuple, Union, Literal

import numpy as np
import skrf as rf

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.surrogates import register

from mwlab.filters.cm import CouplingMatrix, parse_m_key
from mwlab.filters.topologies import Topology
from mwlab.filters.devices import Filter


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
#                                  CMFilter
# -----------------------------------------------------------------------------

@register("cm_filter", "cm", "coupling_matrix_filter")
class CMFilter(BaseSurrogate):
    """
    Surrogate для 2-портового фильтра, считающий S-параметры по coupling-matrix.

    По умолчанию возвращает rf.Network (output="network"), т.е. напрямую
    совместим с Selector/Aggregator/Specification.
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
        # сетка частот
        f_grid: Union[np.ndarray, Sequence[float], rf.Frequency],
        unit: str = "Hz",
        # управление добротностью
        include_qu: str = "none",
        Q_fixed: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        # базовая матрица связи
        base_mvals: Optional[Mapping[str, float]] = None,
        base_qu: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        base_phase_a: Optional[Union[float, Sequence[float], Mapping[int, float]]] = None,
        base_phase_b: Optional[Union[float, Sequence[float], Mapping[int, float]]] = None,
        # torch/device параметры расчёта
        device: str = "cpu",
        method: str = "auto",
        fix_sign: bool = False,
        # формат выхода
        output: OutputKind = "network",
    ):
        """
        Parameters
        ----------
        topo : Topology
            Топология матрицы связи. Для этого суррогата ports должен быть 2.
        kind, f_edges/f0/bw/fbw
            Частотная постановка (LP/HP/BP/BR) и параметры преобразования Ω↔f.
        f_grid : array-like | rf.Frequency
            Частотная сетка для выдачи S-параметров.
        unit : str
            Единицы f_grid, если f_grid задан численно (Hz/kHz/MHz/GHz).
        include_qu : str
            Политика добротности:
              - "none": добротность не оптимизируем, используем Q_fixed
              - иначе: разрешаем брать Q/qu из x (см. _extract_qu_from_x)
        Q_fixed : float | array-like | None
            Физическая добротность Q (скаляр или длиной order).
        base_mvals/base_qu/base_phase_a/base_phase_b
            Параметры по умолчанию (если не переопределены в x).
        device/method/fix_sign
            Пробрасываются в cm.sparams(...)
        output : {"network","numpy","torch"}
            - network -> rf.Network (single) / list[rf.Network] (batch)
            - numpy   -> np.ndarray (F,2,2) / (B,F,2,2)
            - torch   -> torch.Tensor complex64 (F,2,2) / (B,F,2,2)
        """
        if topo.ports != 2:
            raise ValueError("CMFilter поддерживает только 2-портовые устройства (topo.ports == 2)")

        self.topo = topo
        self.include_qu = str(include_qu)
        self.Q_fixed = Q_fixed

        self.device = str(device)
        self.method = str(method)
        self.fix_sign = bool(fix_sign)

        self.output: OutputKind = output
        self.unit = str(unit)

        # --- частотная сетка ---
        if isinstance(f_grid, rf.Frequency):
            self._rf_freq_obj: Optional[rf.Frequency] = f_grid
            # skrf хранит frequency.f в Гц
            self.f_grid = np.asarray(f_grid.f, dtype=float)

            # unit "для пользователя" (как задано в rf.Frequency), но omega считаем в Hz
            self.unit = str(getattr(f_grid, "unit", "Hz"))
            self._omega_unit = "Hz"
        else:
            self._rf_freq_obj = None
            arr = np.asarray(f_grid, dtype=float)
            if arr.ndim != 1:
                raise ValueError("f_grid должен быть 1-D массивом частот")
            self.f_grid = arr

            self.unit = str(unit)
            self._omega_unit = self.unit

        # --- базовая CouplingMatrix (шаблон) ---
        mvals0 = dict(base_mvals or {})
        self._cm_base = CouplingMatrix(
            topo=self.topo,
            mvals=mvals0,
            qu=base_qu,
            phase_a=base_phase_a,
            phase_b=base_phase_b,
        )

        # --- Filter нужен как «контейнер» параметров Ω↔f и для qu_scale ---
        self._filter = Filter(
            self._cm_base,  # cm здесь только для инициализации Filter; дальше cm не мутируем
            kind=kind,
            f_edges=f_edges,
            f0=f0,
            bw=bw,
            fbw=fbw,
            name="CMFilter(base)",
        )
        # qu_scale постоянен для фиксированного kind/fbw
        self._qu_scale = float(self._filter._qu_scale())

        self.meta = CMFilterMeta(
            kind=self._filter.kind,
            unit=self.unit,
            f0=self._filter.f0,
            bw=self._filter.bw,
            fbw=self._filter.fbw,
            f_edges=self._filter.f_edges,
        )

        # --- валидация политики добротности ---
        if self.include_qu == "none":
            if Q_fixed is None:
                raise ValueError("include_qu='none' требует Q_fixed (скаляр или вектор длиной order)")
            q_arr = np.asarray(Q_fixed, dtype=float)
            if q_arr.ndim > 0 and q_arr.size not in (1, self.topo.order):
                raise ValueError(f"Q_fixed: ожидалось 1 или {self.topo.order} значений, получено {q_arr.size}")

        # --- кэш rf.Frequency для rf.Network (часто используется в Objective/Specification) ---
        self._rf_frequency_cached: Optional[rf.Frequency] = None

        # --- кэш omega: CPU-версия + перенос на нужное устройство по требованию ---
        self._omega_cpu = self._compute_omega_cpu()
        self._omega_by_device: Dict[str, Any] = {}  # device -> torch.Tensor

    # ----------------------------------------------------------------- удобства
    @property
    def filter(self) -> Filter:
        """
        Возвращает объект Filter (референс).
        Важно: CMFilter НЕ мутирует self.filter.cm в predict().
        """
        return self._filter

    @classmethod
    def from_filter(
        cls,
        flt: Filter,
        *,
        f_grid: Union[np.ndarray, Sequence[float], rf.Frequency],
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
        Создать CMFilter из уже существующего Filter.

        topo + частотные параметры берём из flt.
        Базовые mvals/qu/phase — из flt.cm (с опциональным перекрытием base_mvals).
        """
        cm = flt.cm
        m0 = dict(cm.mvals)
        if base_mvals:
            m0.update({k: float(v) for k, v in base_mvals.items()})

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
            base_qu=cm.qu,
            base_phase_a=cm.phase_a,
            base_phase_b=cm.phase_b,
            device=device,
            method=method,
            fix_sign=fix_sign,
            output=output,
        )

    # ----------------------------------------------------------------- omega cache
    def _compute_omega_cpu(self):
        """
        Предвычисляет Ω для текущей частотной сетки на CPU.
        Гарантирует torch.Tensor на CPU.
        """
        if not hasattr(self._filter, "_omega"):
            raise RuntimeError(
                "Filter не содержит _omega(...). Нужен публичный эквивалент "
                "или придётся реализовать Ω↔f в этом модуле."
            )

        w = self._filter._omega(self.f_grid, unit=self._omega_unit, device="cpu")  # type: ignore[attr-defined]

        import torch
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(w, dtype=torch.float32)
        else:
            # на всякий случай — гарантируем CPU
            w = w.detach().to("cpu")

        if w.ndim != 1:
            w = w.reshape(-1)

        return w

    def _omega(self, device: str):
        """
        Возвращает omega-тензор на заданном устройстве.
        Кэшируем перенос, чтобы не делать .to(device) на каждом вызове.
        """
        dev = str(device)
        if dev == "cpu":
            return self._omega_cpu

        if dev in self._omega_by_device:
            return self._omega_by_device[dev]

        import torch  # локальный импорт
        w = self._omega_cpu
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(w, dtype=torch.float32)
        self._omega_by_device[dev] = w.to(dev)
        return self._omega_by_device[dev]

    # ----------------------------------------------------------------- извлечение M/qu из x
    def _extract_mvals_from_x(self, x: Mapping[str, float]) -> Dict[str, float]:
        """
        Извлекает параметры вида "M<i>_<j>" из x.

        Индексы i,j — 1-based, как в mwlab.filters.cm.parse_m_key.
        """
        mvals: Dict[str, float] = {}
        K = self.topo.size
        for k, v in x.items():
            if not isinstance(k, str) or not k.startswith("M"):
                continue
            i, j = parse_m_key(k)
            if not (1 <= i <= K and 1 <= j <= K):
                raise ValueError(f"Параметр {k!r}: индексы выходят за предел 1…{K}")
            mvals[k] = float(v)
        return mvals

    def _extract_qu_from_x(self, x: Mapping[str, float]) -> Optional[Union[float, np.ndarray]]:
        """
        Извлекает добротности из x, если include_qu != "none".

        Поддерживаем:
          - qu (скаляр) или qu_1..qu_n (вектор)  -> возвращаем как есть (это cm.qu)
          - Q  (скаляр) или Q_1..Q_n  (вектор)   -> переводим в qu через qu_scale

        Нормировка qu_scale:
          - LP/HP: 1
          - BP/BR: FBW (см. Filter._qu_scale())
        """
        if self.include_qu == "none":
            return None

        n = self.topo.order
        scale = self._qu_scale

        # 1) qu
        if "qu" in x:
            return float(x["qu"])
        keys = [f"qu_{i}" for i in range(1, n + 1)]
        if any(k in x for k in keys):
            arr = np.zeros((n,), dtype=float)
            for i, k in enumerate(keys, 1):
                if k not in x:
                    raise KeyError(f"Ожидался параметр {k} для qu-вектора (order={n})")
                arr[i - 1] = float(x[k])
            return arr

        # 2) Q -> qu
        if "Q" in x:
            return float(x["Q"]) * scale
        keys = [f"Q_{i}" for i in range(1, n + 1)]
        if any(k in x for k in keys):
            arr = np.zeros((n,), dtype=float)
            for i, k in enumerate(keys, 1):
                if k not in x:
                    raise KeyError(f"Ожидался параметр {k} для Q-вектора (order={n})")
                arr[i - 1] = float(x[k]) * scale
            return arr

        return None

    # ----------------------------------------------------------------- сборка CouplingMatrix
    def _build_cm(self, x: Mapping[str, float]) -> CouplingMatrix:
        """
        Собирает CouplingMatrix для точки x:
          - mvals: base + overrides из x
          - qu:
              include_qu == "none" -> Q_fixed (физический) -> qu через qu_scale
              иначе               -> qu/Q из x (если есть), иначе base_qu
          - фазы берём из базовой cm (на первом этапе фиксированные)
        """
        # mvals
        mvals = dict(self._cm_base.mvals)
        mvals.update(self._extract_mvals_from_x(x))

        # qu
        qu_final = self._cm_base.qu

        if self.include_qu == "none":
            scale = self._qu_scale
            q = np.asarray(self.Q_fixed, dtype=float)
            if q.ndim == 0 or q.size == 1:
                qu_final = float(q.reshape(-1)[0]) * scale
            else:
                if q.size != self.topo.order:
                    raise ValueError(f"Q_fixed: ожидалось {self.topo.order} значений, получено {q.size}")
                qu_final = q * scale
        else:
            qu_from_x = self._extract_qu_from_x(x)
            if qu_from_x is not None:
                qu_final = qu_from_x

        return CouplingMatrix(
            topo=self.topo,
            mvals=mvals,
            qu=qu_final,
            phase_a=self._cm_base.phase_a,
            phase_b=self._cm_base.phase_b,
        )

    # ----------------------------------------------------------------- rf.Frequency helper
    def _rf_frequency(self) -> rf.Frequency:
        """
        Возвращает rf.Frequency, соответствующий self.f_grid.
        Кэшируем объект, потому что он часто создаётся при выдаче rf.Network.
        """
        if self._rf_freq_obj is not None:
            return self._rf_freq_obj

        if self._rf_frequency_cached is None:
            self._rf_frequency_cached = rf.Frequency.from_f(
                np.asarray(self.f_grid, dtype=float),
                unit=self.unit.lower(),
            )
        return self._rf_frequency_cached

    # ----------------------------------------------------------------- low-level compute
    def _predict_s_torch(self, x: Mapping[str, float]):
        """
        Низкоуровневый расчёт S(Ω) -> torch.Tensor complex64 формы (F,2,2).

        Здесь нет rf.Network и нет мутирования Filter.
        """
        cm = self._build_cm(x)
        omega = self._omega(self.device)  # torch.Tensor на self.device
        # CouplingMatrix.sparams ожидает omega и возвращает torch complex (F,2,2)
        return cm.sparams(
            omega,
            device=self.device,
            method=self.method,
            fix_sign=self.fix_sign,
        )

    # ----------------------------------------------------------------- output converters
    def _to_numpy(self, S_t) -> np.ndarray:
        """torch.Tensor -> np.ndarray"""
        return S_t.detach().cpu().numpy()

    def _to_network(self, S_t) -> rf.Network:
        """torch.Tensor -> rf.Network"""
        S_np = self._to_numpy(S_t)
        return rf.Network(frequency=self._rf_frequency(), s=S_np)

    # ----------------------------------------------------------------- public: predict_network
    def predict_network(self, x: Mapping[str, float]) -> rf.Network:
        """
        Всегда возвращает rf.Network (не зависит от self.output).
        Удобно для passes_spec и любых мест, где требуется гарантированный Network.
        """
        S_t = self._predict_s_torch(x)
        return self._to_network(S_t)

    def batch_predict_network(self, xs: Sequence[Mapping[str, float]]) -> List[rf.Network]:
        """
        Всегда возвращает список rf.Network (не зависит от self.output).

        Примечание:
          Здесь есть цикл по xs, потому что у каждой точки своя CouplingMatrix.
          При этом внутри нет цикла по частоте: omega — общий вектор.
        """
        if not xs:
            return []
        freq = self._rf_frequency()
        nets: List[rf.Network] = []
        for x in xs:
            S_t = self._predict_s_torch(x)
            nets.append(rf.Network(frequency=freq, s=self._to_numpy(S_t)))
        return nets

    # ----------------------------------------------------------------- BaseSurrogate API
    def predict(self, x: Mapping[str, float], *, return_std: bool = False):
        """
        predict(x) — основной контракт BaseSurrogate.

        По умолчанию (output="network") возвращает rf.Network, т.е. совместим
        с Objective/Specification без адаптеров.
        """
        if return_std:
            raise NotImplementedError("CMFilter does not support uncertainty (σ).")

        S_t = self._predict_s_torch(x)

        if self.output == "torch":
            return S_t
        if self.output == "numpy":
            return self._to_numpy(S_t)
        if self.output == "network":
            return self._to_network(S_t)

        raise ValueError(f"Unknown output kind: {self.output!r}")

    def batch_predict(
        self,
        xs: Sequence[Mapping[str, float]],
        *,
        return_std: bool = False,
    ):
        """
        batch_predict(xs) — батчевый API.

        Для output="network" возвращаем list[rf.Network] (как ожидают Objective/Specification).
        Для output="numpy"/"torch" возвращаем массив/тензор формы (B,F,2,2).
        """
        if return_std:
            raise NotImplementedError("CMFilter does not support uncertainty (σ).")

        if not xs:
            if self.output == "network":
                return []
            if self.output == "numpy":
                return np.zeros((0, self.f_grid.size, 2, 2), dtype=np.complex64)
            if self.output == "torch":
                import torch
                return torch.zeros((0, self.f_grid.size, 2, 2), dtype=torch.complex64)
            raise ValueError(f"Unknown output kind: {self.output!r}")

        # 1) network — отдельный быстрый путь без переключения self.output
        if self.output == "network":
            return self.batch_predict_network(xs)

        # 2) numpy/torch — считаем по одному и стекуем (MVP)
        outs = [self._predict_s_torch(x) for x in xs]

        if self.output == "torch":
            import torch
            return torch.stack(outs, dim=0)

        if self.output == "numpy":
            import torch
            S = torch.stack(outs, dim=0)  # (B,F,2,2)
            return S.detach().cpu().numpy()

        raise ValueError(f"Unknown output kind: {self.output!r}")

    # ----------------------------------------------------------------- passes_spec (совместимость с Specification)
    def passes_spec(self, xs, spec):
        """
        Быстрая проверка спецификации на предсказаниях.

        Сейчас (под текущие Selector/Specification) мы *гарантируем* rf.Network:
          - одиночная точка: predict_network(x) -> spec.is_ok(net)
          - батч: batch_predict_network(xs) -> np.fromiter(spec.is_ok(...))

        Позже можно добавить spec.fast_is_ok(S_np) и проверять без rf.Network.
        """
        from collections.abc import Mapping as _Mapping

        # одиночная точка
        if isinstance(xs, _Mapping):
            net = self.predict_network(xs)
            return bool(spec.is_ok(net))

        # батч
        nets = self.batch_predict_network(xs)
        return np.fromiter((spec.is_ok(net) for net in nets), dtype=bool)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CMFilter(order={self.topo.order}, kind={self.meta.kind}, "
            f"include_qu={self.include_qu!r}, output={self.output!r})"
        )


__all__ = ["CMFilter", "CMFilterMeta"]
