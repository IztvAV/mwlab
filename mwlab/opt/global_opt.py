# mwlab/opt/global_opt.py
# -*- coding: utf-8 -*-
"""
Оркестратор глобальной оптимизации поверх суррогата (Nevergrad)
================================================================

Назначение
----------
Дать единый «взрослый» вход для запуска глобальной оптимизации поверх
суррогатной модели, сохраняя слабую связность модулей:

* Сборка surrogate (из Lightning-модуля или готового BaseSurrogate);
* Формирование цели:
    - 'penalty'         → PenaltyObjective (минимизация суммарного штрафа);
    - 'feasible_yield'  → FeasibleYieldObjective (penalty − α·Y_local для feasible-точек);
      (также допускается кастомная objective_fn);
* Запуск оптимизации (Nevergrad через NGOptimizer);
* Пост-обработка результатов: отчёт по лучшему, топ-K, (опц.) отчётная оценка yield;
* Удобный результат с историей и «снимком» конфигурации прогона.

Ключевые моменты
----------------
* Полностью **исключена** поддержка guard-bands.
* Полностью **убрана** поддержка ранней остановки.
* Для режима 'feasible_yield' используются локальные допуски вокруг точки (MC),
  включение yield-члена — строго по `spec.is_ok()`.

Зависимости
-----------
* Ваши модули:
  - `mwlab.opt.objectives.penalty.PenaltyObjective, FeasibleYieldObjective`
  - `mwlab.opt.nevergrad_runner.NGOptimizer, NGResult`
* MWLab:
  - `mwlab.opt.design.space.DesignSpace`
  - `mwlab.opt.objectives.specification.Specification`
  - `mwlab.opt.surrogates.base.BaseSurrogate`
  - `mwlab.opt.surrogates.nn.NNSurrogate` (если передаёте Lightning-модуль)
  - (опц.) `mwlab.opt.objectives.yield_max.YieldObjective` — для *отчётной* оценки yield на топ-K

Пример
------
>>> from mwlab.opt.design.space import DesignSpace, ContinuousVar
>>> from mwlab.opt.objectives.specification import Specification
>>> from mwlab.opt.global_opt import GlobalOptConfig, GlobalOptimizer
>>>
>>> space = DesignSpace({"w": ContinuousVar(-1e-4, 1e-4), "gap": ContinuousVar(-8e-5, 8e-5)})
>>> spec  = ...  # соберите из YAML или кода
>>> pl_module = ...  # обученный Lightning-регрессор (swap_xy=False)
>>>
>>> # A) Режим 'penalty'
>>> cfg = GlobalOptConfig(
...     space=space, spec=spec, pl_module=pl_module,
...     objective_mode="penalty",
...     algo="NGOpt", budget=5000, population=64, num_workers=8,
...     seed=2025, log_every=10, topk=20, compute_yield=False,
... )
>>> res = GlobalOptimizer(cfg).run()
>>> print(res.best_value, res.best_params)
>>>
>>> # B) Режим 'feasible_yield' (локальные допуски ±50 мкм, α=0.2)
>>> cfg2 = GlobalOptConfig(
...     space=space, spec=spec, pl_module=pl_module,
...     objective_mode="feasible_yield",
...     fy_delta=50e-6, fy_delta_mode="abs",
...     fy_n_mc=4096, fy_sampler="sobol", fy_rng=2025,
...     fy_alpha=0.2,  # вес поощрения yield
...     topk=20, compute_yield=True,
... )
>>> res2 = GlobalOptimizer(cfg2).run()
>>> print(res2.best_value, res2.best_params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Callable

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Внешние типы/классы из MWLab
# ────────────────────────────────────────────────────────────────────────────

from mwlab.opt.design.space import DesignSpace
from mwlab.opt.objectives.specification import Specification
from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.samplers import get_sampler
from mwlab.opt.design.space import ContinuousVar, IntegerVar, OrdinalVar, CategoricalVar


# Обёртка над Lightning-модулем (если surrogate ещё не построен)
try:  # pragma: no cover
    from mwlab.opt.surrogates.nn import NNSurrogate
    _HAS_NN_SUR = True
except Exception:  # pragma: no cover
    NNSurrogate = None  # type: ignore[assignment]
    _HAS_NN_SUR = False

# (опционально) для отчётов S-параметров
try:  # pragma: no cover
    import skrf as rf
    _HAS_SKRF = True
except Exception:  # pragma: no cover
    rf = None  # type: ignore[assignment]
    _HAS_SKRF = False

# (опц.) отчётная оценка yield (НЕ для оптимизации)
try:  # pragma: no cover
    from mwlab.opt.objectives.yield_max import YieldObjective
    _HAS_YIELD = True
except Exception:  # pragma: no cover
    YieldObjective = None  # type: ignore[assignment]
    _HAS_YIELD = False

# ────────────────────────────────────────────────────────────────────────────
# Ваши цели и раннер
# ────────────────────────────────────────────────────────────────────────────

from mwlab.opt.objectives.penalty import PenaltyObjective, FeasibleYieldObjective
from mwlab.opt.nevergrad_runner import NGOptimizer, NGResult


# ────────────────────────────────────────────────────────────────────────────
#                             Конфигурация прогона
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalOptConfig:
    """
    Все «ручки» оркестратора в одном месте.

    Минимальный набор: `space`, `spec` и **либо** `surrogate`, **либо** `pl_module`.
    Если хотите оптимизировать не penalty/feasible_yield, передайте `objective_fn`.
    """
    # Базовые сущности
    space: DesignSpace
    spec: Optional[Specification] = None
    surrogate: Optional[BaseSurrogate] = None           # уже готовый surrogate X→S
    pl_module: Any = None                                # обученный Lightning-модуль (swap_xy=False)

    # Режим цели: 'penalty' | 'feasible_yield'
    objective_mode: str = "penalty"

    # Кастомная цель (если задана — имеет приоритет над objective_mode)
    objective_fn: Optional[Callable[[Mapping[str, Any]], float]] = None

    # Nevergrad (см. NGOptimizer)
    algo: str = "NGOpt"
    budget: int = 2000
    population: Optional[int] = None
    num_workers: int = 1
    seed: Optional[int] = None
    log_every: Optional[int] = None
    show_progress: bool = True

    # Отбор и отчётность
    topk: int = 10

    # Warm-start (бэйзлайн-оценки для отчёта; на NG не влияют)
    warm_start: Optional[Sequence[Mapping[str, Any]]] = None

    # (опц.) отчётная оценка yield на топ-K (быстрое Monte-Carло по surrogate)
    compute_yield: bool = False
    yield_n_mc: int = 8192
    yield_sampler: str = "normal"  # "normal" | "sobol" | ...
    yield_k: int = 5               # на скольких лучших считать yield

    # Параметры режима 'feasible_yield'
    # Если используете 'penalty', эти поля игнорируются.
    fy_base_space: Optional[DesignSpace] = None     # если None → будет использован cfg.space
    fy_delta: Optional[Union[float, Dict[str, Union[float, Tuple[float, float]]]]] = None
    fy_delta_mode: str = "abs"                      # 'abs' | 'rel'
    fy_tolerance_space: Optional[DesignSpace] = None
    fy_vary_integers: bool = False
    fy_vary_categorical: bool = False
    fy_n_mc: int = 4096
    fy_sampler: str = "sobol"
    fy_sampler_kwargs: Optional[Dict[str, Any]] = None
    fy_rng: Optional[int] = 2025
    fy_alpha: float = 0.1            # вес поощрения yield
    fy_y_clip_eps: float = 0.0       # небольшой клип Y для устойчивости (0..0.49)


# ────────────────────────────────────────────────────────────────────────────
#                             Результат оркестрации
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalOptResult:
    """
    Сводка одного прогона.
    """
    # Лучшее решение (по цели, которая оптимизировалась)
    best_params: Dict[str, Any]
    best_value: float

    # (опц.) предсказанная сеть для best (если был surrogate)
    best_net: Optional["rf.Network"]

    # Отчёт спецификации для best (если spec/суррогат заданы)
    spec_report: Optional[Dict[str, Any]]

    # «Сырой» результат раннера
    ng: NGResult

    # Топ-K по оптимизируемой цели (что возвращает раннер)
    top_k: List[Dict[str, Any]]  # [{..., "__value__": float}, ...]

    # (опц.) yield по top-K (если включено compute_yield)
    yield_summary: Optional[List[Dict[str, Any]]]

    # Полезные поля конфигурации (для логов/репродукции)
    cfg_snapshot: Dict[str, Any]


# ────────────────────────────────────────────────────────────────────────────
#                                   Оркестратор
# ────────────────────────────────────────────────────────────────────────────

class GlobalOptimizer:
    """
    Высокоуровневая «склейка»:
        (space, spec, surrogate|pl_module, objective_mode|objective_fn)
        → NGOptimizer.run()
        → отчёты и доп. метрики (yield по топ-K — опционально).
    """

    def __init__(self, cfg: GlobalOptConfig):
        self.cfg = cfg

        # --- 1) подготовка surrogate (если не передан готовый) ----------------
        self.surrogate: Optional[BaseSurrogate] = cfg.surrogate
        if self.surrogate is None and cfg.pl_module is not None:
            if not _HAS_NN_SUR:
                raise RuntimeError("NNSurrogate недоступен (импорт не удался).")
            # ТРЕБОВАНИЕ: Lightning-модуль должен быть «прямым» (swap_xy=False)
            if getattr(cfg.pl_module, "swap_xy", False):
                raise ValueError("Ожидается прямая модель (swap_xy=False) для surrogate X→S.")
            self.surrogate = NNSurrogate(pl_module=cfg.pl_module, design_space=cfg.space)
        # Если surrogate всё ещё None — возможно, будет передана кастом-цель objective_fn.
        # Это валидный сценарий: тогда отчёты по spec/сетям могут быть недоступны.

        # --- 2) выбор/сборка цели --------------------------------------------
        self.objective = self._make_objective_callable()

        # --- 3) раннер Nevergrad ----------------------------------------------
        self.ng = NGOptimizer(
            algo=cfg.algo,
            budget=cfg.budget,
            population=cfg.population,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            log_every=cfg.log_every,
            show_progress=cfg.show_progress,
        )

        # --- 4) warm-start бэйзлайн-оценки (для отчёта; NG мы ими не «кормим») --
        self._warm_start_evals: List[Dict[str, Any]] = []
        if cfg.warm_start:
            for x in cfg.warm_start:
                try:
                    self._warm_start_evals.append({"x": dict(x), "value": float(self.objective(x))})
                except Exception:
                    self._warm_start_evals.append({"x": dict(x), "value": float("inf")})

    # ------------------------------------------------------------------ цель по умолчанию

    def _make_objective_callable(self) -> Callable[[Mapping[str, Any]], float]:
        """
        Логика выбора цели:

        1) Если передана `objective_fn` — используем её как есть.
        2) Иначе собираем цель согласно `objective_mode`:
           - 'penalty'        → PenaltyObjective(surrogate, spec)
           - 'feasible_yield' → FeasibleYieldObjective(...)

        Для 'penalty' и 'feasible_yield' требуется surrogate и spec.
        Для 'feasible_yield' обязательно задать локальные допуски (fy_delta
        или fy_tolerance_space). Базовое пространство берём из fy_base_space
        или cfg.space.
        """
        if self.cfg.objective_fn is not None:
            return self.cfg.objective_fn

        mode = str(self.cfg.objective_mode).lower()
        if mode not in ("penalty", "feasible_yield"):
            raise ValueError("objective_mode должен быть 'penalty' или 'feasible_yield'")

        if self.surrogate is None or self.cfg.spec is None:
            raise ValueError(
                "Нужно либо задать objective_fn, либо передать surrogate + spec "
                "для сборки встроенной цели."
            )

        if mode == "penalty":
            return PenaltyObjective(
                surrogate=self.surrogate,
                spec=self.cfg.spec,
                reduction="sum",
                large_penalty=1e12,
            )

        # mode == "feasible_yield"
        base_space = self.cfg.fy_base_space or self.cfg.space
        if self.cfg.fy_tolerance_space is None and self.cfg.fy_delta is None:
            raise ValueError(
                "Для 'feasible_yield' укажите fy_tolerance_space или fy_delta (+ fy_delta_mode)."
            )

        return FeasibleYieldObjective(
            surrogate=self.surrogate,
            spec=self.cfg.spec,
            base_space=base_space,
            delta=self.cfg.fy_delta,
            delta_mode=self.cfg.fy_delta_mode,
            tolerance_space=self.cfg.fy_tolerance_space,
            vary_integers=self.cfg.fy_vary_integers,
            vary_categorical=self.cfg.fy_vary_categorical,
            n_mc=self.cfg.fy_n_mc,
            sampler=self.cfg.fy_sampler,
            sampler_kwargs=(self.cfg.fy_sampler_kwargs or {}),
            rng=self.cfg.fy_rng,
            alpha=self.cfg.fy_alpha,
            y_clip_eps=self.cfg.fy_y_clip_eps,
            reduction="sum",
            large_penalty=1e12,
        )

    # --------------------------------------------------------------------------- public API

    def run(self) -> GlobalOptResult:
        """
        Запускает оптимизацию, формирует отчёты и возвращает `GlobalOptResult`.
        """
        # === 1) Nevergrad ===
        ng_res: NGResult = self.ng.run(
            space=self.cfg.space,
            objective=self.objective,
            topk=self.cfg.topk,
        )

        # === 2) Лучшее решение и сеть (если есть surrogate) ===
        best_x = dict(ng_res.best_x)
        best_v = float(ng_res.best_value)

        best_net = None
        if self.surrogate is not None and _HAS_SKRF:
            try:
                best_net = self.surrogate.predict(best_x)  # type: ignore[assignment]
            except Exception:
                best_net = None

        # === 3) Отчёт спецификации (если есть spec и сеть) ===
        spec_report = None
        if best_net is not None and self.cfg.spec is not None:
            try:
                spec_report = self.cfg.spec.report(best_net)
            except Exception:
                spec_report = None

        # === 4) Топ-K — как вернул раннер ===
        top_k = list(ng_res.topk)

        # === 5) (опц.) отчётная оценка yield для первых K кандидатов ===
        yield_summary: Optional[List[Dict[str, Any]]] = None
        if self.cfg.compute_yield and _HAS_YIELD and self.cfg.spec is not None and self.surrogate is not None:
            yield_summary = self._compute_yield_for_top(top_k)

        # === 6) Снимок конфигурации ===
        snapshot = dict(
            algo=self.cfg.algo,
            budget=self.cfg.budget,
            population=self.cfg.population,
            num_workers=self.cfg.num_workers,
            seed=self.cfg.seed,
            objective=("custom" if self.cfg.objective_fn is not None else self.cfg.objective_mode),
            topk=self.cfg.topk,
            show_progress=self.cfg.show_progress,
        )
        if (self.cfg.objective_fn is None) and (str(self.cfg.objective_mode).lower() == "feasible_yield"):
            snapshot.update(dict(
                fy_delta=self.cfg.fy_delta,
                fy_delta_mode=self.cfg.fy_delta_mode,
                fy_n_mc=self.cfg.fy_n_mc,
                fy_sampler=self.cfg.fy_sampler,
                fy_rng=self.cfg.fy_rng,
                fy_alpha=self.cfg.fy_alpha,
                fy_y_clip_eps=self.cfg.fy_y_clip_eps,
                fy_vary_integers=self.cfg.fy_vary_integers,
                fy_vary_categorical=self.cfg.fy_vary_categorical,
            ))

        return GlobalOptResult(
            best_params=best_x,
            best_value=best_v,
            best_net=best_net,
            spec_report=spec_report,
            ng=ng_res,
            top_k=top_k,
            yield_summary=yield_summary,
            cfg_snapshot=snapshot,
        )

    # --------------------------------------------------------------------------- helpers

    # ───────────────────────────────────────────────────────────────────
    # Вспомогательный: построить ЛОКАЛЬНЫЙ DesignSpace вокруг x
    # ───────────────────────────────────────────────────────────────────
    def _build_local_space_for_report(self, x: Mapping[str, Any]) -> DesignSpace:
        """
        Строит локальный DesignSpace вокруг точки x, используя cfg.fy_* настройки.
        Если локальные допуски не заданы (ни fy_delta, ни fy_tolerance_space),
        возвращает None -> сигнализируя, что нужно fallback на глобальный space.
        """
        base_space = self.cfg.fy_base_space or self.cfg.space
        # нет локальных допусков → пусть вызывающий решит, что делать (глобальный fallback)
        if self.cfg.fy_tolerance_space is None and self.cfg.fy_delta is None:
            return None  # type: ignore[return-value]

        # 1) собрать карту радиусов dm, dp для каждого параметра
        dm_dp_map: Dict[str, Tuple[float, float]] = {}
        if self.cfg.fy_tolerance_space is not None:
            # асимметричные радиусы из tolerance_space, центрированные на x
            td = self.cfg.fy_tolerance_space.to_center_delta(centers=x, sym=False)  # {name: (-dm, +dp)}
            for name, (dm, dp) in td.items():
                dm_dp_map[name] = (abs(float(dm)), abs(float(dp)))
        else:
            delta = self.cfg.fy_delta
            mode = str(self.cfg.fy_delta_mode).lower()
            for name in base_space.names():
                center = float(x[name]) if name in x else 0.0
                d_spec = delta[name] if isinstance(delta, dict) and name in delta else delta
                if isinstance(d_spec, tuple):
                    dm, dp = map(float, d_spec)
                    dm, dp = abs(dm), abs(dp)
                else:
                    r = float(d_spec)  # type: ignore[arg-type]
                    dm, dp = r, r
                if mode == "rel":
                    # относительный режим: масштабируем на |center|
                    scale = abs(center)
                    dm *= scale
                    dp *= scale
                dm_dp_map[name] = (dm, dp)

        # 2) построить локальные переменные, сохранив типы
        local_vars: Dict[str, Any] = {}
        for name in base_space.names():
            base_var = base_space[name]
            if name not in x:
                raise KeyError(f"В точке x отсутствует параметр '{name}'")
            xval = x[name]

            # Categorical/Ordinal
            if getattr(base_var, "levels", None) is not None:
                levels = list(base_var.levels)  # type: ignore[attr-defined]
                if not self.cfg.fy_vary_categorical:
                    # заморозка уровня
                    xstr = xval if xval in levels else str(xval)
                    cls = CategoricalVar if isinstance(base_var, CategoricalVar) else OrdinalVar
                    local_vars[name] = cls(levels=[xstr])  # type: ignore[call-arg]
                else:
                    cls = CategoricalVar if isinstance(base_var, CategoricalVar) else OrdinalVar
                    local_vars[name] = cls(levels=levels)  # type: ignore[call-arg]
                continue

            # Integer
            if getattr(base_var, "is_integer", False):
                step = int(getattr(base_var, "step", 1))
                lo_g, hi_g = base_var.bounds()
                lo_g, hi_g = int(round(lo_g)), int(round(hi_g))
                if not self.cfg.fy_vary_integers:
                    ival = int(round(float(xval)))
                    if step > 1:
                        ival = int(round(ival / step) * step)
                    ival = int(np.clip(ival, lo_g, hi_g))
                    local_vars[name] = IntegerVar(lower=ival, upper=ival, step=step, unit=getattr(base_var, "unit", ""))
                else:
                    dm, dp = dm_dp_map[name]
                    c = int(round(float(xval)))
                    lo = int(np.floor(c - dm))
                    hi = int(np.ceil(c + dp))
                    if step > 1:
                        lo = int(round(lo / step) * step)
                        hi = int(round(hi / step) * step)
                    lo = int(np.clip(lo, lo_g, hi_g))
                    hi = int(np.clip(hi, lo_g, hi_g))
                    if hi < lo:
                        lo = hi = int(np.clip(c, lo_g, hi_g))
                    local_vars[name] = IntegerVar(lower=lo, upper=hi, step=step, unit=getattr(base_var, "unit", ""))
                continue

            # Continuous
            lo_g, hi_g = base_var.bounds()
            dm, dp = dm_dp_map[name]
            c = float(xval)
            lo = float(np.clip(c - dm, lo_g, hi_g))
            hi = float(np.clip(c + dp, lo_g, hi_g))
            if hi < lo:
                lo = hi = float(np.clip(c, lo_g, hi_g))
            local_vars[name] = ContinuousVar(lower=lo, upper=hi, unit=getattr(base_var, "unit", ""))

        return DesignSpace(local_vars)

    # ───────────────────────────────────────────────────────────────────
    # Отчётный yield по top-K: ЛОКАЛЬНО (если заданы fy_*) или глобально
    # ───────────────────────────────────────────────────────────────────
    def _compute_yield_for_top(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """
        Считает yield для первых K кандидатов.
        Если задан хотя бы один из локальных параметров (fy_delta или fy_tolerance_space),
        то для КАЖДОГО кандидата строится локальный DesignSpace вокруг его x
        и оценивается локальный yield. Иначе — fallback на глобальный cfg.space.
        """
        assert self.cfg.spec is not None and self.surrogate is not None
        k = max(1, int(self.cfg.yield_k))
        n_mc = int(self.cfg.yield_n_mc)
        sampler_name = str(self.cfg.yield_sampler)
        # сид для детерминизма (тот же принцип, что и в FeasibleYieldObjective)
        rng = int(self.cfg.fy_rng) if self.cfg.fy_rng is not None else (
            int(self.cfg.seed) if self.cfg.seed is not None else None)

        out: List[Dict[str, Any]] = []
        for i, rec in enumerate(rows[:k]):
            # отделяем параметры от служебного поля __value__
            x = {k_: v_ for k_, v_ in rec.items() if k_ != "__value__"}

            # 1) локальное или глобальное пространство
            local_space = None
            try:
                local_space = self._build_local_space_for_report(x)
            except Exception as e:
                # если не получилось построить локальное — упадём на глобальное и логнём
                print(f"[yield] Local space build failed for idx={i}: {type(e).__name__}: {e}")
                local_space = None

            space_to_use = local_space if local_space is not None else self.cfg.space

            # 2) сэмплы (устойчиво к форме)
            try:
                sampler = get_sampler(sampler_name, rng=rng)
                pts = space_to_use.sample(n_mc, sampler=sampler, reject_invalid=False)
                names = list(space_to_use.names())

                if isinstance(pts, list) and (len(pts) == 0 or isinstance(pts[0], Mapping)):
                    xs = [dict(p) for p in pts]  # список словарей
                else:
                    arr = np.asarray(pts, dtype=float)
                    if arr.ndim != 2:
                        raise RuntimeError(f"DesignSpace.sample вернул неожиданную форму: {arr.shape}")
                    B, D = arr.shape
                    if D == len(names):
                        xs = [dict(zip(names, arr[j].tolist())) for j in range(B)]
                    elif B == len(names):
                        xs = [dict(zip(names, arr[:, j].tolist())) for j in range(D)]
                    else:
                        raise RuntimeError(
                            f"Не удаётся сопоставить семплы с параметрами: shape={arr.shape}, D={len(names)}")

                # 3) батч-прогон и подсчёт pass
                nets = self.surrogate.batch_predict(xs)
                ok = [self.cfg.spec.is_ok(net) for net in nets]
                y = float(np.mean(ok))
            except Exception as e:
                print(f"[yield] Compute failed for idx={i}: {type(e).__name__}: {e}")
                y = float("nan")

            out.append({"idx": i, "yield": y, "n_mc": n_mc, "sampler": sampler_name,
                        "mode": ("local" if local_space is not None else "global")})
        return out

    # --------------------------------------------------------------------------- conveniences

    @staticmethod
    def center_point(space: DesignSpace) -> Dict[str, Any]:
        """
        Быстрый «центр» пространства:
          • для Continuous/Integer — (lo+hi)/2 (с посадкой на сетку `step`);
          • для Ordinal/Categorical — средний уровень по индексу.
        Удобно использовать для `warm_start`.
        """
        out: Dict[str, Any] = {}
        for n in space.names():
            var = space[n]
            lo, hi = var.bounds()
            if getattr(var, "levels", None) is not None:
                levels = list(var.levels)  # type: ignore[attr-defined]
                idx = int(np.clip(round(0.5 * (len(levels) - 1)), 0, len(levels) - 1))
                out[n] = levels[idx]
            else:
                val = 0.5 * (lo + hi)
                if getattr(var, "is_integer", False):
                    step = int(getattr(var, "step", 1))
                    ival = int(round(val))
                    if step > 1:
                        ival = int(round(ival / step) * step)
                    out[n] = ival
                else:
                    out[n] = float(val)
        return out

    def rank_candidates(self, points: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ранжирует произвольный пул `points` по **текущей цели оркестратора**
        (penalty или feasible_yield, либо custom objective).
        Возвращает список dict с полем `__value__`, отсортированный по возрастанию.
        """
        out: List[Dict[str, Any]] = []
        for p in points:
            try:
                v = float(self.objective(p))
            except Exception:
                v = float("inf")
            row = dict(p)
            row["__value__"] = v
            out.append(row)
        return sorted(out, key=lambda r: float(r["__value__"]))
