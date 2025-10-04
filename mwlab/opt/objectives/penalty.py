#mwlab/opt/objectives/penalty.py
# -*- coding: utf-8 -*-
"""
Penalty & Feasible-Yield objectives for surrogate-based optimization
===================================================================

В модуле реализованы две цели для глобальной оптимизации поверх суррогата:

1) PenaltyObjective
   -----------------
   Минимизация суммарного штрафа `Specification.penalty(net)`.
   Особенности:
     • устойчивость к ошибкам (возврат `large_penalty`);
     • батчевая оценка (через `surrogate.batch_predict`, если доступно);
     • **никакой встроенной поддержки guard-bands**: если требуется
       «ужесточённая» спецификация, подготовьте её снаружи и передайте как `spec`.

2) FeasibleYieldObjective
   -----------------------
   «Составная» цель для одного прогона оптимизатора:

       J(x) = penalty(x) − α · I{spec.is_ok(net)} · Y_local(x)

   где:
     • `penalty(x) = spec.penalty(surrogate.predict(x), reduce)` — неотрицательный штраф;
     • `I{…}` — индикатор выполнимости *в центральной точке* (без MC);
     • `Y_local(x)` — локальный yield, оцененный методом Монте-Карло
       в окрестности `x` (см. ниже).

   Режим — «жёсткий»: слагаемое с `Y_local` участвует **только если**
   центральная точка выполнима (`spec.is_ok(net) == True`). Порогов (`feasible_tol`)
   и «шторок» (сглаживания переключения) **нет**.

Оценка локального yield
-----------------------
* Локальный дизайн-спейс строится **вокруг точки x**:
  - либо напрямую задан через `tolerance_space`;
  - либо собирается из `base_space` + `delta` (симметричные или асимметричные радиусы),
    с интерпретацией `delta` как абсолютной величины (`delta_mode='abs'`) или доли
    от |центра| (`'rel'`).
* Непрерывные переменные: интервалы `[x − δ⁻, x + δ⁺]`, клип к глобальным границам.
* Целочисленные переменные: учитывается шаг `step`; по умолчанию **заморожены**
  (`vary_integers=False`), при разрешении — диапазон ограничивается глобальными границами.
* Ordinal/Categorical: по умолчанию **заморожены** на текущем уровне
  (`vary_categorical=False`); при разрешении — допускается варьирование уровней.
* Выборка MC генерируется сэмплерами MWLab (по умолчанию `'sobol'`) с опциональным
  фиксированным сидом `rng` для детерминизма. Локальные ограничения на точки
  **дополнительно не применяются** (выборка формируется внутри построенного
  локального пространства).

Контракты
---------
* `surrogate` : `mwlab.opt.surrogates.base.BaseSurrogate`
    - обязан реализовывать `predict(x) -> rf.Network` и **желательно**
      `batch_predict(list[dict]) -> list[rf.Network]`.
* `spec` : `mwlab.opt.objectives.specification.Specification`
    - обязан предоставлять `is_ok(net)` и `penalty(net, reduce=...)`.

Зависимости
-----------
* обязательные: `numpy`, `mwlab` (DesignSpace, Samplers, Specification, BaseSurrogate)
* опциональные: `skrf` — только для типа `rf.Network` (используется мягкая проверка).

Примеры
-------
>>> # 1) Минимизация penalty
>>> obj = PenaltyObjective(surrogate=sur, spec=spec)
>>> val = obj({"w": 12e-6, "gap": 40e-6})
>>> arr = obj.batch([{"w": 10e-6}, {"w": 20e-6}])   # np.ndarray формы (B,)

>>> # 2) Составная цель: penalty → затем −α·Y_local (жёсткий режим)
>>> fy = FeasibleYieldObjective(
...     surrogate=sur,
...     spec=spec,
...     base_space=space,                # типы/границы параметров
...     delta=50e-6, delta_mode="abs",   # радиусы локальных допусков
...     n_mc=4096, sampler="sobol", rng=2025,
...     alpha=0.1,
... )
>>> cost = fy({"w": 12e-6, "gap": 40e-6})  # >=0 при infeasible, иначе может быть < 0
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, List, Union, Optional
import warnings
import math

import numpy as np

# skrf — для типа rf.Network (duck-typing допускается даже если пакет не установлен)
try:
    import skrf as rf
    _HAS_SKRF = True
except Exception:  # pragma: no cover
    rf = None      # type: ignore[assignment]
    _HAS_SKRF = False

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.objectives.specification import Specification

# Дизайн-спейс и типы переменных — нужны для сборки локального пространства
from mwlab.opt.design.space import (
    DesignSpace,
    ContinuousVar,
    IntegerVar,
    OrdinalVar,
    CategoricalVar,
)

# Сэмплеры DOE из MWLab (Sobol / Halton / LHS / Normal …)
from mwlab.opt.design.samplers import get_sampler, BaseSampler


# ────────────────────────────────────────────────────────────────────────────
#                                Типы / алиасы
# ────────────────────────────────────────────────────────────────────────────

PointDict = Dict[str, float]
ReduceMode = str  # {'sum','mean','max'}


# ────────────────────────────────────────────────────────────────────────────
#                            Внутренние помощники
# ────────────────────────────────────────────────────────────────────────────

def _is_network(obj: Any) -> bool:
    """
    Мягкая проверка, что surrogate вернул «похожее на rf.Network».
    Если skrf недоступен — используем duck-typing по атрибутам.
    """
    if _HAS_SKRF and isinstance(obj, rf.Network):  # type: ignore[arg-type]
        return True
    # Duck-typing: у rf.Network есть .s, .frequency, .s_db, .s_mag
    return hasattr(obj, "s") and hasattr(obj, "frequency")


def _sigmoid(x: float) -> float:
    """Численно безопасная логистическая σ(x) = 1 / (1 + exp(-x))."""
    # Ограничим диапазон аргумента во избежание overflow:
    if x >= 50.0:
        return 1.0
    if x <= -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ────────────────────────────────────────────────────────────────────────────
#                               PenaltyObjective
# ────────────────────────────────────────────────────────────────────────────

class PenaltyObjective:
    """
    Целевая функция: **минимизировать суммарный штраф** Specification,
    посчитанный на предсказании суррогата.

    Параметры
    ---------
    surrogate : BaseSurrogate
        Модель X→S. Должна поддерживать predict(), желательно batch_predict().
    spec : Specification
        Спецификация (не мутируется; guard-bands не поддерживаются).
    reduce : {'sum','mean','max'}, default='sum'
        Агрегация штрафов по критериям.
    large_penalty : float, default=1e9
        Что возвращать, если surrogate/decoding упал на конкретной точке.
    """

    def __init__(
        self,
        *,
        surrogate: BaseSurrogate,
        spec: Specification,
        reduce: ReduceMode = "sum",
        large_penalty: float = 1e9,
    ):
        self.sur = surrogate
        self.spec = spec
        self.reduce = str(reduce)
        self.large_penalty = float(large_penalty)

    # ---------------------------------------------------------------- single
    def __call__(self, x: Mapping[str, float]) -> float:
        """
        Оценка одной точки `x` → штраф (float). Ошибки локализуются.
        """
        try:
            net = self.sur.predict(x)
            if not _is_network(net):
                raise TypeError("surrogate.predict() вернул объект, не похожий на rf.Network")
            return float(self.spec.penalty(net, reduce=self.reduce))
        except Exception as e:  # pragma: no cover
            warnings.warn(
                f"[PenaltyObjective] ошибка оценки точки {dict(x)}: {e!r} → возвращаю large_penalty",
                RuntimeWarning,
            )
            return self.large_penalty

    # ---------------------------------------------------------------- batch
    def batch(self, xs: Sequence[Mapping[str, float]]) -> np.ndarray:
        """
        Батчевый расчёт штрафов. Возвращает массив формы (B,).
        """
        if not xs:
            return np.zeros((0,), dtype=float)

        # Пытаемся использовать быстрый путь batch_predict
        try:
            nets = self.sur.batch_predict(xs)  # ожидаем список rf.Network
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[PenaltyObjective.batch] batch_predict упал: {e!r}; fallback на построчный режим", RuntimeWarning)
            return np.fromiter((self(x) for x in xs), dtype=float, count=len(xs))

        out = np.empty((len(nets),), dtype=float)
        for i, net in enumerate(nets):
            try:
                if not _is_network(net):
                    raise TypeError("batch_predict: элемент не похож на rf.Network")
                out[i] = float(self.spec.penalty(net, reduce=self.reduce))
            except Exception:
                out[i] = self.large_penalty
        return out

    # ---------------------------------------------------------------- report
    def report(self, x: Mapping[str, float]) -> Dict[str, Any]:
        """
        Подробный отчёт по спецификации.
        Формат совместим со Specification.report().
        """
        try:
            net = self.sur.predict(x)
            if not _is_network(net):
                raise TypeError("surrogate.predict() вернул объект, не похожий на rf.Network")
            rep = self.spec.report(net)
            # добавим верхний уровень удобных полей
            rep["__objective__"] = float(self.spec.penalty(net, reduce=self.reduce))
            rep["__spec_name__"] = str(self.spec.name)
            return rep
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[PenaltyObjective.report] ошибка: {e!r}", RuntimeWarning)
            return {"__objective__": self.large_penalty, "__error__": repr(e)}

    # ---------------------------------------------------------------- vector-API
    def evaluate_vector(self, z: Sequence[float], names: Sequence[str]) -> float:
        """
        Утилита для оптимизаторов: собрать dict из (names,z) и вызвать __call__.
        """
        x = {k: float(v) for k, v in zip(names, z)}
        return self(x)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        return f"PenaltyObjective(reduce={self.reduce})"


# ────────────────────────────────────────────────────────────────────────────
#                           FeasibleYieldObjective
# ────────────────────────────────────────────────────────────────────────────
class FeasibleYieldObjective:
    """
    Составная цель: всегда минимизируем penalty и, ДОПОЛНИТЕЛЬНО,
    если точка выполнима по ТЗ (spec.is_ok(net) == True), поощряем локальный yield:

        J(x) = penalty(x) - alpha * I{is_ok(x)} * Y_local(x)

    Где:
      • penalty(x) = spec.penalty(surrogate.predict(x), reduce)
        — неотрицательный штраф (у softplus-компараторов > 0).
      • I{is_ok(x)} — индикатор выполнимости в центральной точке (без MC).
      • Y_local(x) — доля pass по локальным технологическим допускам вокруг x
        (детерминированный MC c фиксированным сидом).

    ВАЖНО:
      • На infeasible-точках Y не считаем (экономим расчёты).
      • Никаких порогов feasible_tol и «шторок» нет — включение yield строго по is_ok().
      • Значение цели может стать отрицательным, если alpha * Y > penalty — это ок.
        Если используете ранний останов «best ≤ 0», учтите это.

    Параметры
    ---------
    surrogate : BaseSurrogate
        Прямая модель X→S.
    spec : Specification
        Техническое задание (penalty + is_ok()).
    base_space : DesignSpace | None
        Глобальное пространство (тип/границы/шаги). Нужно, если допуски задаются через `delta`.
    delta : float | dict[str, float|(float,float)] | None
        Радиусы локальных допусков для continuous/int.
          - число → одинаковый радиус для всех;
          - dict → по именам; значения: r или (dm, dp) для асимметрии.
        Используется, если `tolerance_space` не задан.
    delta_mode : {'abs','rel'}, default='abs'
        Интерпретация величин `delta`: абсолютная или доля от |центра|.
    tolerance_space : DesignSpace | None
        Готовое пространство допусков; радиусы извлекаются через
        `.to_center_delta(centers=x, sym=False)` и центрируются на x.
    vary_integers : bool, default=False
        Флуктуации целочисленных параметров в локальном MC (по умолчанию — заморожены).
    vary_categorical : bool, default=False
        Флуктуации ordinal/categorical (по умолчанию — заморожены на текущем уровне).
    n_mc : int, default=4096
        Размер выборки Монте-Карло для оценки локального yield.
    sampler : str, default='sobol'
        Имя сэмплера MWLab ('sobol', 'normal', 'lhs', ...).
    sampler_kwargs : dict | None
        Доп. аргументы фабрики сэмплера.
    rng : int | None, default=2025
        Сид генератора (детерминизм решётки MC между вызовами).
    alpha : float, default=0.1
        Вес поощрения yield (масштабируется относительно penalty).
    y_clip_eps : float, default=0.0
        Небольшой клип Y в [y_clip_eps, 1 - y_clip_eps] для численной устойчивости.
    reduce : {'sum','mean','max'}, default='sum'
        Агрегация штрафов в spec.penalty.
    large_penalty : float, default=1e9
        Возврат при сбоях surrogate/decoding.

    Примечания
    ----------
    • Локальные допуски строятся вокруг центральной точки x с учётом типов:
        - continuous: [x - dm, x + dp] ∩ глобальные границы;
        - integer:   учитываем step; при vary_integers=False — фиксируем в x;
        - ordinal/categorical: при vary_categorical=False — фиксируем уровень.
    • Для стабильности MC выборка генерируется одним и тем же сидом `rng` и
      «аффинно» отображается локальным пространством (собирается через DesignSpace).
    """

    def __init__(
        self,
        *,
        surrogate: BaseSurrogate,
        spec: Specification,
        base_space: DesignSpace | None = None,
        delta: float | Dict[str, float | Tuple[float, float]] | None = None,
        delta_mode: str = "abs",
        tolerance_space: DesignSpace | None = None,
        vary_integers: bool = False,
        vary_categorical: bool = False,
        n_mc: int = 4096,
        sampler: str = "sobol",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        rng: Optional[int] = 2025,
        alpha: float = 0.1,
        y_clip_eps: float = 0.0,
        reduce: ReduceMode = "sum",
        large_penalty: float = 1e9,
    ):
        # --- базовые сущности
        self.sur = surrogate
        self.spec = spec

        # --- источники локальных допусков
        self.base_space = base_space
        self.tol_space = tolerance_space
        self.delta = delta
        self.delta_mode = str(delta_mode)

        if self.tol_space is None and (self.base_space is None or self.delta is None):
            raise ValueError(
                "FeasibleYieldObjective: задайте либо tolerance_space, либо (base_space и delta)"
            )

        # --- флаги варьирования типов
        self.vary_integers = bool(vary_integers)
        self.vary_categorical = bool(vary_categorical)

        # --- Монте-Карло-настройки
        self.n_mc = int(n_mc)
        if self.n_mc <= 0:
            raise ValueError("n_mc must be a positive integer")
        self.sampler_name = str(sampler)
        self.sampler_kwargs = dict(sampler_kwargs or {})
        self.rng = None if rng is None else int(rng)

        # --- вес и нормализация yield
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.y_clip_eps = float(max(0.0, y_clip_eps))
        if not (0.0 <= self.y_clip_eps < 0.5):
            raise ValueError("y_clip_eps must be in [0, 0.5)")

        # --- прочее
        self.reduce = str(reduce)
        self.large_penalty = float(large_penalty)

    # ────────────────────────────────────────────────────────────
    # Внутренние строительные блоки
    # ────────────────────────────────────────────────────────────

    def _build_local_space(self, x: Mapping[str, Any]) -> DesignSpace:
        """
        Собирает локальный DesignSpace вокруг точки x, **сохраняя типы переменных**.

        Логика:
          1) Типы и глобальные границы берём из base_space.
          2) Радиусы допусков: из tolerance_space (to_center_delta) ИЛИ из delta (+ delta_mode).
          3) integer: учитываем step и глобальные границы; при vary_integers=False — замораживаем.
          4) ordinal/categorical: при vary_categorical=False — фиксируем текущий уровень.
          5) Всегда клипуем локальные границы глобальными.
        """
        if self.base_space is None:
            raise RuntimeError("base_space обязателен для сборки локального пространства")

        # --- собрать карту асимметричных радиусов dm/dp
        dm_dp_map: Dict[str, Tuple[float, float]] = {}

        if self.tol_space is not None:
            td = self.tol_space.to_center_delta(centers=x, sym=False)  # name -> (-dm, +dp)
            for name, (dm, dp) in td.items():
                dm_dp_map[name] = (abs(float(dm)), abs(float(dp)))
        else:
            for name in self.base_space.names():
                center = float(x[name]) if name in x else 0.0
                d_spec = self.delta[name] if isinstance(self.delta, dict) and name in self.delta else self.delta

                if isinstance(d_spec, tuple):
                    dm, dp = map(float, d_spec)
                    dm, dp = abs(dm), abs(dp)
                else:
                    r = float(d_spec)  # type: ignore[arg-type]
                    dm, dp = r, r

                if self.delta_mode == "rel":
                    scale = abs(center)
                    dm *= scale
                    dp *= scale

                dm_dp_map[name] = (dm, dp)

        # --- построить локальные переменные с сохранением типа
        local_vars: Dict[str, Any] = {}
        for name in self.base_space.names():
            base_var = self.base_space[name]
            if name not in x:
                raise KeyError(f"В точке x отсутствует параметр '{name}'")
            xval = x[name]

            # 1) Categorical/Ordinal
            if getattr(base_var, "levels", None) is not None:
                levels = list(base_var.levels)  # type: ignore[attr-defined]
                if not self.vary_categorical:
                    if xval in levels:
                        lvl = xval
                    elif isinstance(xval, (int, np.integer)) and 0 <= int(xval) < len(levels):
                        lvl = levels[int(xval)]
                    else:
                        raise ValueError(f"Unknown level '{xval}' for '{name}'")
                    cls = CategoricalVar if isinstance(base_var, CategoricalVar) else OrdinalVar
                    local_vars[name] = cls(levels=[lvl])  # type: ignore[call-arg]
                else:
                    cls = CategoricalVar if isinstance(base_var, CategoricalVar) else OrdinalVar
                    local_vars[name] = cls(levels=levels)  # type: ignore[call-arg]
                continue

            # 2) Integer
            if getattr(base_var, "is_integer", False):
                step = int(getattr(base_var, "step", 1))
                lo_g, hi_g = base_var.bounds()
                lo_g, hi_g = int(round(lo_g)), int(round(hi_g))

                if not self.vary_integers:
                    ival = int(round(float(xval)))
                    if step > 1:
                        ival = lo_g + int(round((ival - lo_g) / step)) * step
                    ival = int(np.clip(ival, lo_g, hi_g))
                    local_vars[name] = IntegerVar(lower=ival, upper=ival, step=step, unit=getattr(base_var, "unit", ""))
                else:
                    dm, dp = dm_dp_map[name]
                    c = int(round(float(xval)))
                    lo = int(np.floor(c - dm))
                    hi = int(np.ceil (c + dp))
                    if step > 1:
                        lo = lo_g + int(round((lo - lo_g) / step)) * step
                        hi = lo_g + int(round((hi - lo_g) / step)) * step
                    lo = int(np.clip(lo, lo_g, hi_g))
                    hi = int(np.clip(hi, lo_g, hi_g))
                    if hi < lo:
                        lo = hi = int(np.clip(c, lo_g, hi_g))
                    local_vars[name] = IntegerVar(lower=lo, upper=hi, step=step, unit=getattr(base_var, "unit", ""))
                continue

            # 3) Continuous
            lo_g, hi_g = base_var.bounds()
            dm, dp = dm_dp_map[name]
            c = float(xval)
            lo = float(np.clip(c - dm, lo_g, hi_g))
            hi = float(np.clip(c + dp, lo_g, hi_g))
            if hi < lo:
                lo = hi = float(np.clip(c, lo_g, hi_g))
            local_vars[name] = ContinuousVar(lower=lo, upper=hi, unit=getattr(base_var, "unit", ""))

        return DesignSpace(local_vars)

    def _yield_local(self, x: Mapping[str, Any]) -> float:
        """
        Детерминированная оценка локального yield вокруг x:
          1) строим локальный DesignSpace;
          2) генерируем n_mc точек (одинаковая «решётка» по сидy rng);
          3) surrogate.batch_predict → список сетей;
          4) считаем долю pass по spec.is_ok().
        """
        space = self._build_local_space(x)
        sampler: BaseSampler = get_sampler(self.sampler_name, rng=self.rng, **self.sampler_kwargs)
        pts = space.sample(self.n_mc, sampler=sampler, reject_invalid=False)

        # nets = self.sur.batch_predict(pts)
        # ok = [self.spec.is_ok(net) for net in nets]
        ok = self.sur.passes_spec(pts, self.spec)  # np.bool_
        y = float(np.mean(ok))

        if self.y_clip_eps > 0.0:
            eps = min(0.49, float(self.y_clip_eps))
            y = float(np.clip(y, eps, 1.0 - eps))
        return y

    # ────────────────────────────────────────────────────────────
    # Публичный API
    # ────────────────────────────────────────────────────────────

    def __call__(self, x: Mapping[str, float]) -> float:
        """
        Стоимость одной точки: penalty(x) минус вклад yield, если точка выполнима.
        """
        try:
            net = self.sur.predict(x)
            if not _is_network(net):
                raise TypeError("surrogate.predict() вернул объект, не похожий на rf.Network")

            p = float(self.spec.penalty(net, reduce=self.reduce))
            if not self.spec.is_ok(net):
                return p  # infeasible: без MC

            y = self._yield_local(x)
            return p - self.alpha * y

        except Exception as e:  # pragma: no cover
            warnings.warn(f"[FeasibleYieldObjective] ошибка оценки {dict(x)}: {e!r} → large_penalty", RuntimeWarning)
            return self.large_penalty

    def batch(self, xs: Sequence[Mapping[str, float]]) -> np.ndarray:
        """
        Батч-оценка: penalty считается batched; yield — только для выполнимых.
        Возвращает массив формы (B,).
        """
        if not xs:
            return np.zeros((0,), dtype=float)

        try:
            nets = self.sur.batch_predict(xs)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[FeasibleYieldObjective.batch] batch_predict упал: {e!r}; fallback на поэлементный режим", RuntimeWarning)
            return np.fromiter((self(x) for x in xs), dtype=float, count=len(xs))

        B = len(xs)
        out = np.empty((B,), dtype=float)

        # 1) penalty и is_ok — batched
        feas_mask = np.zeros((B,), dtype=bool)
        pens = np.empty((B,), dtype=float)
        for i, net in enumerate(nets):
            try:
                if not _is_network(net):
                    raise TypeError("batch_predict: элемент не похож на rf.Network")
                pens[i] = float(self.spec.penalty(net, reduce=self.reduce))
                feas_mask[i] = bool(self.spec.is_ok(net))
            except Exception:
                pens[i] = math.inf
                feas_mask[i] = False

        # 2) На infeasible → просто penalty
        out[~feas_mask] = pens[~feas_mask]

        # 3) Для feasible — penalty - alpha * Y_local
        for i, x in enumerate(xs):
            if feas_mask[i]:
                try:
                    y = self._yield_local(x)
                    out[i] = pens[i] - self.alpha * y
                except Exception:
                    out[i] = self.large_penalty

        return out

    def report(self, x: Mapping[str, float]) -> Dict[str, Any]:
        """
        Развёрнутый отчёт:
          {
            "__objective__": cost,
            "__penalty__" : penalty,
            "__feasible__": bool,
            "__yield__"   : y | None,    # None если infeasible
            "__alpha__"   : alpha,
          }
        """
        rep: Dict[str, Any] = {"__alpha__": self.alpha}
        try:
            net = self.sur.predict(x)
            if not _is_network(net):
                raise TypeError("surrogate.predict() вернул объект, не похожий на rf.Network")
            p = float(self.spec.penalty(net, reduce=self.reduce))
            feas = bool(self.spec.is_ok(net))

            if not feas:
                rep.update({"__objective__": p, "__penalty__": p, "__feasible__": False, "__yield__": None})
                return rep

            y = self._yield_local(x)
            cost = p - self.alpha * y
            rep.update({"__objective__": float(cost), "__penalty__": p, "__feasible__": True, "__yield__": float(y)})
            return rep

        except Exception as e:  # pragma: no cover
            warnings.warn(f"[FeasibleYieldObjective.report] ошибка: {e!r}", RuntimeWarning)
            rep.update({"__objective__": self.large_penalty, "__error__": repr(e)})
            return rep

    def evaluate_vector(self, z: Sequence[float], names: Sequence[str]) -> float:
        """Собрать dict из (names,z) и вызвать __call__ (удобно для оптимизаторов)."""
        x = {k: float(v) for k, v in zip(names, z)}
        return self(x)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"FeasibleYieldObjective(alpha={self.alpha}, sampler={self.sampler_name}, "
                f"n_mc={self.n_mc}, reduce={self.reduce})")



# ────────────────────────────────────────────────────────────────────────────
#                                  __all__
# ────────────────────────────────────────────────────────────────────────────

__all__ = [
    "PenaltyObjective",
    "FeasibleYieldObjective",
]
