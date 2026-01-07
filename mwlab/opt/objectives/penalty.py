#mwlab/opt/objectives/penalty.py
# -*- coding: utf-8 -*-
"""
mwlab.opt.objectives.penalty
===========================

Penalty & Feasible-Yield objectives for surrogate-based optimization
-------------------------------------------------------------------

В модуле реализованы две цели для глобальной оптимизации поверх суррогата:

1) PenaltyObjective
   ----------------
   Минимизация суммарного штрафа `Specification.penalty(net)`.
   Особенности:
     • устойчивость к ошибкам (возврат `large_penalty`);
     • батчевая оценка (через `surrogate.batch_predict`, если доступно);
     • никакой встроенной поддержки guard-bands:
       если требуется «ужесточённая» спецификация, подготовьте её снаружи
       и передайте как `spec`.

2) FeasibleYieldObjective
   ----------------------
   «Составная» цель для одного прогона оптимизатора:

       J(x) = penalty(x) − α · I{spec.is_ok(net)} · Y_local(x)

   где:
     • penalty(x) = spec.penalty(net, reduction) — неотрицательный штраф;
     • I{…} — индикатор выполнимости в центральной точке (без MC);
     • Y_local(x) — локальный yield, оцененный методом Монте-Карло
       в окрестности x.

   Режим — «жёсткий»: Y_local учитывается только если центральная точка
   выполнима (is_ok == True). Порогов feasible_tol/«шторок» нет.

Контракт на тип сети
--------------------
Объект, возвращаемый surrogate.predict/batch_predict, должен быть совместим
с mwlab.opt.objectives.network_like.NetworkLike (duck-typing):
  - net.frequency.f (частоты в Гц),
  - net.s (комплексные S-параметры),
  - net.s_mag (|S|),
  - net.s_db (20*log10|S|).

skrf НЕ требуется и не импортируется.

Производительность
------------------
Критический момент: выполнять spec.penalty(net) и затем spec.is_ok(net)
— значит прогонять все критерии ДВАЖДЫ. В FeasibleYieldObjective это устранено:
мы используем один проход через spec.evaluate(net) и из результатов берём:
  - penalty (свёртка weighted_penalty),
  - all_ok (AND по r.ok).

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

from typing import Any, Dict, Mapping, Sequence, Tuple, Optional, Literal
import warnings
import math

import numpy as np

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.objectives.specification import Specification
from mwlab.opt.objectives.network_like import NetworkLike

from mwlab.opt.design.space import (
    DesignSpace,
    ContinuousVar,
    IntegerVar,
    OrdinalVar,
    CategoricalVar,
)
from mwlab.opt.design.samplers import get_sampler, BaseSampler


# ────────────────────────────────────────────────────────────────────────────
#                                Типы / алиасы
# ────────────────────────────────────────────────────────────────────────────

ReductionMode = Literal["sum", "mean", "max"]


# ────────────────────────────────────────────────────────────────────────────
#                        Внутренние проверки и утилиты
# ────────────────────────────────────────────────────────────────────────────

def _normalize_reduction(reduction: str) -> ReductionMode:
    """
    Приводим reduction к каноническому виду и валидируем.

    Делаем это здесь, чтобы:
      - ошибки были раньше и понятнее,
      - не зависеть от того, как именно проверяет Specification.
    """
    r = str(reduction).strip().lower()
    if r not in ("sum", "mean", "max"):
        raise ValueError("reduction must be one of: 'sum', 'mean', 'max'")
    return r  # type: ignore[return-value]


def _is_networklike(obj: Any) -> bool:
    """
    Мягкая runtime-проверка совместимости с NetworkLike.

    ВАЖНО:
    - Мы НЕ делаем isinstance(..., NetworkLike), т.к. NetworkLike может быть Protocol.
    - Мы НЕ вызываем вычислительно дорогие методы, только проверяем атрибуты.
    """
    freq = getattr(obj, "frequency", None)
    if freq is None or not hasattr(freq, "f"):
        return False

    # Минимально необходимые свойства для селекторов:
    # - selectors.SComplexSelector/PhaseSelector используют net.s
    # - selectors.SMagSelector использует net.s_mag / net.s_db
    if not hasattr(obj, "s"):
        return False
    if not hasattr(obj, "s_mag") or not hasattr(obj, "s_db"):
        return False

    return True


def _reduce_penalties(values: np.ndarray, reduction: ReductionMode) -> float:
    """
    Свёртка массива штрафов (обычно weighted_penalty по критериям).
    """
    if values.size == 0:
        return 0.0
    if reduction == "sum":
        return float(np.sum(values))
    if reduction == "mean":
        return float(np.mean(values))
    # reduction == "max"
    return float(np.max(values))


def _penalty_and_all_ok_via_evaluate(
    spec: Specification,
    net: NetworkLike,
    reduction: ReductionMode,
) -> Tuple[float, bool]:
    """
    Получить (penalty, all_ok) за ОДИН прогон спецификации.

    Почему так:
      - spec.penalty(net) обычно делает evaluate(net);
      - spec.is_ok(net) тоже делает evaluate(net);
      => двойной расчёт. Здесь мы избегаем этого.

    penalty считаем по weighted_penalty, чтобы совпадать со Specification.penalty().
    """
    results = spec.evaluate(net)
    all_ok = bool(all(r.ok for r in results))
    penalties = np.asarray([r.weighted_penalty for r in results], dtype=float)
    p = _reduce_penalties(penalties, reduction)
    return p, all_ok


def _align_down(v: int, origin: int, step: int) -> int:
    """
    Выравнивание вниз по сетке origin + k*step.
    Для lower-bound это корректнее, чем round().
    """
    if step <= 1:
        return v
    return origin + int(math.floor((v - origin) / step)) * step


def _align_up(v: int, origin: int, step: int) -> int:
    """
    Выравнивание вверх по сетке origin + k*step.
    Для upper-bound это корректнее, чем round().
    """
    if step <= 1:
        return v
    return origin + int(math.ceil((v - origin) / step)) * step


# ────────────────────────────────────────────────────────────────────────────
#                               PenaltyObjective
# ────────────────────────────────────────────────────────────────────────────

class PenaltyObjective:
    """
    Целевая функция: минимизировать суммарный штраф Specification,
    посчитанный на предсказании суррогата.

    Параметры
    ---------
    surrogate : BaseSurrogate
        Модель X→S. Должна поддерживать predict(), желательно batch_predict().
    spec : Specification
        Спецификация (guard-bands не поддерживаются).
    reduction : {'sum','mean','max'}, default='sum'
        Агрегация штрафов по критериям.
    large_penalty : float, default=1e9
        Что возвращать, если surrogate/декодинг/оценка упали на конкретной точке.
    """

    def __init__(
        self,
        *,
        surrogate: BaseSurrogate,
        spec: Specification,
        reduction: str = "sum",
        large_penalty: float = 1e9,
    ):
        self.sur = surrogate
        self.spec = spec
        self.reduction: ReductionMode = _normalize_reduction(reduction)
        self.large_penalty = float(large_penalty)

    # ---------------------------------------------------------------- single
    def __call__(self, x: Mapping[str, float]) -> float:
        """
        Оценка одной точки `x` → штраф (float). Ошибки локализуются.
        """
        try:
            net = self.sur.predict(x)
            if not _is_networklike(net):
                raise TypeError("surrogate.predict() вернул объект, не совместимый с NetworkLike")
            return float(self.spec.penalty(net, reduction=self.reduction))
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
            nets = self.sur.batch_predict(xs)
        except Exception as e:  # pragma: no cover
            warnings.warn(
                f"[PenaltyObjective.batch] batch_predict упал: {e!r}; fallback на построчный режим",
                RuntimeWarning,
            )
            return np.fromiter((self(x) for x in xs), dtype=float, count=len(xs))

        if len(nets) != len(xs):  # защитимся от некорректной реализации surrogate
            warnings.warn(
                "[PenaltyObjective.batch] batch_predict вернул список другой длины; fallback на построчный режим",
                RuntimeWarning,
            )
            return np.fromiter((self(x) for x in xs), dtype=float, count=len(xs))

        out = np.empty((len(nets),), dtype=float)
        for i, net in enumerate(nets):
            try:
                if not _is_networklike(net):
                    raise TypeError("batch_predict: элемент не совместим с NetworkLike")
                out[i] = float(self.spec.penalty(net, reduction=self.reduction))
            except Exception:  # pragma: no cover
                out[i] = self.large_penalty
        return out

    # ---------------------------------------------------------------- report
    def report(self, x: Mapping[str, float]) -> Dict[str, Any]:
        """
        Подробный отчёт по спецификации.

        ВАЖНО: не пересчитываем penalty отдельно, чтобы не дублировать работу.
        Specification.report(...) уже делает evaluate(...) один раз.
        """
        try:
            net = self.sur.predict(x)
            if not _is_networklike(net):
                raise TypeError("surrogate.predict() вернул объект, не совместимый с NetworkLike")

            rep = self.spec.report(net, reduction=self.reduction)

            # rep["__penalty__"] формируется в Specification.report(...)
            rep["__objective__"] = float(rep.get("__penalty__", self.large_penalty))
            rep["__spec_name__"] = str(self.spec.name)
            return rep
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[PenaltyObjective.report] ошибка: {e!r}", RuntimeWarning)
            return {"__objective__": self.large_penalty, "__error__": repr(e)}

    # ---------------------------------------------------------------- vector-API
    def evaluate_vector(self, z: Sequence[float], names: Sequence[str]) -> float:
        """
        Утилита для оптимизаторов: собрать dict из (names,z) и вызвать __call__.

        Защищаемся от тихого усечения zip().
        """
        if len(z) != len(names):
            raise ValueError("evaluate_vector: длины z и names не совпадают")
        x = {k: float(v) for k, v in zip(names, z)}
        return self(x)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        return f"PenaltyObjective(reduction={self.reduction})"


# ────────────────────────────────────────────────────────────────────────────
#                           FeasibleYieldObjective
# ────────────────────────────────────────────────────────────────────────────

class FeasibleYieldObjective:
    """
    Составная цель: всегда минимизируем penalty и, ДОПОЛНИТЕЛЬНО,
    если точка выполнима по ТЗ (spec.is_ok(net) == True), поощряем локальный yield:

        J(x) = penalty(x) - alpha * I{is_ok(x)} * Y_local(x)

    Где:
      • penalty(x) = spec.penalty(surrogate.predict(x), reduction)
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
    reduction : {'sum','mean','max'}, default='sum'
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

    Ключевые улучшения по сравнению с предыдущей версией:
    ------------------------------------------------
    1) penalty и is_ok вычисляются за один проход (через spec.evaluate).
    2) В batch() не возвращаем inf, только large_penalty.
    3) В _yield_local есть fallback, если surrogate не реализует passes_spec.
    4) Integer bounds выравниваются по step корректно: lower вниз, upper вверх.
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
        reduction: str = "sum",
        large_penalty: float = 1e9,
    ):
        self.sur = surrogate
        self.spec = spec

        # --- источники локальных допусков
        self.base_space = base_space
        self.tol_space = tolerance_space
        self.delta = delta

        dm = str(delta_mode).strip().lower()
        if dm not in ("abs", "rel"):
            raise ValueError("delta_mode must be 'abs' or 'rel'")
        self.delta_mode = dm

        if self.tol_space is None and (self.base_space is None or self.delta is None):
            raise ValueError("FeasibleYieldObjective: задайте либо tolerance_space, либо (base_space и delta)")

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
        self.reduction: ReductionMode = _normalize_reduction(reduction)
        self.large_penalty = float(large_penalty)

    # ────────────────────────────────────────────────────────────
    # Внутренние строительные блоки
    # ────────────────────────────────────────────────────────────

    def _build_local_space(self, x: Mapping[str, Any]) -> DesignSpace:
        """
        Собирает локальный DesignSpace вокруг точки x, **сохраняя типы переменных**.

        Источники допусков:
          - tolerance_space.to_center_delta(...), либо
          - base_space + delta (+ delta_mode).

        ВАЖНО про delta_mode='rel':
          - интерпретируем delta как долю от |центра|;
          - если центр ~ 0, используем scale=1.0 (иначе допуск станет 0).
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
            # delta задан либо числом, либо dict по именам
            if self.delta is None:
                raise RuntimeError("delta is None (внутренняя ошибка конфигурации)")

            for name in self.base_space.names():
                if name not in x:
                    raise KeyError(f"В точке x отсутствует параметр '{name}'")

                center = float(x[name])

                if isinstance(self.delta, dict):
                    if name not in self.delta:
                        raise KeyError(f"delta dict не содержит ключ '{name}' (нужен допуск для каждого параметра)")
                    d_spec = self.delta[name]
                else:
                    d_spec = self.delta

                if isinstance(d_spec, tuple):
                    dm, dp = map(float, d_spec)
                    dm, dp = abs(dm), abs(dp)
                else:
                    r = float(d_spec)  # type: ignore[arg-type]
                    dm, dp = abs(r), abs(r)

                if self.delta_mode == "rel":
                    scale = abs(center)
                    if scale == 0.0:
                        scale = 1.0
                    dm *= scale
                    dp *= scale

                dm_dp_map[name] = (dm, dp)

        # --- построить локальные переменные с сохранением типа
        local_vars: Dict[str, Any] = {}

        for name in self.base_space.names():
            base_var = self.base_space[name]
            xval = x[name]

            # 1) Categorical/Ordinal: определяем по наличию levels
            if getattr(base_var, "levels", None) is not None:
                levels = list(base_var.levels)  # type: ignore[attr-defined]

                if not self.vary_categorical:
                    # фиксируем текущий уровень
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

                c = int(round(float(xval)))
                c = int(np.clip(c, lo_g, hi_g))

                if not self.vary_integers:
                    # замораживаем (но приводим к сетке step)
                    if step > 1:
                        c = lo_g + int(round((c - lo_g) / step)) * step
                        c = int(np.clip(c, lo_g, hi_g))
                    local_vars[name] = IntegerVar(
                        lower=c,
                        upper=c,
                        step=step,
                        unit=getattr(base_var, "unit", ""),
                    )
                else:
                    dm, dp = dm_dp_map[name]
                    lo = int(math.floor(c - dm))
                    hi = int(math.ceil(c + dp))

                    # ВАЖНО: lower выравниваем вниз, upper вверх
                    lo = _align_down(lo, lo_g, step)
                    hi = _align_up(hi, lo_g, step)

                    lo = int(np.clip(lo, lo_g, hi_g))
                    hi = int(np.clip(hi, lo_g, hi_g))

                    if hi < lo:
                        lo = hi = c

                    local_vars[name] = IntegerVar(
                        lower=lo,
                        upper=hi,
                        step=step,
                        unit=getattr(base_var, "unit", ""),
                    )
                continue

            # 3) Continuous
            lo_g, hi_g = base_var.bounds()
            dm, dp = dm_dp_map[name]
            c = float(xval)

            lo = float(np.clip(c - dm, lo_g, hi_g))
            hi = float(np.clip(c + dp, lo_g, hi_g))
            if hi < lo:
                lo = hi = float(np.clip(c, lo_g, hi_g))

            local_vars[name] = ContinuousVar(
                lower=lo,
                upper=hi,
                unit=getattr(base_var, "unit", ""),
            )

        return DesignSpace(local_vars)

    def _yield_local(self, x: Mapping[str, Any]) -> float:
        """
        Детерминированная оценка локального yield вокруг x:

          1) строим локальный DesignSpace;
          2) генерируем n_mc точек (детерминизм задаётся rng);
          3) проверяем pass по spec:
             - если surrogate поддерживает passes_spec -> используем его (быстро),
             - иначе fallback: batch_predict + spec.is_ok.
        """
        space = self._build_local_space(x)

        sampler: BaseSampler = get_sampler(self.sampler_name, rng=self.rng, **self.sampler_kwargs)
        pts = space.sample(self.n_mc, sampler=sampler, reject_invalid=False)

        # Быстрый путь (GPU-friendly для нейросуррогатов)
        fn = getattr(self.sur, "passes_spec", None)
        if callable(fn):
            ok = fn(pts, self.spec)
            ok_arr = np.asarray(ok, dtype=bool)
        else:
            # Fallback: batch_predict -> list[NetworkLike] -> is_ok поштучно
            nets = self.sur.batch_predict(pts)
            ok_arr = np.fromiter((self.spec.is_ok(net) for net in nets), dtype=bool, count=len(nets))

        y = float(np.mean(ok_arr))

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

        ВАЖНО: penalty и all_ok считаются одним проходом через spec.evaluate.
        """
        try:
            net = self.sur.predict(x)
            if not _is_networklike(net):
                raise TypeError("surrogate.predict() вернул объект, не совместимый с NetworkLike")

            p, feas = _penalty_and_all_ok_via_evaluate(self.spec, net, self.reduction)
            if not feas:
                return p  # infeasible: без MC

            y = self._yield_local(x)
            return p - self.alpha * y

        except Exception as e:  # pragma: no cover
            warnings.warn(
                f"[FeasibleYieldObjective] ошибка оценки {dict(x)}: {e!r} → large_penalty",
                RuntimeWarning,
            )
            return self.large_penalty

    def batch(self, xs: Sequence[Mapping[str, float]]) -> np.ndarray:
        """
        Батч-оценка: penalty считается batched; yield — только для выполнимых.
        Возвращает массив формы (B,).

        Замечание:
        - penalty/is_ok для центральных точек всё равно приходится считать по одной
          сети (по результатам batch_predict).
        """
        if not xs:
            return np.zeros((0,), dtype=float)

        try:
            nets = self.sur.batch_predict(xs)
        except Exception as e:  # pragma: no cover
            warnings.warn(
                f"[FeasibleYieldObjective.batch] batch_predict упал: {e!r}; fallback на поэлементный режим",
                RuntimeWarning,
            )
            return np.fromiter((self(x) for x in xs), dtype=float, count=len(xs))

        if len(nets) != len(xs):
            warnings.warn(
                "[FeasibleYieldObjective.batch] batch_predict вернул список другой длины; fallback на поэлементный режим",
                RuntimeWarning,
            )
            return np.fromiter((self(x) for x in xs), dtype=float, count=len(xs))

        B = len(xs)
        out = np.empty((B,), dtype=float)

        feas_mask = np.zeros((B,), dtype=bool)
        pens = np.empty((B,), dtype=float)

        # 1) penalty и feasible — по центральным сетям
        for i, net in enumerate(nets):
            try:
                if not _is_networklike(net):
                    raise TypeError("batch_predict: элемент не совместим с NetworkLike")

                p, feas = _penalty_and_all_ok_via_evaluate(self.spec, net, self.reduction)
                pens[i] = p
                feas_mask[i] = feas
            except Exception:  # pragma: no cover
                pens[i] = self.large_penalty
                feas_mask[i] = False

        # 2) На infeasible → просто penalty
        out[~feas_mask] = pens[~feas_mask]

        # 3) Для feasible — penalty - alpha * Y_local
        for i, x in enumerate(xs):
            if feas_mask[i]:
                try:
                    y = self._yield_local(x)
                    out[i] = pens[i] - self.alpha * y
                except Exception:  # pragma: no cover
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
            if not _is_networklike(net):
                raise TypeError("surrogate.predict() вернул объект, не совместимый с NetworkLike")

            p, feas = _penalty_and_all_ok_via_evaluate(self.spec, net, self.reduction)

            if not feas:
                rep.update({"__objective__": p, "__penalty__": p, "__feasible__": False, "__yield__": None})
                return rep

            y = self._yield_local(x)
            cost = p - self.alpha * y
            rep.update(
                {"__objective__": float(cost), "__penalty__": float(p), "__feasible__": True, "__yield__": float(y)}
            )
            return rep

        except Exception as e:  # pragma: no cover
            warnings.warn(f"[FeasibleYieldObjective.report] ошибка: {e!r}", RuntimeWarning)
            rep.update({"__objective__": self.large_penalty, "__error__": repr(e)})
            return rep

    def evaluate_vector(self, z: Sequence[float], names: Sequence[str]) -> float:
        """Собрать dict из (names,z) и вызвать __call__."""
        if len(z) != len(names):
            raise ValueError("evaluate_vector: длины z и names не совпадают")
        x = {k: float(v) for k, v in zip(names, z)}
        return self(x)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FeasibleYieldObjective(alpha={self.alpha}, sampler={self.sampler_name}, "
            f"n_mc={self.n_mc}, reduction={self.reduction})"
        )


__all__ = [
    "PenaltyObjective",
    "FeasibleYieldObjective",
]