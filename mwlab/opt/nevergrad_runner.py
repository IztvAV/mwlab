# mwlab/opt/nevergrad_runner.py
# -*- coding: utf-8 -*-
"""
Nevergrad: тонкая обёртка для глобальной оптимизации по произвольной цели
=========================================================================

Цель модуля — дать **минималистичный, но удобный** слой над Nevergrad,
который:

* строит параметризацию из `mwlab.opt.design.space.DesignSpace`;
* уважает типы переменных (Continuous / Integer(step) / Ordinal / Categorical);
* позволяет оптимизировать **любую** цель `objective(x: dict) -> float`
  (например, `PenaltyObjective` из A);
* использует батчевую оценку, если у цели есть `objective.batch(list[dict])`;
* поддерживает ранний останов:
  - `early_stop_on_zero`: если глобальный минимум ≤ tol и держится `patience` итераций;
* пишет полную историю (каждая оценка): i, value, best, x;
* возвращает удобный результат с best-точкой, историей и top-K лучшими.

Почему **здесь**, а не в оркестраторе?
--------------------------------------
Оркестратор (D) лишь «складывает» элементы. Конкретная работа с Nevergrad —
в этом модуле: это облегчает юнит-тесты, повторное использование и изоляцию
зависимости `nevergrad`.

Зависимости
-----------
    pip install nevergrad

Пример
------
>>> from mwlab.opt.design.space import DesignSpace, ContinuousVar, IntegerVar
>>> from mwlab.opt.objectives.penalty import PenaltyObjective
>>> space = DesignSpace({
...     "w": ContinuousVar(-1e-4, 1e-4),
...     "n_taps": IntegerVar(lower=2, upper=12, step=2),
... })
>>> obj = PenaltyObjective(surrogate=sur, spec=spec)
>>> opt = NGOptimizer(algo="NGOpt", budget=2000, population=64, seed=2025, num_workers=8,
...                   patience=20, early_stop_on_zero=True)
>>> result = opt.run(space=space, objective=obj, topk=20)
>>> print(result.best_value, result.best_x)
>>> # result.trace — pandas.DataFrame (если установлен pandas), иначе list[dict]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Callable
import math
import warnings
import time
from concurrent.futures import ThreadPoolExecutor

# ────────────────────────────────────────────────────────────────────────────
# Внешняя зависимость: nevergrad
# ────────────────────────────────────────────────────────────────────────────
try:
    import nevergrad as ng
except Exception as _NG_ERR:  # pragma: no cover
    ng = None
    _NG_IMPORT_ERROR = _NG_ERR  # чтобы показать понятную ошибку при использовании

# прогресс-бар (опционально)
try:  # pragma: no cover
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False

# ────────────────────────────────────────────────────────────────────────────
# MWLab: дизайн-спейс и типы переменных
# ────────────────────────────────────────────────────────────────────────────
from mwlab.opt.design.space import (
    DesignSpace,
    ContinuousVar,
    IntegerVar,
    OrdinalVar,
    CategoricalVar,
)

# pandas — опционально: для удобного DataFrame-трейса
try:  # pragma: no cover
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


# ────────────────────────────────────────────────────────────────────────────
#                               Типы / протоколы
# ────────────────────────────────────────────────────────────────────────────

ObjectiveLike = Callable[[Mapping[str, Any]], float]

class _HasBatch:
    """Протокол: цель с батч-API."""
    def batch(self, xs: Sequence[Mapping[str, Any]]) -> "np.ndarray":  # noqa: F821
        ...  # type: ignore[func-returns-value]

# numpy только для аннотаций/типов в docstring; импортировать ниже в функции,
# чтобы не держать модуль, если он не нужен.
import numpy as np  # неизбежная зависимость; уже есть в проекте


# ────────────────────────────────────────────────────────────────────────────
#                                 Результат
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class NGResult:
    """Результат одного запуска оптимизации."""
    best_x: Dict[str, Any]
    best_value: float
    algo: str
    budget: int
    evaluations: int
    seed: Optional[int]
    trace: "pd.DataFrame|List[Dict[str, Any]]"  # зависит от наличия pandas
    topk: List[Dict[str, Any]]                  # [{..., "__value__": float}, ...]
    elapsed_sec: float

    # --- внутренности nevergrad (best-effort; полезно для мета-оптимизаторов NGOpt/NgIohTuned)
    # outer_optimizer: класс реального объекта nevergrad-оптимизатора (ask/tell)
    # inner_optimizer: выбранный базовый оптимизатор (часто optimizer.optim / optimizer._optim)
    # inner_components: список компонент (если это портфолио/competitor)
    # optimizer_repr: repr(...) для отладки
    outer_optimizer: Optional[str] = None
    inner_optimizer: Optional[str] = None
    inner_components: Optional[List[str]] = None
    optimizer_repr: Optional[str] = None

    def to_dataframe(self) -> "pd.DataFrame":
        """Удобный доступ к трейсу в виде DataFrame (если установлен pandas)."""
        if not _HAS_PANDAS:
            raise ImportError("pandas не установлен. Установите: pip install pandas")
        return self.trace if isinstance(self.trace, pd.DataFrame) else pd.DataFrame(self.trace)

# ────────────────────────────────────────────────────────────────────────────
#                                 Оптимизатор
# ────────────────────────────────────────────────────────────────────────────

class NGOptimizer:
    """
    Лёгкая обёртка над Nevergrad.

    Параметры
    ---------
    algo : str, default="NGOpt"
        Имя оптимизатора из `ng.optimizers.registry` (без учёта регистра),
        например: "NGOpt", "CMA", "DE", "OnePlusOne", "TwoPointsDE", "PSO" …
    budget : int
        Общее число **оценок цели** (не итераций), максимум вызовов `objective`.
    population : int, optional
        Размер «пачки» кандидатов на одной микро-итерации ask/tell.
        Если `None`, по умолчанию берём `max(1, num_workers)`.
        Не во всех оптимизаторах это «популяция» в строгом смысле — мы используем
        это как **батч-размер** оценок за шаг.
    num_workers : int, default=1
        Количество потоков **оценки** цели (внутри модуля). Если цель умеет
        `batch(...)` — используем батч и не запускаем потоки (обычно быстрее
        и детерминированнее). Если батча нет — используем ThreadPoolExecutor.
    seed : int | None
        Сид RNG для воспроизводимости.
    patience : int | None
        Ранний останов по критерию «цель близка к нулю»: если `None`, не используем.
        Если задано число N — завершаем, когда **глобальный минимум** ≤ zero_tol
        и удерживается **N последовательных шагов** (в терминах наших батч-шагов).
    early_stop_on_zero : bool, default=True
        Включить/отключить ранний останов (работает только если `patience` задан).
    zero_tol : float, default=0.0
        Порог «ноль штрафа/цели». Для мягких целей можно поставить 1e-9, 1e-6 и т.п.
    log_every : int | None
        Если задано — печатать краткий прогресс каждые `log_every` батч-шагов.
        На ход оптимизации не влияет.
    """

    def __init__(
        self,
        *,
        algo: str = "NGOpt",
        budget: int = 10_000,
        population: Optional[int] = None,
        num_workers: int = 1,
        seed: Optional[int] = None,
        patience: Optional[int] = None,
        early_stop_on_zero: bool = True,
        zero_tol: float = 0.0,
        log_every: Optional[int] = None,
        show_progress: bool = True,
    ):
        if ng is None:  # pragma: no cover
            raise ImportError(
                f"nevergrad не установлен или не импортируется: {_NG_IMPORT_ERROR}\n"
                f"Установите: pip install nevergrad"
            )

        self.algo = str(algo)
        self.budget = int(budget)
        self.num_workers = max(1, int(num_workers))
        self.population = int(population) if population is not None else max(1, self.num_workers)
        self.seed = None if seed is None else int(seed)
        self.patience = None if patience is None else max(1, int(patience))
        self.early_stop_on_zero = bool(early_stop_on_zero)
        self.zero_tol = float(zero_tol)
        self.log_every = None if log_every is None else max(1, int(log_every))
        self.show_progress = bool(show_progress)

        # внутренняя история (одна запись на КАЖДЫЙ вызов цели)
        self._history: List[Dict[str, Any]] = []
        self._best_so_far: float = math.inf
        self._eval_count: int = 0

        # счётчик «сколько шагов подряд глобальный минимум ≤ zero_tol»
        self._zero_hold_streak: int = 0

        # --- отладка/интроспекция: сохраняем последний созданный nevergrad-оптимизатор
        # Это важно, потому что сам объект optimizer создаётся внутри run() как локальная переменная.
        self._last_parametrization = None
        self._last_optimizer = None
        self._last_optimizer_info: Dict[str, Any] = {}

    @staticmethod
    def _extract_ng_optimizer_info(optimizer: Any) -> Dict[str, Any]:
        """
        Извлечь информацию о nevergrad-оптимизаторе (best-effort):
          - outer: класс объекта с ask/tell (реальный оптимизатор)
          - inner: выбранный базовый оптимизатор у мета-оптимизаторов (optim/_optim)
          - components: список компонент у портфолио/competitor (если доступно)
          - repr: строковое представление repr(...) для отладки
        """
        out: Dict[str, Any] = {
            "outer": None,
            "inner": None,
            "components": None,
            "repr": None,
        }
        if optimizer is None:
            return out

        out["outer"] = type(optimizer).__name__
        try:
            out["repr"] = repr(optimizer)
        except Exception:
            out["repr"] = None

        # У мета-оптимизаторов (NGOpt/NgIohTuned) часто есть ссылка на выбранный внутренний оптимизатор:
        #   optimizer.optim или optimizer._optim
        inner = None
        for attr in ("optim", "_optim"):
            try:
                inner = getattr(optimizer, attr, None)
                if inner is not None and inner is not optimizer:
                    break
            except Exception:
                inner = None

        if inner is not None and inner is not optimizer:
            out["inner"] = type(inner).__name__
            # Портфолио/competitor: иногда хранит список компонент
            for cand in ("optimizers", "_optimizers", "optims", "_optims"):
                try:
                    comps = getattr(inner, cand, None)
                    if comps is not None:
                        out["components"] = [type(c).__name__ for c in list(comps)]
                        break
                except Exception:
                    continue
        else:
            # Иногда список компонент есть у outer-объекта напрямую
            for cand in ("optimizers", "_optimizers", "optims", "_optims"):
                try:
                    comps = getattr(optimizer, cand, None)
                    if comps is not None:
                        out["components"] = [type(c).__name__ for c in list(comps)]
                        break
                except Exception:
                    continue

        return out

    # ──────────────────────────────────────────────────────────────
    #          Построение NG-параметризации из DesignSpace
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_parametrization(space: DesignSpace) -> ng.p.Dict:
        """
        Для каждой переменной `DesignSpace` строим Nevergrad-параметр:

        * ContinuousVar  → p.Scalar(lower, upper)
        * IntegerVar     → p.Scalar(lower, upper).set_integer_casting()
                           (учёт `step` делаем при **декоде** результата)
        * OrdinalVar     → p.Choice(levels)
        * CategoricalVar → p.Choice(levels)

        Возвращаем `ng.p.Dict`, чтобы `candidate.value` сразу был `dict`.
        """
        mapping: Dict[str, ng.p.Parameter] = {}
        for name in space.names():
            var = space[name]

            if isinstance(var, ContinuousVar):
                lo, hi = var.bounds()
                mapping[name] = ng.p.Scalar(lower=float(lo), upper=float(hi))
                continue

            if isinstance(var, IntegerVar):
                lo, hi = var.bounds()
                p = ng.p.Scalar(lower=float(lo), upper=float(hi)).set_integer_casting()
                # Запомним шаг на параметре — пригодится при декоде
                setattr(p, "_mw_step", int(getattr(var, "step", 1)))  # type: ignore[attr-defined]
                mapping[name] = p
                continue

            if isinstance(var, OrdinalVar):
                mapping[name] = ng.p.Choice(list(var.levels))
                continue

            if isinstance(var, CategoricalVar):
                mapping[name] = ng.p.Choice(list(var.levels))
                continue

            raise TypeError(f"Неизвестный тип переменной '{name}': {type(var)}")

        return ng.p.Dict(**mapping)

    @staticmethod
    def _decode(space: DesignSpace, cand_value: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Аккуратно приводит значения из NG к «физическим»:
        * IntegerVar — округление и посадка на сетку `step`, затем клип в [lower, upper];
        * Остальные — как есть (границы NG гарантирует).
        """
        out: Dict[str, Any] = {}
        for name, val in cand_value.items():
            var = space[name]

            if isinstance(var, IntegerVar):
                step = int(getattr(var, "step", 1))
                ival = int(round(float(val)))
                if step > 1:
                    origin = int(var.lower)
                    k = int(round((ival - origin) / step))
                    ival = origin + k * step
                ival = int(max(var.lower, min(ival, var.upper)))
                out[name] = ival
                continue

            if isinstance(var, ContinuousVar):
                x = float(val)
                lo, hi = var.bounds()
                out[name] = float(max(lo, min(x, hi)))
                continue

            # Ordinal/Categorical — значение Choice (уровень/строка)
            out[name] = val

        return out

    # ──────────────────────────────────────────────────────────────
    #                             Запуск
    # ──────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        space: DesignSpace,
        objective: ObjectiveLike | _HasBatch,
        topk: int = 10,
    ) -> NGResult:
        """
        Главный метод: собрать оптимизатор, выполнить ask/tell-цикл, вернуть результат.

        Parameters
        ----------
        space : DesignSpace
            Пространство параметров.
        objective : callable(dict)->float (и/или .batch(list[dict])->np.ndarray)
            Целевая функция. Если у объекта есть метод `.batch(...)`, он будет
            использован для векторной оценки (значительно быстрее).
        topk : int
            Сколько лучших уникальных точек вернуть в результате.

        Returns
        -------
        NGResult
        """
        # IMPORTANT: допускаем повторные вызовы run() на одном экземпляре оптимизатора
        self._history = []
        self._best_so_far = math.inf
        self._eval_count = 0
        self._zero_hold_streak = 0

        # 0) Параметризация и оптимизатор
        parametrization = self._build_parametrization(space)
        opt_cls = self._resolve_optimizer(self.algo)
        optimizer = opt_cls(parametrization=parametrization, budget=self.budget, num_workers=self.num_workers)

        # Сохраняем для внешней интроспекции/отладки (например, чтобы понять, что выбрал NGOpt внутри)
        self._last_parametrization = parametrization
        self._last_optimizer = optimizer

        # СИДИРОВАНИЕ RNG: даём корректный RandomState (нужен .randn для CMA)
        if self.seed is not None:
            import numpy as _np
            _rs = _np.random.RandomState(int(self.seed))
            # 1) параметризация: если поддерживает random_state — передаём именно RandomState
            try:
                parametrization.random_state = _rs
            except Exception:
                pass
            # 2) оптимизатор: если уже есть _rng с .randn → просто seed(); иначе подменим на RandomState
            try:
                if hasattr(optimizer, "_rng") and hasattr(optimizer._rng, "randn"):
                    optimizer._rng.seed(int(self.seed))
                else:
                    optimizer._rng = _rs  # у CMA это работает корректно
            except Exception:
                pass

        # 1) Основной цикл ask/tell по **батч-шагам**
        start = time.time()
        pbar = None
        if self.show_progress and _HAS_TQDM:
            pbar = tqdm(total=self.budget, desc=f"{self.algo} | budget {self.budget}", leave=True)

        remaining = self.budget
        step_idx = 0

        while remaining > 0:
            step_idx += 1
            batch_size = min(self.population, remaining)

            # 1.1 ask() → кандидаты (NG-объекты), получим их словарные значения
            cands = [optimizer.ask() for _ in range(batch_size)]
            xs = [self._decode(space, c.value) for c in cands]  # type: ignore[attr-defined]

            # 1.2 оцениваем цель: батчево, если доступно
            if hasattr(objective, "batch") and callable(getattr(objective, "batch")):
                try:
                    vals = getattr(objective, "batch")(xs)  # type: ignore[misc]
                    vals = np.asarray(vals, dtype=float).reshape(-1)
                    if vals.shape[0] != len(xs):
                        raise ValueError("objective.batch вернул массив неправильной длины")
                except Exception as e:  # pragma: no cover
                    warnings.warn(
                        f"[NGOptimizer] objective.batch упал: {e!r}; переключаюсь на поэлементную оценку",
                        RuntimeWarning,
                    )
                    # небатчевый fallback с учётом num_workers
                    if self.num_workers > 1:
                        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                            vals = np.fromiter(ex.map(lambda _x: float(objective(_x)), xs), dtype=float, count=len(xs))
                    else:
                        vals = np.fromiter((float(objective(x)) for x in xs), dtype=float, count=len(xs))

            else:
                # небатчевый режим: используем ThreadPoolExecutor при num_workers>1
                if self.num_workers > 1:
                    with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                        vals = np.fromiter(ex.map(lambda _x: float(objective(_x)), xs), dtype=float, count=len(xs))
                else:
                    vals = np.fromiter((float(objective(x)) for x in xs), dtype=float, count=len(xs))

            # 1.3 tell() → сообщаем оптимизатору значения
            for cand, val, x in zip(cands, vals, xs):
                loss = float(val)
                # учтём нечисловые ответы (safety)
                if not math.isfinite(loss):
                    loss = 1e12
                optimizer.tell(cand, loss)  # обновляем NG-состояние

                # логирование детально — на КАЖДЫЙ вызов цели
                self._eval_count += 1
                prev_best = self._best_so_far
                self._best_so_far = min(self._best_so_far, loss)
                self._history.append({
                    "i": self._eval_count,
                    "value": loss,
                    "best": self._best_so_far,
                    "x": x,
                })
                if pbar is not None:
                    pbar.update(1)
                    improve = max(0.0, prev_best - self._best_so_far)
                    pbar.set_postfix({
                        "best": f"{self._best_so_far:.4g}",
                        "last": f"{loss:.4g}",
                        "impr": f"{improve:.2g}",
                    })

            remaining -= batch_size

            # 1.4 ранний останов: «ноль держится N шагов»
            if (self.patience is not None) and self.early_stop_on_zero:
                if self._best_so_far <= self.zero_tol:
                    self._zero_hold_streak += 1
                else:
                    self._zero_hold_streak = 0
                if self._zero_hold_streak >= self.patience:
                    if self.log_every:
                        print(f"[NG] early stop: best ≤ {self.zero_tol} удерживается {self.patience} шагов")
                    break

            # 1.5 опциональный лог
            if (self.log_every is not None) and (step_idx % self.log_every == 0):
                print(f"[NG] step {step_idx:5d} | evals {self._eval_count:6d}/{self.budget:6d} | "
                      f"best {self._best_so_far:.6g}")

        if pbar is not None:
            pbar.close()
        elapsed = time.time() - start

        # --- Интроспекция nevergrad: снимаем информацию ПОСЛЕ оптимизации
        # (у некоторых мета-оптимизаторов внутренности окончательно проявляются/стабилизируются только к концу)
        opt_info = self._extract_ng_optimizer_info(optimizer)
        self._last_optimizer_info = dict(opt_info)

        # 2) Выбираем лучший результат: в первую очередь — из нашей истории,
        #    т.к. provide_recommendation() у некоторых оптимизаторов может вернуть loss=None.
        if not self._history:
            # Бюджет 0 или что-то пошло не так — попробуем взять рекомендацию как есть
            recommendation = optimizer.provide_recommendation()
            best_x = self._decode(space, recommendation.value)  # type: ignore[attr-defined]
            best_v = float("inf") if getattr(recommendation, "loss", None) is None else float(recommendation.loss)
        else:
            hist_best = min(self._history, key=lambda r: float(r["value"]))
            best_x = dict(hist_best["x"])
            best_v = float(hist_best["value"])
            # Дополнительно: если рекомендация NG есть и у неё валидный loss — возьмём минимум из двух
            try:
                recommendation = optimizer.provide_recommendation()
                rec_loss = getattr(recommendation, "loss", None)
                if rec_loss is not None:
                    rec_v = float(rec_loss)
                    if math.isfinite(rec_v) and rec_v < best_v:
                        best_v = rec_v
                        best_x = self._decode(space, recommendation.value)  # type: ignore[attr-defined]
            except Exception:
                pass

        # 3) Топ-K (уникализация по значениям X, округлённым до 12 знаков)
        top_list = self._make_topk_from_history(self._history, k=topk)

        # 4) Трейс
        trace_obj: "pd.DataFrame|List[Dict[str, Any]]"
        if _HAS_PANDAS:
            # плоская таблица: i, value, best, …flatten(x)…
            rows = self._history
            x_keys = sorted({k for r in rows for k in (r.get("x") or {}).keys()})
            data: List[Dict[str, Any]] = []
            for r in rows:
                row = {"i": r["i"], "value": r["value"], "best": r["best"]}
                for k in x_keys:
                    row[k] = (r["x"].get(k) if isinstance(r.get("x"), dict) else None)
                data.append(row)
            trace_obj = pd.DataFrame(data)
        else:
            trace_obj = list(self._history)

        return NGResult(
            best_x=best_x,
            best_value=best_v,
            algo=self.algo,
            budget=self.budget,
            evaluations=self._eval_count,
            seed=self.seed,
            trace=trace_obj,
            topk=top_list,
            elapsed_sec=elapsed,
            outer_optimizer=opt_info.get("outer"),
            inner_optimizer=opt_info.get("inner"),
            inner_components = opt_info.get("components"),
            optimizer_repr = opt_info.get("repr"),
        )

    # ──────────────────────────────────────────────────────────────
    #                               Вспомогательные
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_optimizer(name: str):
        """Находим класс оптимизатора по имени (без учёта регистра) в реестре NG."""
        # В некоторых версиях NG ключи — чувствительны к регистру. Поддержим оба пути.
        reg = ng.optimizers.registry  # короткая ссылка
        if name in reg:
            return reg[name]
        # Поищем case-insensitive
        low = name.lower()
        for k in reg.keys():
            if k.lower() == low:
                return reg[k]
        # Ничего не нашли — соберём список доступных
        choices = ", ".join(sorted(reg.keys()))
        raise KeyError(f"Оптимизатор Nevergrad '{name}' не найден. Доступные: {choices}")

    @staticmethod
    def _make_topk_from_history(history: List[Dict[str, Any]], *, k: int = 10) -> List[Dict[str, Any]]:
        """
        Топ-K уникальных по X кандидатов из истории (меньший value — лучше).
        Уникализация: кортеж (name, round(value, 12)) по всем параметрам.
        """
        if not history:
            return []
        sorted_hist = sorted(history, key=lambda r: float(r["value"]))
        uniq: List[Dict[str, Any]] = []
        seen: set = set()

        def _canon(v: Any) -> Any:
            # численные значения округляем, категориальные/ordinal оставляем как есть
            try:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    return round(float(v), 12)
            except Exception:
                pass
            return v

        for rec in sorted_hist:
            x = rec.get("x", {})
            key = tuple((n, _canon(v)) for n, v in sorted(x.items()))
            if key in seen:
                continue
            seen.add(key)
            row = dict(x)
            row["__value__"] = float(rec["value"])
            uniq.append(row)
            if len(uniq) >= k:
                break
        return uniq

