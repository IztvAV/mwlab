# mwlab/opt/design/samplers.py
"""
MWLab · opt · design · samplers
================================
Базовая инфраструктура **сэмплеров** – генераторов точек DOE (Design of
Experiments) внутри произвольного `DesignSpace`.

Главная цель – дать инженер‑RF пользователю единый, воспроизводимый и
расширяемый способ получать наборы точек для обучения суррогатной модели,
анализа допуска или глобальной оптимизации.

**Ключевые принципы**
--------------------
1. **Воспроизводимость**  – единственный источник случайности (`seed`) →
   синхронно инициализирует NumPy и (опц.) PyTorch‐ГСЧ.
2. **Инкрементальность**  – последовательные вызовы `sample()` продолжают
   последовательность без повторений (поддержано `fast_forward` Sobol / Halton).
3. **Registry‑pattern**  – новый класс‑сэмплер регистрируется декоратором
   `@register("alias", *aliases)`.  Отсюда:

   ```python
   pts = space.sample(128, sampler="sobol", scramble=False, rng=2025)
   ```
4. **Минимальные зависимости**  – SciPy ≥ 1.11 входит в core; `pyDOE2` –
   опционально `pip install mwlab[doe]` (плейсхолдеры дают понятный ImportError).
5. **Поддержка int / cat переменных**  – в helper `_to_dict` проверяем
   `is_integer` и `levels`.

Структура файла
---------------
* **Registry**  – '_SAMPLER_REGISTRY', `@register` / `get_sampler`.
* **RNG helper** – `_resolve_rng` объединяет NumPy + Torch.
* **BaseSampler** – абстракт, хранит состояние и предоставляет
  `state_dict()` / `load_state_dict()`.
* **Core‑samplers** – Sobol, Halton, LHS, LHS‑maximin, Normal, 2^d plan.
* **pyDOE2‑samplers** – Fractional factorial, Plackett–Burman, CCD, BB.

Пример
------
```python
from mwlab.opt.design.space import DesignSpace, ContinuousVar
from mwlab.opt.design.samplers import get_sampler

space = DesignSpace({
    "w"  : ContinuousVar(-1e-4, 1e-4),
    "gap": ContinuousVar(center=0.0, delta=1e-4),
})

sampler = get_sampler("sobol", scramble=True, rng=42)
pts     = sampler.sample(space, n=4)
# → [{'w': 6.10e-05, 'gap': 4.27e-05}, ...]
```
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Tuple, Union
import warnings
import importlib.util

import numpy as np

# Torch – опционален (для синхронного сидирования и SobolEngine)
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# SciPy QMC – базовые Sobol / Halton / LHS + optimize_lhs
from scipy.stats import qmc  # type: ignore

# ────────────────────────────────────────────────────────────────────────────
#                                 Typing
# ────────────────────────────────────────────────────────────────────────────
PointDict = Dict[str, float]
RngLike   = Union[int, np.random.Generator, "torch.Generator"]  # noqa: F821

# ────────────────────────────────────────────────────────────────────────────
#                            Registry helpers
# ────────────────────────────────────────────────────────────────────────────
_SAMPLER_REGISTRY: Dict[str, "BaseSampler"] = {}

def register(*aliases: str) -> Callable[["BaseSampler"], "BaseSampler"]:
    """Декоратор регистрации класса‑сэмплера под одним или несколькими alias‑ами.

    Пример
    -------
    >>> @register("sobol", "qmc_sobol")
    ... class SobolSampler(BaseSampler): ...
    """
    def _wrap(cls):  # type: ignore[override]
        for name in aliases:
            if name in _SAMPLER_REGISTRY:
                raise KeyError(f"Sampler alias '{name}' already exists")
            _SAMPLER_REGISTRY[name] = cls
        cls._sampler_aliases = aliases  # type: ignore[attr-defined]
        return cls
    return _wrap

def get_sampler(name: str, **kwargs) -> "BaseSampler":
    """Фабрика: возвращает экземпляр сэмплера по alias‑у."""
    if name not in _SAMPLER_REGISTRY:
        raise KeyError(
            f"Sampler '{name}' not found. Available: {list(_SAMPLER_REGISTRY)}"
        )
    return _SAMPLER_REGISTRY[name](**kwargs)  # type: ignore[call-arg]

# ────────────────────────────────────────────────────────────────────────────
#                       RNG – единый источник случайности
# ────────────────────────────────────────────────────────────────────────────

def _resolve_rng(
    rng: RngLike | None = None,
) -> Tuple[np.random.Generator, "torch.Generator" | None, int]:
    """Возвращает кортеж *(np_rng, torch_gen | None, seed_int)*.

    Логика
    ------
    * `None`         → seed = 0 (по‑умолчанию)
    * `int`          → тот же seed для NumPy и Torch
    * NumPy.Generator → считаем начальный seed неизменным и синхронизируем Torch
    * Torch.Generator → наоборот
    """

    # case: None / int
    if rng is None or isinstance(rng, int):
        seed = int(rng or 0) & 0xFFFFFFFF
        np_rng = np.random.default_rng(seed)
        t_gen = torch.Generator().manual_seed(seed) if torch is not None else None  # type: ignore[arg-type]
        return np_rng, t_gen, seed

    # case: numpy Generator
    if isinstance(rng, np.random.Generator):
        np_rng = rng
        # сохраняем переданный генератор как есть и используем тот же seed для Torch
        seed = int(np_rng.bit_generator.state["state"]["state"] & 0xFFFFFFFF)  # type: ignore[index]
        t_gen = torch.Generator().manual_seed(seed) if torch is not None else None  # type: ignore[arg-type]
        return np_rng, t_gen, seed

    # case: torch Generator
    if torch is not None and isinstance(rng, torch.Generator):
        t_gen = rng  # type: ignore[assignment]
        seed = int(t_gen.initial_seed() & 0xFFFFFFFF)
        np_rng = np.random.default_rng(seed)
        return np_rng, t_gen, seed

    raise TypeError("rng must be int | numpy.Generator | torch.Generator | None")

# ────────────────────────────────────────────────────────────────────────────
#                               BaseSampler
# ────────────────────────────────────────────────────────────────────────────

class BaseSampler:
    """Абстрактный предок для всех сэмплеров.

    Дочерний класс **обязан** реализовать `sample(space, n)`.

    Минимальные требования к `space`:
        * `len(space)`            – число переменных (d)
        * `bounds()`              – ndarray (low, high) формы (d,)
        * `variables` **или** `_vars` – итерируемое с meta‑объектами
          (используется в `_to_dict`)
    """

    requires: str | None = None  # строка для docs – «pyDOE2», «torch», …

    def __init__(self, *, rng: RngLike | None = None):
        self._np_rng, self._torch_gen, self._seed = _resolve_rng(rng)
        self._idx = 0  # сколько точек уже выдано (для fast_forward)

    # ------------------------------------------------------------------ API
    def sample(self, space, n: int) -> List[PointDict]:  # noqa: D401, WPS110
        """Сгенерировать `n` новых точек.  Обязательно переопределить."""
        raise NotImplementedError

    # ---------------------------------------------------------------- helper
    @staticmethod
    def _to_dict(space, arr: np.ndarray) -> List[PointDict]:
        """(n,d) ndarray → List[dict] с учётом типов переменных.

        * Если `v.is_integer` → круглое до int.
        * Если `v.levels`     → индекс уровня по ближайшему целому.
        * Иначе float.
        """
        # допускаем две возможные структуры DesignSpace
        if hasattr(space, "_vars"):
            vars_  = list(space._vars.values())  # type: ignore[attr-defined]
            names  = list(space._vars.keys())
        else:
            vars_  = space.variables  # type: ignore[attr-defined]
            names  = [v.name for v in vars_]

        out: List[PointDict] = []
        for row in arr:
            p: PointDict = {}
            for name, val, var in zip(names, row, vars_):
                if getattr(var, "is_integer", False):
                    p[name] = int(round(val))
                elif getattr(var, "levels", None) is not None:
                    idx = int(np.clip(np.round(val), 0, len(var.levels) - 1))
                    p[name] = var.levels[idx]
                else:
                    p[name] = float(val)
            out.append(p)
        return out

    # ---------------------------------------------------------------- state
    def reset(self, rng: RngLike | None = None):
        """Сбросить внутренний счётчик + пересоздать RNG (полная перезагрузка)."""
        self.__init__(rng=rng)  # type: ignore[misc]

    def state_dict(self) -> Dict[str, Any]:
        return {"idx": self._idx, "seed": self._seed}

    def load_state_dict(self, d: Mapping[str, Any]):
        self._idx  = int(d["idx"])
        self._seed = int(d["seed"])
        self._np_rng, self._torch_gen, _ = _resolve_rng(self._seed)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        req = f", requires={self.requires}" if self.requires else ""
        return (
            f"{self.__class__.__name__}(seed={self._seed}, idx={self._idx}{req})"
        )

# ────────────────────────────────────────────────────────────────────────────
#                         SciPy QMC samplers
# ────────────────────────────────────────────────────────────────────────────

@register("sobol", "qmc_sobol")
class SobolSampler(BaseSampler):
    """Квази‑равномерная последовательность Соболя.

    *Поддерживает* fast‑forward, поэтому многократные вызовы возвращают
    продолжение без пересоздания объекта.
    """

    def __init__(self, *, scramble: bool = True, rng: RngLike | None = None):
        super().__init__(rng=rng)
        self.scramble = scramble
        self._engine: qmc.Sobol | None = None

    def sample(self, space, n: int) -> List[PointDict]:
        d = len(space)
        if self._engine is None:
            self._engine = qmc.Sobol(d, scramble=self.scramble, seed=self._seed)
            if self._idx:
                self._engine.fast_forward(self._idx)
        pts01 = self._engine.random(n)  # (n,d) в [0,1]
        self._idx += n
        lows, highs = space.bounds()
        return self._to_dict(space, lows + pts01 * (highs - lows))

@register("halton", "qmc_halton")
class HaltonSampler(BaseSampler):
    """Последовательность Халтон – устойчива при d > 20."""

    def __init__(self, *, scramble: bool = True, rng: RngLike | None = None):
        super().__init__(rng=rng)
        self.scramble = scramble
        self._engine: qmc.Halton | None = None

    def sample(self, space, n: int) -> List[PointDict]:
        d = len(space)
        if self._engine is None:
            self._engine = qmc.Halton(d, scramble=self.scramble, seed=self._seed)
            if self._idx:
                self._engine.fast_forward(self._idx)
        pts01 = self._engine.random(n)
        self._idx += n
        lows, highs = space.bounds()
        return self._to_dict(space, lows + pts01 * (highs - lows))

@register("lhs", "latin")
class LHSampler(BaseSampler):
    """Classic Latin Hypercube (без оптимизации расстояний)."""

    def sample(self, space, n: int) -> List[PointDict]:
        d = len(space)
        lhs = qmc.LatinHypercube(d, seed=self._seed + self._idx)
        pts01 = lhs.random(n)
        self._idx += n
        lows, highs = space.bounds()
        return self._to_dict(space, lows + pts01 * (highs - lows))

@register("lhs_maximin")
class LHSMaximinSampler(BaseSampler):
    """LHS + оптимизация критерия «maximin» (`scipy.stats.qmc.optimize_lhs`)."""

    def __init__(self, *, iterations: int = 50, rng: RngLike | None = None):
        super().__init__(rng=rng)
        self.iterations = iterations

    def sample(self, space, n: int) -> List[PointDict]:
        d = len(space)
        if d > 10 or n < 2 * d:
            warnings.warn(
                "optimize_lhs ненадежен при d>10 или n<2d – использую обычный LHS",
                RuntimeWarning,
            )
            return LHSampler(rng=self._seed + self._idx).sample(space, n)
        lhs = qmc.LatinHypercube(d, seed=self._seed + self._idx)
        pts01 = lhs.random(n)
        try:
            pts01 = qmc.optimize_lhs(pts01, criterion="maximin", iterations=self.iterations)
        except Exception as err:  # pragma: no cover
            warnings.warn(f"optimize_lhs failed: {err}; возвращаю исходный LHS", RuntimeWarning)
        self._idx += n
        lows, highs = space.bounds()
        return self._to_dict(space, lows + pts01 * (highs - lows))

@register("normal", "gauss")
class NormalSampler(BaseSampler):
    """Независимые нормальные распределения вокруг центра.

    *Радиус* Δ задаётся самим `DesignSpace` как (upper − lower)/2.
    ``k`` определяет ширину полосы: `σ = Δ / k`.
    По‑умолчанию `k=3` → 99.7 % точек внутри [lower, upper].
    """

    def __init__(self, *, k: float = 3.0, clip: bool = True, rng: RngLike | None = None):
        super().__init__(rng=rng)
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = float(k)
        self.clip = bool(clip)

    def sample(self, space, n: int) -> List[PointDict]:
        lows, highs = space.bounds()
        centers = (lows + highs) / 2
        sigmas  = (highs - lows) / (2 * self.k)
        Z = self._np_rng.standard_normal((n, len(space)))
        pts = centers + sigmas * Z
        if self.clip:
            pts = np.clip(pts, lows, highs)
        self._idx += n
        return self._to_dict(space, pts)

@register("factorial_full", "ff2k")
class FactorialFullSampler(BaseSampler):
    """Полный 2^d план (углы гиперкуба). Предупреждение при d>max_dim."""

    def __init__(
        self,
        *,
        max_dim: int = 10,
        rng: RngLike | None = None,
        shuffle: bool = False
    ):
        super().__init__(rng=rng)
        self.max_dim = max_dim
        self.shuffle = shuffle

    def sample(self, space, n: int | None = None) -> List[PointDict]:
        d = len(space)
        if d > self.max_dim:
            raise ValueError(f"Full factorial взрывается: 2^{d} точек – слишком много")

        # генерируем 2^d углов гиперкуба
        grid = np.indices((2,) * d).reshape(d, -1).T.astype(float)  # (2^d, d)
        if self.shuffle:
            self._np_rng.shuffle(grid)   # воспроизводимое перемешивание

        lows, highs = space.bounds()
        pts = lows + grid * (highs - lows)
        self._idx += len(pts)
        return self._to_dict(space, pts)

# ────────────────────────────────────────────────────────────────────────────
#                       Optional samplers – pyDOE2
# ────────────────────────────────────────────────────────────────────────────

_spec = importlib.util.find_spec("pyDOE2")
if _spec is not None:
    import pyDOE2 as pyd  # type: ignore

    @register("factorial_frac", "ff_fractional")
    class FractionalFactorialSampler(BaseSampler):
        """Дробный 2^k план (Resolution IV)."""

        requires = "pyDOE2"

        def __init__(self, *, gen: str | None = None, rng: RngLike | None = None):
            super().__init__(rng=rng)
            self.gen = gen or "a"  # по умолчанию генератор 'a b ab …'

        def sample(self, space, n: int | None = None) -> List[PointDict]:
            d = len(space)
            gen_str = self.gen if self.gen.strip() else "a" * d
            design = pyd.fracfact(gen_str)
            design = (design + 1) / 2
            lows, highs = space.bounds()
            pts = lows + design * (highs - lows)
            self._idx += len(pts)
            return self._to_dict(space, pts)

    @register("pb", "plackett_burman")
    class PlackettBurmanSampler(BaseSampler):
        """Plackett–Burman дизайн (N = 4·⌈d/4⌉)."""

        requires = "pyDOE2"

        def sample(self, space, n: int | None = None) -> List[PointDict]:
            d = len(space)
            design = pyd.pbdesign(d)
            design = (design + 1) / 2
            lows, highs = space.bounds()
            pts = lows + design * (highs - lows)
            self._idx += len(pts)
            return self._to_dict(space, pts)

    @register("ccd", "central_comp")
    class CCDSampler(BaseSampler):
        """Central Composite Design (alpha='orth')"""

        requires = "pyDOE2"

        def __init__(
                self,
                *,
                alpha: str = "orth",
                center: Tuple[int, int] = (4, 4),
                rng: RngLike | None = None,
        ):
            super().__init__(rng=rng)  # захватываем seed
            self.alpha = alpha
            self.center = center

        def sample(self, space, n: int | None = None) -> List[PointDict]:
            d = len(space)
            design = pyd.ccdesign(d, alpha=self.alpha, center=self.center, seed=self._seed)
            design = (design + 1) / 2
            lows, highs = space.bounds()
            pts = lows + design * (highs - lows)
            self._idx += len(pts)
            return self._to_dict(space, pts)

    @register("box_behnken")
    class BoxBehnkenSampler(BaseSampler):
        """Box–Behnken план для локальной Response Surface Methodology."""

        requires = "pyDOE2"

        def __init__(self, *, rng: RngLike | None = None):
            super().__init__(rng=rng)

        def sample(self, space, n: int | None = None) -> List[PointDict]:
            d = len(space)
            design = pyd.bbdesign(d, seed=self._seed)
            design = (design + 1) / 2
            lows, highs = space.bounds()
            pts = lows + design * (highs - lows)
            self._idx += len(pts)
            return self._to_dict(space, pts)
else:

    def _placeholder(alias: str, dep: str = "pyDOE2"):
        @register(alias)
        class _Missing(BaseSampler):
            requires = dep

            def sample(self, *_, **__):  # noqa: D401
                raise ImportError(
                    f"Sampler '{alias}' требует внешнюю зависимость: {dep}. Установите: pip install mwlab[doe]"
                )

    for _alias in ("factorial_frac", "pb", "ccd", "box_behnken"):
        _placeholder(_alias)

# ────────────────────────────────────────────────────────────────────────────
#                           __all__ для IDE/экспорта
# ────────────────────────────────────────────────────────────────────────────
__all__ = [
    "BaseSampler",
    "get_sampler",
    *list(_SAMPLER_REGISTRY.keys()),
]


