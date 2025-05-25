# mwlab/opt/design/space.py
"""
MWLab · opt · design · space
============================
Описание **project design‑space** – набора переменных, в которых мы будем
строить DOE, обучать суррогат и запускать оптимизацию.

Ключевые сущности:
------------------
* `BaseVar` – абстрактный класс + четыре реальных типа переменных
  `ContinuousVar`, `IntegerVar`, `OrdinalVar`, `CategoricalVar`.
* `DesignSpace` – контейнер этих переменных, умеет:
  • нормировать / денормировать точки  ↔  [0,1]^d,
  • генерировать DOE через плагины‑сэмплеры (`space.sample(n, "sobol")`),
  • хранить и валидировать *constraints* (неравенства/маски),
  • (де)сериализоваться в JSON/YAML.

Поддерживаются следующие способы построения пространства параметров:
1. **Явный** – через словарь `name: ContinuousVar`;
2. **Метод `from_center_delta`** – быстрый билд пространства, имея
   словарь центров и общий δ (в абсолютных или относительных единицах).
3. **YAML/JSON** – для CLI‑runner‑а (методы *to_yaml* / *from_file*).
Новая функциональность по сравнению с первоначальным MVP

"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# ────────────────────────────────────────────────────────────────────────────
#              Помощники типов (для JSON‑friendly структуры)
# ────────────────────────────────────────────────────────────────────────────
JSONable = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


# ────────────────────────────────────────────────────────────────────────────
#              1. BaseVar + конкретные типы переменных
# ────────────────────────────────────────────────────────────────────────────

class BaseVar:
    """Абстрактная проектная переменная.

    *Внутренне* все описывается числом: даже категориальные уровни
    маппятся на индексы `0..k-1`.  Каждый Var дает переход:

    * `to_unit(z)`   – перевод z∈[0,1] → физическая величина
    * `from_unit(x)` – обратное преобразование
    """

    name: str = ""
    unit: str = ""
    is_integer: bool = False           # Hint для сэмплеров/кодеков
    levels: List[str] | None = None    # Для cat/ordinal

    # --- обязательные методы ---------------------------------------------
    def bounds(self) -> Tuple[float, float]:
        raise NotImplementedError

    def to_unit(self, z: float):  # noqa: D401
        raise NotImplementedError

    def from_unit(self, x) -> float:  # noqa: D401
        raise NotImplementedError

    # --- (де)сериализация --------------------------------------------------
    def to_json(self) -> Dict[str, JSONable]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, d: Mapping[str, Any]):
        raise NotImplementedError


# ------------------------------------------------------------------
# 1.1 ContinuousVar – непрерывный диапазон
# ------------------------------------------------------------------

@dataclass
class ContinuousVar(BaseVar):
    lower: float | None = None
    upper: float | None = None
    center: float | None = None
    delta: float | None = None
    unit: str = ""

    # ---------------- post‑init проверка
    def __post_init__(self):
        if self.lower is None or self.upper is None:
            if self.center is None or self.delta is None:
                raise ValueError("ContinuousVar: задайте либо (lower,upper), либо (center,delta)")
            self.lower = float(self.center - self.delta)
            self.upper = float(self.center + self.delta)
        if self.upper <= self.lower:
            raise ValueError("upper must be > lower")

    # ---------------- реализация API
    def bounds(self) -> Tuple[float, float]:
        return float(self.lower), float(self.upper)

    def to_unit(self, z: float) -> float:
        lo, hi = self.bounds()
        return lo + z * (hi - lo)

    def from_unit(self, x: float) -> float:
        lo, hi = self.bounds()
        return (float(x) - lo) / (hi - lo)

    # ---------------- (де)сериализация
    def to_json(self) -> Dict[str, JSONable]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v not in (None, "")}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]):
        d = dict(d)
        d.pop("type", None)
        return cls(**d)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# 1.2 IntegerVar – целочисленный диапазон с шагом
# ------------------------------------------------------------------

@dataclass
class IntegerVar(BaseVar):
    lower: int
    upper: int
    step: int = 1
    unit: str = ""
    is_integer: bool = True

    def bounds(self) -> Tuple[float, float]:
        return float(self.lower), float(self.upper)

    def to_unit(self, z: float) -> int:
        lo, hi = self.bounds()
        val = lo + z * (hi - lo)
        # аккуратно на сетку step – через nearest integer
        stepped = round(val / self.step) * self.step
        return int(np.clip(stepped, self.lower, self.upper))

    def from_unit(self, x: int) -> float:
        lo, hi = self.bounds()
        return (int(x) - lo) / (hi - lo)

    def to_json(self) -> Dict[str, JSONable]:
        return {
            "type": "integer",
            "lower": self.lower,
            "upper": self.upper,
            "step": self.step,
            "unit": self.unit,
        }

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d.pop("type", None)
        return cls(lower=int(d["lower"]), upper=int(d["upper"]), step=int(d.get("step", 1)), unit=d.get("unit", ""))


# ------------------------------------------------------------------
# 1.3 OrdinalVar – упорядоченные уровни (low<mid<high)
# ------------------------------------------------------------------

@dataclass
class OrdinalVar(BaseVar):
    levels: List[str]

    def bounds(self) -> Tuple[float, float]:
        return 0.0, float(len(self.levels) - 1)

    def _idx(self, z: float) -> int:  # helper: безопасно в границах
        return int(np.clip(round(z * (len(self.levels) - 1)), 0, len(self.levels) - 1))

    def to_unit(self, z: float) -> str:
        return self.levels[self._idx(z)]

    def from_unit(self, x: str) -> float:
        idx = self.levels.index(x)
        return idx / (len(self.levels) - 1)

    def to_json(self) -> Dict[str, JSONable]:
        return {"type": "ordinal", "levels": self.levels}

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d.pop("type", None)
        return cls(levels=list(d["levels"]))


# ------------------------------------------------------------------
# 1.4 CategoricalVar – без упорядочивания
# ------------------------------------------------------------------

@dataclass
class CategoricalVar(BaseVar):
    levels: List[str]

    def bounds(self) -> Tuple[float, float]:
        return 0.0, float(len(self.levels) - 1)

    def _idx(self, z: float) -> int:
        return int(np.clip(round(z * (len(self.levels) - 1)), 0, len(self.levels) - 1))

    def to_unit(self, z: float) -> str:
        return self.levels[self._idx(z)]

    def from_unit(self, x: str) -> float:
        idx = self.levels.index(x)
        return idx / (len(self.levels) - 1)

    def to_json(self) -> Dict[str, JSONable]:
        return {"type": "categorical", "levels": self.levels}

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d.pop("type", None)
        return cls(levels=list(d["levels"]))


# ────────────────────────────────────────────────────────────────────────────
#              2. DesignSpace – контейнер переменных + сервисы
# ────────────────────────────────────────────────────────────────────────────

class DesignSpace:
    """Коллекция переменных.

    Основные возможности:
    • bounds()               – numpy‑векторы low/high
    • normalize/denormalize  – dict <-> z∈[0,1]^d
    • sample()               – через plug‑in сэмплеры
    • constraints            – доп. булевы функции над точкой
    • from_center_delta()    – быстрый билд от центра и ±δ.
    """

    # ------------------------------------------------------------ init
    def __init__(self, vars: Mapping[str, BaseVar]):
        if not vars:
            raise ValueError("DesignSpace: пустой список переменных")
        self._vars: Dict[str, BaseVar] = dict(vars)
        self._names = list(self._vars.keys())
        lows, highs = zip(*(v.bounds() for v in self._vars.values()))
        self._lows = np.asarray(lows, dtype=np.float64)
        self._highs = np.asarray(highs, dtype=np.float64)
        self._constraints: List[Callable[[Dict[str, Any]], bool]] = []

    # ---------------- mapping‑helpers
    def __len__(self):
        return len(self._vars)

    def __getitem__(self, name: str) -> BaseVar:  # new ⬅
        """Позволяет обращаться `space["x"]` → объект переменной."""
        return self._vars[name]

    def __iter__(self):  # new ⬅
        return iter(self._vars)

    def __contains__(self, name: str):  # new ⬅
        return name in self._vars

    # ---------------- stats helpers

    def names(self) -> List[str]:  # noqa: D401
        return list(self._names)

    def bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._lows.copy(), self._highs.copy()

    # ---------------- dict/vector conversions
    def vector(self, point: Mapping[str, Any]) -> NDArray[np.float64]:
        return np.fromiter((point[k] for k in self._names), dtype=np.float64)

    def dict(self, vec: Sequence[Any]) -> Dict[str, Any]:
        return {k: v for k, v in zip(self._names, vec)}

    # ---------------- normalize/denormalize
    def normalize(self, point: Mapping[str, Any]) -> NDArray[np.float64]:
        z_vals = [self._vars[k].from_unit(point[k]) for k in self._names]
        return np.asarray(z_vals, dtype=np.float64)

    def denormalize(self, z: Sequence[float]) -> Dict[str, Any]:
        if len(z) != len(self):
            raise ValueError("dimension mismatch")
        return {k: self._vars[k].to_unit(float(val)) for k, val in zip(self._names, z)}

    # ---------------- constraints
    def add_constraint(self, func: Callable[[Dict[str, Any]], bool]):
        self._constraints.append(func)

    def _is_valid(self, p: Dict[str, Any]):
        return all(f(p) for f in self._constraints)

    # ---------------- sampling
    def sample(self, n: int, *, sampler: str | "BaseSampler" = "sobol", reject_invalid: bool = True,
               max_attempts: int = 1000, **kw) -> List[Dict[str, Any]]:
        from .samplers import get_sampler  # локальный импорт – избегаем циклов

        smp = get_sampler(sampler, **kw) if isinstance(sampler, str) else sampler
        if not self._constraints or not reject_invalid:
            return smp.sample(self, n)
        pts: List[Dict[str, Any]] = []
        attempts = 0
        while len(pts) < n and attempts < max_attempts:
            need = n - len(pts)  # ← сколько еще точек нужно
            cand = smp.sample(self, need)
            pts.extend(p for p in cand if self._is_valid(p))
            attempts += 1
        if len(pts) < n:
            raise RuntimeError("DesignSpace.sample: не удалось набрать валидных точек – увеличьте max_attempts")
        return pts[:n]

    # ---------------- from_center_delta helper
    @classmethod
    def from_center_delta(
        cls,
        centers: Mapping[str, float],
        *,
        delta: float | Mapping[str, float],
        mode: str = "abs",  # "abs" | "rel" (% от центра)
        unit: str = "",
    ) -> "DesignSpace":
        """Быстрый билдер: все переменные – ContinuousVar.

        * `delta` – либо число для всех, либо dict[key→δ].
        * `mode="rel"` – интерпретируем δ как долю: 0.1 → ±10 %.
        """
        vars: Dict[str, BaseVar] = {}
        for k, center in centers.items():
            d = delta[k] if isinstance(delta, Mapping) else delta
            d = float(d)
            if mode == "rel":
                d = d * abs(center)
            vars[k] = ContinuousVar(center=center, delta=d, unit=unit)
        return cls(vars)

    # ---------------- I/O JSON/YAML
    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for n, v in self._vars.items():
            d = v.to_json()
            d["type"] = v.__class__.__name__.replace("Var", "").lower()
            payload[n] = d
        return payload

    def to_yaml(self) -> str:
        import yaml
        return yaml.dump({"variables": self.to_json()}, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str):
        import yaml
        return cls.from_dict(yaml.safe_load(text)["variables"])

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_yaml(path.read_text())
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        vars: Dict[str, BaseVar] = {}
        for name, cfg in d.items():
            vtype = cfg.get("type", "continuous")
            if vtype == "continuous":
                vars[name] = ContinuousVar.from_json(cfg)
            elif vtype == "integer":
                vars[name] = IntegerVar.from_json(cfg)
            elif vtype == "ordinal":
                vars[name] = OrdinalVar.from_json(cfg)
            elif vtype == "categorical":
                vars[name] = CategoricalVar.from_json(cfg)
            else:
                raise ValueError(f"Unknown variable type '{vtype}' for '{name}'")
        return cls(vars)

    # ---------------- repr
    def __repr__(self):  # pragma: no cover
        inner = ", ".join(f"{k}…" for k in self._names)
        return f"DesignSpace({inner})"


# -----------------------------------------------------------------------------
# __all__ – публичный интерфейс
# -----------------------------------------------------------------------------
__all__ = [
    "ContinuousVar",
    "IntegerVar",
    "OrdinalVar",
    "CategoricalVar",
    "DesignSpace",
]
