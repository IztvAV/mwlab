#mwlab/filters/topologies.py
"""
MWLab · filters · topologies
===========================
Описание **топологий** (графов связей) СВЧ‑фильтров и мультиплексеров и
реестр готовых шаблонов.

Основные возможности
--------------------
* **Topology** – неизменяемое описание: количество резонаторов (`order`),
  портов (`ports`) и кортеж ненулевых связей `links` (верхний треугольник,
  1‑based).
* **Registry‑pattern**: декоратор :pyfunc:`register_topology` регистрирует
  шаблоны, которые затем получаются через :pyfunc:`get_topology` или
  перечисляются в :pyfunc:`list_topologies`.
* **Сериализация** – `to_dict` / `from_dict` для сохранения в JSON/HDF5.
* **Утилиты**: диапазоны `res_indices`, `port_indices`, быстрая
  `adjacency_matrix()` (NumPy‑bool), экспорт `to_networkx()`.
* **Расширенные проверки**: :pyfunc:`Topology.validate_mvals` теперь
  поддерживает режим `strict=False`, когда допускаются нулевые (пропущенные)
  коэффициенты M.
* Предустановленные шаблоны: **folded**, **transversal** (aka *canonical*).

Быстрый старт (пример)
----------------------
```python
from mwlab.filters.topologies import (
    Topology, get_topology, list_topologies, register_topology
)

# 1) ручная топология фильтра‑цепочки (5 резонаторов + 2 порта)
topo = Topology(
    order=5,
    ports=2,
    links=[                # верхний треугольник
        (1, 2), (2, 3), (3, 4), (4, 5),   # соседние
        (1, 6), (5, 7)                    # связи с портами
    ],
    name="chain-5"
)
print(topo)
# Topology(name='chain-5', order=5, ports=2, links=6)

# 2) готовый шаблон «folded»
topo_f = get_topology("folded", order=6)
print(topo_f.links[:4])
# ((1, 2), (1, 7), (1, 8), (2, 3))

# 3) список зарегистрированных шаблонов
print(list_topologies())
# ['canonical', 'folded', 'transversal']
```
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
#                              Topology‑error
# -----------------------------------------------------------------------------
class TopologyError(ValueError):
    """Ошибки, связанные с построением или валидацией :class:`Topology`."""


# -----------------------------------------------------------------------------
#                                   Topology
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Topology:
    """Неизменяемое описание графа связей.

    Параметры
    ----------
    order : int
        Количество **резонаторов** (узлов).
    ports : int
        Количество **портов**.
    links : Sequence[Tuple[int,int]]
        Ненулевые элементы **верхнего** треугольника матрицы связи в 1‑based
        обозначениях ``(i,j)``, где ``1 ≤ i < j ≤ order+ports``.
        Дубликаты и зеркальные пары автоматически удаляются.
    name : str, optional
        Человекочитаемый идентификатор (не влияет на сравнение/хеширование).

    Примечание
    -----
    * Диагональ и знаки M_{ij} к топологии отношения не имеют.
    * Порты пронумерованы последовательно:
      ``order+1 … order+ports``.
    """

    order: int
    ports: int
    links: Sequence[Tuple[int, int]] = field(repr=False)
    name: str | None = None

    # ------------------------------------------------------------------ init
    def __post_init__(self):
        k = self.order + self.ports  # полный размер матрицы
        if self.order <= 0 or self.ports <= 0:
            raise TopologyError("order и ports должны быть > 0")

        clean: List[Tuple[int, int]] = []
        for i, j in self.links:
            i, j = (j, i) if j < i else (i, j)           # верхний треугольник
            if not (1 <= i < j <= k):
                raise TopologyError(
                    f"link ({i},{j}) выходит за предел 1…{k} или i ≥ j"
                )
            clean.append((i, j))

        # удаляем дубли, сортируем, сохраняем неизменяемый tuple
        object.__setattr__(self, "links", tuple(sorted(set(clean))))

        #проверка: каждый порт должен быть связан хотя бы с одним узлом
        for p in self.port_indices:
            if not any(p in e for e in self.links):
                raise TopologyError(f"порт {p} не имеет ни одной связи")

    # ---------------------------------------------------------------- props / helpers
    @property
    def size(self) -> int:
        """Полный порядок расширенной матрицы ``K = order + ports``."""
        return self.order + self.ports

    # — новые удобные диапазоны —
    @property
    def res_indices(self) -> range:
        """Диапазон индексов резонаторов (1‑based)."""
        return range(1, self.order + 1)

    @property
    def port_indices(self) -> range:
        """Диапазон индексов портов (1‑based)."""
        return range(self.order + 1, self.size + 1)

    # ---------------------------------------------------------------- списки / матрицы
    def adjacency_matrix(self) -> np.ndarray:
        """Возвращает логическую матрицу смежности ``(K, K)`` (верх+низ)."""
        K = self.size
        adj = np.zeros((K, K), dtype=bool)
        rows, cols = zip(*self.links) if self.links else ([], [])
        adj[np.array(rows) - 1, np.array(cols) - 1] = True
        # отражаем нижний треугольник
        return adj | adj.T

    # ---------------------------------------------------------------- сериализация
    def to_dict(self) -> Dict[str, object]:
        """Словарь, пригодный для JSON/YAML/HDF5.«links» — список списков."""
        return {
            "order": self.order,
            "ports": self.ports,
            "links": [list(e) for e in self.links],
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "Topology":
        return cls(
            order=int(d["order"]),
            ports=int(d["ports"]),
            links=[tuple(e) for e in d["links"]],
            name=d.get("name"),
        )

    # ---------------------------------------------------------------- validation
    def validate_mvals(self, mvals: Mapping[str, float], *, strict: bool = True) -> None:
        """Проверка набора коэффициентов ``M{i}_{j}`` на согласие с топологией.

        Параметры
        ----------
        mvals : Mapping[str, float]
            Словарь ненулевых коэффициентов матрицы связи.
        strict : bool, default **True**
            * **True**  — требуютcя **все** связи из :pyattr:`links`.
            * **False** — проверяем только отсутствие «лишних» M‑ключей.
        """
        # разбираем ключи вида "M1_4"
        present: set[Tuple[int, int]] = set()
        for key in mvals:
            if not key.startswith("M"):
                continue
            try:
                i, j = map(int, key[1:].split("_", 1))
            except Exception:
                raise TopologyError(f"неверный ключ матрицы связи: {key!r}")
            i, j = (j, i) if i > j else (i, j)
            if i == j:
                continue  # диагональ игнорируем
            present.add((i, j))

        required = set(self.links)
        extra = present - required
        missing = required - present if strict else set()
        if missing or extra:
            raise TopologyError(
                "несоответствие Topology: "
                + (f"не хватает {sorted(missing)}; " if missing else "")
                + (f"лишние {sorted(extra)}" if extra else "")
            )

    # ---------------------------------------------------------------- hash / eq / len / repr
    def __hash__(self) -> int:  # делает объект пригодным как ключ словаря
        return hash((self.order, self.ports, frozenset(self.links)))

    def __eq__(self, other) -> bool:  # сравниваем по множеству связей
        if not isinstance(other, Topology):
            return NotImplemented
        return (
            self.order == other.order
            and self.ports == other.ports
            and set(self.links) == set(other.links)
        )

    def __len__(self) -> int:
        """Количество ненулевых M-элементов **верхнего** треугольника."""
        return len(self.links)

    def __repr__(self) -> str:  # noqa: D401
        nm = f"name='{self.name}', " if self.name else ""
        return f"Topology({nm}order={self.order}, ports={self.ports}, links={len(self)})"

    # ----------------------------- export helpers
    def to_networkx(self):
        """
        Конвертация в :class:`networkx.Graph`.
        Узлы:
            * 1 … ``order``               – резонаторы;
            * ``order+1`` … ``order+ports`` – порты (атрибут ``type='port'``).
        Edges - неориентированные, только *i < j*.
        """
        try:
            import networkx as nx
        except ModuleNotFoundError as err:  # pragma: no cover
            raise ImportError("Метод to_networkx() требует пакет 'networkx'.") from err

        g = nx.Graph()
        for i in self.res_indices:
            g.add_node(i, type="resonator")
        for p in self.port_indices:
            g.add_node(p, type="port")
        g.add_edges_from(self.links)
        return g


# -----------------------------------------------------------------------------
#                      Registry helpers (alias‑pattern)
# -----------------------------------------------------------------------------
_TOPO_REGISTRY: Dict[str, Callable[..., Topology]] = {}


def register_topology(name: str, *aliases: str):
    """Декоратор для регистрации фабрики‑шаблона топологии."""

    def _wrap(func):
        for key in (name, *aliases):
            if key in _TOPO_REGISTRY:
                raise KeyError(f"Topology alias {key!r} уже зарегистрирован")
            _TOPO_REGISTRY[key] = func
        return func

    return _wrap


def get_topology(name: str, /, *args, **kwargs) -> Topology:
    """Возвращает топологию по alias‑у.  См. :func:`list_topologies`."""
    try:
        factory = _TOPO_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Топология {name!r} не найдена. Доступно: {list_topologies()}"
        ) from exc
    return factory(*args, **kwargs)


def list_topologies() -> List[str]:
    """Отсортированный список зарегистрированных alias‑ов."""
    return sorted(_TOPO_REGISTRY)


# -----------------------------------------------------------------------------
#                     Библиотека готовых шаблонов
# -----------------------------------------------------------------------------
@register_topology("folded")
def folded(order: int, *, ports: int = 2, name: str | None = None) -> Topology:
    """«Folded» (складная) цепочка резонаторов + угловая связь.

    Связи:
    • цепочка      1‑2‑…‑n
    • угловая      1‑n
    • порт → R1,   Rn → порт₂
    """
    if order < 3:
        raise TopologyError("folded: order должен быть ≥ 3")
    if ports != 2:
        raise TopologyError("folded: поддерживаются только 2 порта")

    chain = [(i, i + 1) for i in range(1, order)] # соседние
    corner = [(1, order)]                         # 1-n
    p_in = (1, order + 1)                         # входной порт
    p_out = (order, order + 2)                    # выходной порт
    links = chain + corner + [p_in, p_out]
    return Topology(order, ports, links=links, name=name or "folded")


@register_topology("transversal", "canonical")
def transversal(order: int, *, ports: int = 2, name: str | None = None) -> Topology:
    """
    «Transversal»/«Canonical» – каждый резонатор связан с одним из портов.
    Для 2-портового фильтра получаем связи::

        (1-p₁), (2-p₂), (3-p₁), (4-p₂), …

    где ``p₁ = order+1``, ``p₂ = order+2`` – индексы портов.
    Требования: ``ports == 2`` и ``order >= 2``.
    """
    if ports != 2:
        raise TopologyError("transversal: поддерживается только 2 порта")
    if order < 2:
        raise TopologyError("transversal: order должен быть ≥ 2")

    p_in, p_out = order + 1, order + 2
    links = [
        (idx, p_in if idx % 2 else p_out)
        for idx in range(1, order + 1)
    ]
    return Topology(order, ports, links=links, name=name or "transversal")


