# mwlab/filters/cm_param_tying.py
# -*- coding: utf-8 -*-
"""
cm_param_tying.py — связывание (tie) параметров матрицы связи по симметриям
============================================================================

Зачем нужен этот модуль
-----------------------
На практике часто проектируют **симметричные 2-портовые СВЧ-фильтры**, у которых
(нерасширенная) матрица связи симметрична не только относительно главной диагонали,
но и относительно **побочной диагонали** (зеркальная/реверсивная симметрия):
первый резонатор эквивалентен последнему, связь 1–2 равна связи (n-1)–n и т.д.

В архитектуре MWLab уже есть:
* Topology       — задаёт *структуру* (какие связи потенциально ненулевые),
* ParamSchema    — задаёт *полный* вектор параметров в каноническом порядке
                  и умеет собирать (M_real, qu, phase_a, phase_b) для solve_sparams.

Однако симметрия — это не новая топология и не новая физика ядра.
Симметрия — это **редуцированная параметризация**: некоторые параметры должны быть
одинаковыми. Поэтому корректный уровень реализации — *слой над ParamSchema*:
"Parameter tying".

Идея
----
Мы строим объект TiedParamSchema, который:
1) хранит исходную (полную) ParamSchema,
2) задаёт разбиение полного набора ключей на группы равенств,
3) даёт быстрое отображение:
     free_vec  -> full_vec
   через индексный gather:
     full_vec = free_vec[..., full_to_free]

Такой подход:
* сохраняет совместимость со всем существующим кодом (assemble/solve_sparams),
* работает батчево и быстро (без Python-циклов по частотам/матрицам),
* легко расширяется на любые перестановочные симметрии (не только зеркальную).

Термины
-------
* full-space  — пространство полного вектора параметров ParamSchema (L_full).
* free-space  — пространство "свободных" параметров после tie (L_free <= L_full).
* tie_ports   — для 2-портовой зеркальной симметрии обычно означает обмен портов
                P1 <-> P2 при отражении (это именно "привязка портов" к симметрии).
* tie_qu      — связывать ли qu_i зеркально (обычно да, если qu векторные).
* tie_phases  — связывать ли phase_a/phase_b согласно перестановке портов.

Важное ограничение (осознанно)
------------------------------
TiedParamSchema не пытается "лечить" топологию.
Если топология не инвариантна под заданной перестановкой узлов, tie построить нельзя:
это почти всегда означает, что симметрия физически не применима к данному графу связей.

Пример использования
--------------------
>>> from mwlab.filters.topologies import get_topology
>>> from mwlab.filters.cm_param_tying import tied_mirror_schema_2port
>>>
>>> topo = get_topology("folded", order=6)  # 2-port
>>> tied = tied_mirror_schema_2port(topo, include_qu="vec", include_phase=("a",),
...                                tie_ports=True, tie_qu=True, tie_phases=False)
>>> print(tied.free_size, tied.full_size)
>>> # Далее:
>>> # vec_free -> vec_full -> schema.assemble(vec_full) -> solve_sparams(...)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Literal

import torch

from .topologies import Topology
from .cm_schema import ParamSchema, IncludeQU
from .cm_core import DT_R


# ────────────────────────────────────────────────────────────────────────────
#                                   Ошибки
# ────────────────────────────────────────────────────────────────────────────

class SymmetryError(ValueError):
    """Ошибка, связанная с симметрией/перестановкой/невозможностью tie."""


# ────────────────────────────────────────────────────────────────────────────
#                          Вспомогательные парсеры
# ────────────────────────────────────────────────────────────────────────────

def _parse_m_key(tag: str) -> Tuple[int, int]:
    """
    Парсер ключей вида "M<i>_<j>" (1-based).
    Возвращает (min(i,j), max(i,j)). Допускает i==j.

    Важно: здесь мы намеренно НЕ импортируем parse_m_key из mwlab.filters.cm,
    чтобы избежать потенциальных циклических импортов.
    """
    if not isinstance(tag, str) or not tag.startswith("M"):
        raise ValueError(f"Ожидался ключ 'M<i>_<j>', получено {tag!r}")
    try:
        i_str, j_str = tag[1:].split("_", 1)
        i, j = int(i_str), int(j_str)
    except Exception as exc:
        raise ValueError(f"Неверный ключ матрицы связи: {tag!r}") from exc
    if i <= 0 or j <= 0:
        raise ValueError(f"Индексы в {tag!r} должны быть положительными")
    return (i, j) if i <= j else (j, i)


# ────────────────────────────────────────────────────────────────────────────
#                        Перестановка узлов (NodePermutation)
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class NodePermutation:
    """
    Перестановка узлов расширенной матрицы (резонаторы + порты).

    mapping — кортеж длиной K, 1-based:
        mapping[i-1] = pi(i)

    То есть i ∈ [1..K] переходит в pi(i) ∈ [1..K].

    Зачем:
    ------
    Для tie мы хотим сказать: параметр, соответствующий связи (i,j), должен быть равен
    параметру связи (pi(i), pi(j)) (с нормализацией порядка i<j).
    """
    size: int
    mapping: Tuple[int, ...]  # 1-based values, length == size

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("NodePermutation.size должен быть > 0")
        if len(self.mapping) != self.size:
            raise ValueError("NodePermutation.mapping: длина должна совпадать с size")
        # Проверяем, что mapping — перестановка 1..K
        vals = list(self.mapping)
        if sorted(vals) != list(range(1, self.size + 1)):
            raise ValueError("NodePermutation.mapping должна быть перестановкой 1..K")

    def apply(self, i: int) -> int:
        """Применить перестановку к индексу i (1-based)."""
        if not (1 <= i <= self.size):
            raise ValueError(f"apply(): i={i} вне диапазона 1..{self.size}")
        return self.mapping[i - 1]

    def apply_pair(self, i: int, j: int) -> Tuple[int, int]:
        """Применить перестановку к паре (i,j) и вернуть нормализованную (min,max)."""
        a = self.apply(i)
        b = self.apply(j)
        return (a, b) if a <= b else (b, a)


def mirror_perm_2port(order: int, *, tie_ports: bool = True) -> NodePermutation:
    """
    Классическая зеркальная перестановка для 2-портового фильтра в каноническом порядке TAIL:
        узлы: [R1..Rn, P1, P2]  (1-based)

    Резонаторы разворачиваются:
        Ri -> R(n+1-i)

    Порты:
        если tie_ports=True  -> P1 <-> P2
        если tie_ports=False -> P1, P2 остаются на местах

    Возвращает NodePermutation для K = order + 2.
    """
    if order <= 0:
        raise ValueError("mirror_perm_2port: order должен быть > 0")
    K = order + 2
    mapping = [0] * K

    # Резонаторы 1..order
    for i in range(1, order + 1):
        mapping[i - 1] = (order + 1 - i)

    # Порты order+1, order+2
    p1 = order + 1
    p2 = order + 2
    if tie_ports:
        mapping[p1 - 1] = p2
        mapping[p2 - 1] = p1
    else:
        mapping[p1 - 1] = p1
        mapping[p2 - 1] = p2

    return NodePermutation(size=K, mapping=tuple(mapping))


# ────────────────────────────────────────────────────────────────────────────
#                       Проверка инвариантности топологии
# ────────────────────────────────────────────────────────────────────────────

def validate_topology_invariant(topo: Topology, perm: NodePermutation) -> None:
    """
    Проверяет, что топология (набор связей topo.links) инвариантна относительно perm.

    То есть для каждого (i,j) ∈ topo.links должно существовать (pi(i), pi(j)) ∈ topo.links.

    Почему это важно:
    -----------------
    ParamSchema строит блок M_off строго по topo.links. Если связь после отражения
    отсутствует в topo.links, значит "симметричный параметр" не представлен в схеме,
    и корректный tie невозможен без изменения топологии.
    """
    if topo.size != perm.size:
        raise SymmetryError(
            f"validate_topology_invariant: topo.size={topo.size} != perm.size={perm.size}"
        )

    # Нормализуем пары (min,max), чтобы не зависеть от того,
    # как именно topo.links хранит ориентацию ребра.
    links = {(i, j) if i <= j else (j, i) for (i, j) in topo.links}
    for (i, j) in links:
        a, b = perm.apply_pair(i, j)
        if (a, b) not in links:
            raise SymmetryError(
                "Topology не инвариантна относительно заданной перестановки: "
                f"({i},{j}) -> ({a},{b}) отсутствует в topo.links"
            )


# ────────────────────────────────────────────────────────────────────────────
#                         Union-Find (DSU) для групп
# ────────────────────────────────────────────────────────────────────────────

class _DSU:
    """Классический DSU/Union-Find для построения классов эквивалентности индексов."""
    __slots__ = ("p", "r")

    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        # union by rank
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ────────────────────────────────────────────────────────────────────────────
#                                TiedParamSchema
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TiedParamSchema:
    """
    Редуцированная параметризация поверх ParamSchema.

    Атрибуты
    --------
    schema : ParamSchema
        Полная схема.
    free_keys : tuple[str,...]
        Репрезентанты групп (ключи в *свободном* пространстве).
        По умолчанию репрезентант = ключ с минимальным индексом в полной схеме.
    key_to_free : dict[str,int]
        Отображение любого ключа полной схемы -> индекс свободного параметра.
    full_to_free : torch.LongTensor
        Тензор формы (L_full,), где full_to_free[i] = free_index для i-го full-параметра.
        Используется для быстрого разворачивания:
            full_vec = free_vec.index_select(-1, full_to_free)
    rep_full_idx : torch.LongTensor
        Тензор формы (L_free,), индексы репрезентантов в full-векторе.
        Используется для reduce:
            free_vec = full_vec.index_select(-1, rep_full_idx)

    Замечание о скорости
    -------------------
    Чтобы не делать .to(device) для индексных тензоров на каждом вызове, внутри есть
    небольшой кэш индексов по device. Это полезно при больших батчах/частотных сетках.
    """
    schema: ParamSchema
    free_keys: Tuple[str, ...]
    key_to_free: Dict[str, int]
    full_to_free: torch.Tensor   # LongTensor, CPU by default
    rep_full_idx: torch.Tensor   # LongTensor, CPU by default
    groups: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple, repr=False)

    # внутренний mutable-кэш (словарь можно мутировать, даже если dataclass frozen)
    _idx_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # ------------------------- размеры
    @property
    def full_size(self) -> int:
        """Длина полного вектора ParamSchema."""
        return self.schema.size

    @property
    def free_size(self) -> int:
        """Длина свободного (редуцированного) вектора."""
        return len(self.free_keys)

    # ------------------------- внутренний хелпер: индексы на нужном device
    def _get_index_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вернуть (full_to_free_dev, rep_full_idx_dev) на заданном device.
        Кэшируется по строковому имени device.
        """
        key = str(device)
        hit = self._idx_cache.get(key)
        if hit is not None:
            return hit
        f2f = self.full_to_free.to(device=device)
        rep = self.rep_full_idx.to(device=device)
        self._idx_cache[key] = (f2f, rep)
        return f2f, rep

    # ------------------------- преобразования
    def expand(self, vec_free: torch.Tensor) -> torch.Tensor:
        """
        Разворачивает свободный вектор в полный:
            (…, L_free) -> (…, L_full)

        Реализация через index_select по последней оси.
        """
        if vec_free.shape[-1] != self.free_size:
            raise ValueError(
                f"expand: последний размер vec_free должен быть {self.free_size}, "
                f"получено {vec_free.shape[-1]}"
            )
        device = vec_free.device
        full_to_free_dev, _ = self._get_index_tensors(device)
        # full_vec[..., i_full] = vec_free[..., full_to_free[i_full]]
        return vec_free.index_select(dim=-1, index=full_to_free_dev)

    def reduce(self, vec_full: torch.Tensor) -> torch.Tensor:
        """
        Сводит полный вектор к свободному, беря репрезентанты групп:
            (…, L_full) -> (…, L_free)

        Используется для:
        - отладки,
        - преобразования из полного пространства в свободное (если нужно),
        - или когда вы хотите получить "канонический" free-вектор из full-вектора.
        """
        if vec_full.shape[-1] != self.full_size:
            raise ValueError(
                f"reduce: последний размер vec_full должен быть {self.full_size}, "
                f"получено {vec_full.shape[-1]}"
            )
        device = vec_full.device
        _, rep_dev = self._get_index_tensors(device)
        return vec_full.index_select(dim=-1, index=rep_dev)

    # ------------------------- pack/unpack в свободное пространство
    def pack(
        self,
        params: Mapping[str, float],
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = DT_R,
        strict: bool = True,
        default: float = 0.0,
        tol: float = 0.0,
    ) -> torch.Tensor:
        """
        Упаковывает dict параметров сразу в свободный вектор (L_free,).

        Особенности:
        - Можно задавать ключи **любыми** членами tied-группы.
          Например, можно задать "M1_2" или "M(n-1)_n" — это один и тот же free-параметр.
        - Если заданы два ключа из одной группы и значения различаются (с учётом tol),
          будет ошибка (защита от тихих конфликтов).

        Поддерживаемый "сахар":
        - phase_a (скаляр) -> разворачивается на phase_a1..phase_aP, если блок включён.
        - phase_b (скаляр) -> аналогично.
        - qu (скаляр) -> если include_qu="vec" в schema, разворачиваем на qu_1..qu_n.

        Параметры
        ---------
        strict:
            True  -> требовать, чтобы были заданы ВСЕ free_keys,
            False -> отсутствующие заполняются default.
        tol:
            Допуск для сравнения конфликтующих значений внутри одной группы.
            tol=0.0 означает строгое сравнение.
        """

        # Работаем с модифицируемой копией (нужно для сахара)
        mutable: Dict[str, float] = dict(params)

        # --- сахар для фаз: phase_a / phase_b ---
        P = self.schema.topo.ports
        sl_a = self.schema.slices.get("phase_a", slice(0, 0))
        sl_b = self.schema.slices.get("phase_b", slice(0, 0))
        schema_has_a = sl_a.stop > sl_a.start
        schema_has_b = sl_b.stop > sl_b.start

        if schema_has_a and "phase_a" in mutable:
            val = float(mutable["phase_a"])
            for i in range(1, P + 1):
                mutable.setdefault(f"phase_a{i}", val)
            # ВАЖНО: ключ "phase_a" сам по себе НЕ входит в ParamSchema.keys,
            # поэтому удаляем его после разворачивания, чтобы не получить KeyError ниже.
            mutable.pop("phase_a", None)

        if schema_has_b and "phase_b" in mutable:
            val = float(mutable["phase_b"])
            for i in range(1, P + 1):
                mutable.setdefault(f"phase_b{i}", val)
            # Аналогично: "phase_b" не является ключом схемы.
            mutable.pop("phase_b", None)

        # --- сахар для qu: qu (скаляр) при include_qu="vec" ---
        if self.schema.include_qu == "vec" and "qu" in mutable:
            n = self.schema.topo.order
            val = float(mutable["qu"])
            for i in range(1, n + 1):
                mutable.setdefault(f"qu_{i}", val)
            # "qu" как скалярный сахар не присутствует среди ключей vec-схемы.
            mutable.pop("qu", None)

        # --- заполняем значения в free-вектор ---
        if device is None:
            out = torch.full((self.free_size,), float(default), dtype=dtype)
        else:
            dev = torch.device(device)
            out = torch.full((self.free_size,), float(default), dtype=dtype, device=dev)

        seen: Dict[int, float] = {}
        for k, v in mutable.items():
            if not isinstance(k, str):
                continue
            if k not in self.key_to_free:
                raise KeyError(f"pack: ключ {k!r} отсутствует в tied-схеме (и в исходной ParamSchema)")
            fi = self.key_to_free[k]
            fv = float(v)
            if fi in seen:
                # конфликт внутри группы?
                prev = seen[fi]
                if tol > 0.0:
                    if abs(prev - fv) > tol:
                        raise ValueError(
                            f"pack: конфликт значений для tied-группы '{self.free_keys[fi]}': "
                            f"{prev} vs {fv} (tol={tol})"
                        )
                else:
                    if prev != fv:
                        raise ValueError(
                            f"pack: конфликт значений для tied-группы '{self.free_keys[fi]}': "
                            f"{prev} vs {fv}"
                        )
            else:
                seen[fi] = fv
            out[fi] = fv

        if strict:
            missing = [self.free_keys[i] for i in range(self.free_size) if i not in seen]
            if missing:
                raise KeyError(f"pack: отсутствуют free-ключи: {missing}")

        return out

    def unpack(self, vec_free: torch.Tensor) -> Dict[str, float]:
        """Обратное преобразование (для логов/отладки): (L_free,) -> dict(rep_key -> value)."""
        if vec_free.ndim != 1 or vec_free.shape[0] != self.free_size:
            raise ValueError(
                f"unpack: ожидается вектор ({self.free_size},), получено {tuple(vec_free.shape)}"
            )
        return {k: float(vec_free[i].item()) for i, k in enumerate(self.free_keys)}

    # ------------------------- assemble: free -> full -> schema.assemble
    def assemble(self, vec_free: torch.Tensor, *, device: Optional[torch.device | str] = None):
        """
        Комбинированный метод:
            vec_free -> expand -> schema.assemble

        Возвращает те же блоки, что и ParamSchema.assemble():
            M_real, qu, phase_a, phase_b
        """
        dev = torch.device(device) if device is not None else vec_free.device
        if vec_free.device != dev:
            vec_free = vec_free.to(dev)
        vec_full = self.expand(vec_free)
        return self.schema.assemble(vec_full, device=dev)

    # ------------------------- конструктор из групп
    @classmethod
    def from_groups(
        cls,
        schema: ParamSchema,
        groups: Sequence[Sequence[int]],
        *,
        store_groups: bool = True,
    ) -> "TiedParamSchema":
        """
        Построить TiedParamSchema из явного списка групп индексов полной схемы.

        groups: список групп (каждая группа — последовательность индексов full-вектора).
               Индексы должны покрывать ВСЕ элементы [0..L_full-1] ровно один раз.
        """
        L = schema.size
        covered = [False] * L
        for g in groups:
            for idx in g:
                if not (0 <= idx < L):
                    raise ValueError("from_groups: индекс вне диапазона полной схемы")
                if covered[idx]:
                    raise ValueError("from_groups: индекс встречается в нескольких группах")
                covered[idx] = True
        if not all(covered):
            raise ValueError("from_groups: группы не покрывают весь полный вектор")

        # выбираем репрезентанта = минимальный индекс в группе
        rep_and_group = []
        for g in groups:
            g_sorted = sorted(set(int(x) for x in g))
            rep_and_group.append((g_sorted[0], g_sorted))
        rep_and_group.sort(key=lambda x: x[0])

        reps = [rep for rep, _ in rep_and_group]
        free_keys = tuple(schema.keys[r] for r in reps)

        # full_to_free: для каждого full-индекса — номер группы (free-индекс)
        full_to_free = [0] * L
        key_to_free: Dict[str, int] = {}
        group_keys: List[Tuple[str, ...]] = []

        for free_i, (_, g_sorted) in enumerate(rep_and_group):
            keys = tuple(schema.keys[idx] for idx in g_sorted)
            group_keys.append(keys)
            for idx in g_sorted:
                full_to_free[idx] = free_i
                key_to_free[schema.keys[idx]] = free_i

        obj = cls(
            schema=schema,
            free_keys=free_keys,
            key_to_free=key_to_free,
            full_to_free=torch.as_tensor(full_to_free, dtype=torch.long, device="cpu"),
            rep_full_idx=torch.as_tensor(reps, dtype=torch.long, device="cpu"),
            groups=tuple(group_keys) if store_groups else tuple(),
        )
        return obj


# ────────────────────────────────────────────────────────────────────────────
#                  Построение tie-групп по перестановке и флагам
# ────────────────────────────────────────────────────────────────────────────

def build_tied_schema(
    schema: ParamSchema,
    perm: NodePermutation,
    *,
    tie_M: bool = True,
    tie_qu: bool = False,
    tie_phases: bool = False,
    validate_topology: bool = True,
    store_groups: bool = True,
) -> TiedParamSchema:
    """
    Универсальный билдёр TiedParamSchema: объединяет параметры, которые должны быть равны,
    согласно перестановке узлов perm.

    Параметры
    ---------
    tie_M:
        Связывать ли блок M (резонаторная диагональ + связи topo.links).
        Обычно для симметричных фильтров это True.
    tie_qu:
        Связывать ли qu_i зеркально (имеет смысл только если schema.include_qu == "vec").
    tie_phases:
        Связывать ли phase_a*/phase_b* согласно перестановке портов (если фазы включены).
    validate_topology:
        Если True и tie_M=True, проверяем инвариантность topo.links под perm.
    store_groups:
        Если True — сохраняем список ключей групп (для диагностики). Можно выключить,
        чтобы уменьшить память/repr.

    Возвращает
    ----------
    TiedParamSchema
    """
    topo = schema.topo
    if topo.size != perm.size:
        raise SymmetryError(f"build_tied_schema: topo.size={topo.size} != perm.size={perm.size}")

    if tie_M and validate_topology:
        validate_topology_invariant(topo, perm)

    # key -> full_index
    k2i: Dict[str, int] = {k: idx for idx, k in enumerate(schema.keys)}

    L = schema.size
    dsu = _DSU(L)

    # ---------------- tie M-block ----------------
    if tie_M:
        sl_M = schema.slices["M"]
        for full_idx in range(sl_M.start, sl_M.stop):
            key = schema.keys[full_idx]  # "M.._.." или "M.._.."
            i, j = _parse_m_key(key)
            a, b = perm.apply_pair(i, j)
            key2 = f"M{a}_{b}"
            idx2 = k2i.get(key2)
            if idx2 is None:
                # Это означает, что в схеме нет симметричного параметра.
                # Чаще всего — топология не инвариантна (или schema собрана из другой topo).
                raise SymmetryError(
                    f"tie_M: симметричный ключ {key2!r} для {key!r} отсутствует в ParamSchema"
                )
            dsu.union(full_idx, idx2)

    # ---------------- tie qu-block ----------------
    if tie_qu:
        if schema.include_qu == "vec":
            n = topo.order
            for i in range(1, n + 1):
                k1 = f"qu_{i}"
                # perm.apply(i) должен снова попасть в резонаторный диапазон 1..n
                j = perm.apply(i)
                if not (1 <= j <= n):
                    raise SymmetryError(
                        f"tie_qu: perm отправляет резонатор {i} в узел {j}, "
                        f"который не является резонатором (1..{n})"
                    )
                k2 = f"qu_{j}"
                if k1 in k2i and k2 in k2i:
                    dsu.union(k2i[k1], k2i[k2])
                # если qu-блок не включён в схему, ключей просто нет — ничего не делаем.
        else:
            # include_qu == "none" или "scalar": tie_qu либо не имеет смысла, либо тривиален
            pass

    # ---------------- tie phase-blocks ----------------
    if tie_phases:
        order = topo.order
        P = topo.ports

        def _tie_phase(prefix: str) -> None:
            # prefix: "phase_a" или "phase_b"
            for p in range(1, P + 1):
                k1 = f"{prefix}{p}"
                if k1 not in k2i:
                    continue
                # портовый узел в расширенной матрице: order + p
                node = order + p
                node2 = perm.apply(node)
                if not (order + 1 <= node2 <= order + P):
                    raise SymmetryError(
                        f"tie_phases: perm отправляет портовый узел {node} в {node2}, "
                        f"который не является портом (ожидалось {order+1}..{order+P})"
                    )
                p2 = node2 - order
                k2 = f"{prefix}{p2}"
                if k2 not in k2i:
                    raise SymmetryError(
                        f"tie_phases: симметричный ключ {k2!r} для {k1!r} отсутствует в ParamSchema"
                    )
                dsu.union(k2i[k1], k2i[k2])

        _tie_phase("phase_a")
        _tie_phase("phase_b")

    # ---------------- собрать группы по DSU ----------------
    groups_map: Dict[int, List[int]] = {}
    for idx in range(L):
        root = dsu.find(idx)
        groups_map.setdefault(root, []).append(idx)

    groups = list(groups_map.values())
    # Отсортируем группы детерминированно по минимальному индексу
    groups.sort(key=lambda g: min(g))

    return TiedParamSchema.from_groups(schema, groups, store_groups=store_groups)


# ────────────────────────────────────────────────────────────────────────────
#            Удобный конструктор: зеркальная симметрия для 2-портов
# ────────────────────────────────────────────────────────────────────────────

def tied_mirror_schema_2port(
    topo: Topology,
    *,
    include_qu: IncludeQU = "vec",
    include_phase: Tuple[Literal["a", "b"], ...] = ("a",),
    tie_ports: bool = True,
    tie_qu: bool = False,
    tie_phases: bool = False,
    validate_topology: bool = True,
    store_groups: bool = True,
) -> TiedParamSchema:
    """
    Конструктор "под ключ" для наиболее частого случая:
    **2-портовый фильтр** + **зеркальная симметрия относительно побочной диагонали**.

    Шаги:
      1) создаём полную ParamSchema из topo,
      2) строим перестановку mirror_perm_2port(order, tie_ports=...),
      3) (опционально) проверяем инвариантность topo.links,
      4) строим TiedParamSchema (tie_M всегда True).

    Параметры
    ---------
    tie_ports:
        True  -> при отражении меняем местами порты P1<->P2 (типичный случай).
        False -> порты остаются на местах (редко полезно).
    tie_qu:
        Связывать ли qu_i зеркально (только если include_qu="vec").
    tie_phases:
        Связывать ли phase_* согласно perm (как правило имеет смысл только если tie_ports=True
        и фазы действительно включены в include_phase).
    validate_topology:
        Проверять ли, что topo.links инвариантна под отражением.
    """
    if topo.ports != 2:
        raise ValueError("tied_mirror_schema_2port: topo.ports должен быть равен 2")

    schema = ParamSchema.from_topology(
        topo,
        include_qu=include_qu,
        include_phase=include_phase,
    )
    perm = mirror_perm_2port(topo.order, tie_ports=tie_ports)

    return build_tied_schema(
        schema,
        perm,
        tie_M=True,
        tie_qu=tie_qu,
        tie_phases=tie_phases,
        validate_topology=validate_topology,
        store_groups=store_groups,
    )


__all__ = [
    "SymmetryError",
    "NodePermutation",
    "mirror_perm_2port",
    "validate_topology_invariant",
    "TiedParamSchema",
    "build_tied_schema",
    "tied_mirror_schema_2port",
]
