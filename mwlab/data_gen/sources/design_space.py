# mwlab/data_gen/sources/design_space.py
"""
DesignSpaceSource – ParamSource, генерирующий точки DOE «на лету»
===============================================================
Источник построен вокруг связки **DesignSpace + Sampler** (см.
``mwlab.opt.design``).  Он удобен, когда нужно синтезировать большое или даже
бесконечное количество параметров без предварительного сохранения в файл.

Ключевые моменты
----------------
* Каждая сгенерированная точка получает уникальный ``__id`` вида ``"p{n}"``.
* Поддерживается ограничение ``n_total`` – общее число точек, которое можно
  сгенерировать.  При ``None`` поток бесконечный (до остановки Runner-а).
* Метод ``reserve(ids)`` по факту вызывает внутреннюю генерацию *reserve_n*
  новых точек и кладёт их в очередь.  Поэтому очередь всегда удовлетворяет
  первоначальному контракту Runner-а: *сначала* Source «обещает» id, *потом*
  отдаёт точки через ``__iter__``.
* Класс потокобезопасен: ``threading.Lock`` защищает _queue и счётчик.

Особенности реализации
----------------
Поддерживает **два режима**:
1. **Incremental** (ленивый, по‑умолчанию)
   * Подходит для Sobol/Halton и любых сэмплеров, которые корректно
     продолжают последовательность при дозапросах.
2. **Precompute / cache‑all** – строит сразу весь план из `n_total` точек и
   раздаёт их порциями.  Включается, если:

   * Пользователь явно передал `cache_all=True`, или
   * У класса‑сэмплера задан атрибут `requires_full_plan = True` (например,
     для LHS, CCD и других фиксированных планов).

Таким образом Sobol/Halton остаются потоковыми, а LHS‑подобные сэмплеры
автоматически получают полный латинский гиперкуб без искажений.
Примечание: LHS, CCD, … будут автоматически закэшированы целиком;
            поэтому для n_total >> 10⁵ используйте Sobol/Halton или custom-sampler
"""

from __future__ import annotations

import random
import threading
from collections import deque
from typing import Iterator, Sequence

from mwlab.data_gen.base import ParamDict, ParamSource
from mwlab.opt.design.samplers import BaseSampler, get_sampler
from mwlab.opt.design.space import DesignSpace

__all__ = ["DesignSpaceSource"]

class DesignSpaceSource(ParamSource):
    """Источник параметров на основе DesignSpace + Sampler."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        space: DesignSpace,
        sampler: str | BaseSampler = "sobol",
        *,
        n_total: int | None = None,
        reserve_n: int = 1024,
        cache_all: bool | None = None,  # True | False | None(→auto)
        shuffle_cached: bool = True,
        seed: int | None = None,
    ) -> None:
        self.space = space
        self.sampler: BaseSampler = (
            get_sampler(sampler, rng=seed) if isinstance(sampler, str) else sampler
        )
        self.n_total = n_total
        self.reserve_n = reserve_n

        # ---- решаем, нужно ли кэширвоать полностью ----
        auto_requires = getattr(self.sampler.__class__, "requires_full_plan", False)
        self._cache_all = bool(cache_all) if cache_all is not None else bool(auto_requires)
        self._shuffle_cached = shuffle_cached
        self._rand = random.Random(seed) if seed is not None else random

        self._generated = 0  # счётчик для id
        self._queue: deque[ParamDict] = deque()
        self._lock = threading.Lock()

        if self._cache_all:
            if self.n_total is None:
                raise ValueError("cache_all=True требует явного n_total")
            self._precompute_all()

    # ---------------------------------------------------------------- helpers
    def _precompute_all(self):
        """Генерирует полный план и складывает его в очередь."""
        pts = self.sampler.sample(self.space, self.n_total)  # type: ignore[arg-type]
        if self._shuffle_cached:
            self._rand.shuffle(pts)

        for idx, p in enumerate(pts):
            p["__id"] = f"p{idx}"
        self._queue = deque(pts)
        self._generated = self.n_total  # дальнейшая генерация не нужна

    def _produce_incremental(self, k: int):
        """Генерирует k точек для инкрементального режима."""
        pts = self.sampler.sample(self.space, k)
        with self._lock:
            start = self._generated
            self._generated += k
        for off, p in enumerate(pts):
            p["__id"] = f"p{start + off}"
        self._queue.extend(pts)

    # ---------------------------------------------------------------- iterator
    def __iter__(self) -> Iterator[ParamDict]:
        # Предохранитель от «пустых дозагрузок», если sampler по какой-то причине
        # не отдаёт точки (или отдаёт пустой список).
        empty_refills = 0
        MAX_EMPTY_REFILLS = 1024  # защита от бесконечного цикла

        while True:
            if not self._queue:
                if self._cache_all:
                    # Всё уже предвычислено и отдано.
                    break

                # --- ленивый режим ---
                remain = None if self.n_total is None else self.n_total - self._generated
                if remain is not None and remain <= 0:
                    break

                take = self.reserve_n if remain is None else min(self.reserve_n, remain)
                if take <= 0:
                    break

                # Пытаемся дозагрузить.
                before = len(self._queue)
                self._produce_incremental(take)

                if len(self._queue) == before:
                    # сэмплер не добавил ни одной точки — считаем «пустую дозагрузку»
                    empty_refills += 1
                    if empty_refills > MAX_EMPTY_REFILLS:
                        raise RuntimeError(
                            "DesignSpaceSource: sampler не производит точки "
                            "(слишком много пустых попыток дозагрузки)."
                        )
                    # Дадим ещё шанс в следующей итерации
                    continue
                else:
                    empty_refills = 0  # прогресс есть — сбрасываем счётчик

            try:
                yield self._queue.popleft()
            except IndexError:
                # Очередь внезапно пуста — завершаем (это нормальный выход).
                break


    # ---------------------------------------------------------------- length
    def __len__(self):  # noqa: D401
        if self.n_total is None:
            raise NotImplementedError
        return self.n_total

    # ----------------------------- reserve/mark (no‑op) ---------------------
    def reserve(self, ids: Sequence[str]):
        if self._cache_all:
            return
        with self._lock:
            if self.n_total is not None and self._generated >= self.n_total:
                return
            remain = None if self.n_total is None else (self.n_total - self._generated)
            take = self.reserve_n if remain is None else min(self.reserve_n, remain)
            if take > 0 and not self._queue:
                self._produce_incremental(take)

    def mark_done(self, ids: Sequence[str]):  # noqa: D401,WPS110
        pass

    def mark_failed(self, ids: Sequence[str], exc: Exception):  # noqa: D401,WPS110
        pass

