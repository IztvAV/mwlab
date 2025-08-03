# mwlab/data_gen/sources/list.py
"""
ListSource – самый простой *ParamSource*
=======================================
Источник читает уже готовую коллекцию словарей параметров, хранящуюся в памяти.
Главное назначение – юнит‑тесты, прототипы и небольшие синтетические наборы
(< 10‑100 тыс. точек), когда нет смысла тянуть CSV или БД.

Ключевые особенности
--------------------
* **Уникальный идентификатор**.  Каждый dict *обязан* содержать ключ
  ``"__id"`` – строку.  Если у входных элементов id отсутствует, класс
  автоматически присваивает «p0», «p1», …
* **Перемешивание** (``shuffle=True``) позволяет рандомизировать порядок точек
  заранее, сохраняя детерминизм (используется ``random.shuffle``).
* **Копирование или ссылочная работа**.  Параметр ``copy`` контролирует, будет
  ли источник делать глубокий список‑копию (удобно, если внешний код потом
  изменяет исходную коллекцию) или хранить ссылку на оригинальную.
* **reserve / mark_done / mark_failed** – *no‑op*, потому что весь список
  локален и никакой координации между воркерами не требуется.
"""

from __future__ import annotations

import random
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from mwlab.data_gen.base import ParamDict, ParamSource

__all__ = ["ListSource"]


class ListSource(ParamSource):
    """In‑memory источник параметров.

    Parameters
    ----------
    data : Sequence[Mapping[str, Any]]
        Готовый набор точек.  Каждый элемент – произвольный ``dict``.
        *Если* в элементе нет ключа ``"__id"`` – он будет создан
        автоматически на основе позиции в списке.
    shuffle : bool, default False
        Перемешать ли порядок точек сразу при инициализации.
    copy : bool, default False
        Создавать ли внутреннюю копию списка (``list(data)``).  Если ``False`` –
        источник хранит *ссылку* на оригинальный контейнер.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        data: Sequence[Mapping[str, Any]],
        *,
        shuffle: bool = False,
        seed: int | None = None,
        copy: bool = False,
    ) -> None:
        # 1) приводим к списку и при необходимости копируем элементы
        if copy:
            self._data: list[ParamDict] = [dict(p) for p in data]
        else:
            # list() гарантирует индексируемость, даже если пришёл tuple / deque …
            self._data = list(data)

        # 2) назначаем __id, проверяем уникальность
        seen_ids: set[str] = set()
        for idx, item in enumerate(self._data):
            # преобразуем к MutableMapping, чтобы можно было дописать ключ
            if "__id" not in item:
                if not isinstance(item, MutableMapping):
                    item = dict(item)  # сделать мутабельным
                    self._data[idx] = item
                item["__id"] = f"p{idx}"

            # ---- проверка на дубликаты ----
            cur_id = str(item["__id"])
            if cur_id in seen_ids:
                raise ValueError(f"ListSource: дублирующий __id='{cur_id}'")
            seen_ids.add(cur_id)

        # 3) возможно перемешиваем

        if shuffle:
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(self._data)

    # ---------------------------------------------------------------- iterator
    def __iter__(self) -> Iterator[ParamDict]:
        """Просто отдаём элементы по порядку."""
        yield from self._data

    # ---------------------------------------------------------------- length
    def __len__(self) -> int:  # noqa: D401
        return len(self._data)

    # ---------------------------------------------------------------- no‑op hooks
    def reserve(self, ids):  # noqa: D401, WPS110
        # локальный список – ничего резервировать не нужно
        pass

    def mark_done(self, ids):  # noqa: D401, WPS110
        pass

    def mark_failed(self, ids, exc):  # noqa: D401, WPS110
        pass
