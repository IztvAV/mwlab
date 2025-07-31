# mwlab/data_gen/base.py
"""
mwlab.data_gen.base
───────────────────
Базовые абстракции, из которых строится любой генератор данных
в экосистеме MWLab.

Поток данных:

    ParamSource  →  DataGenerator  →  Writer
      (параметры)                     (результат + meta)

* **ParamSource** – отдаёт точки пространства параметров.
* **DataGenerator** – превращает батч параметров в данные
  (TouchstoneData, Tensor, …) и формирует произвольную мета-информацию.
* **Writer** – сохраняет полученные результаты в выбранный backend
  (каталог .sNp, HDF5, LMDB, прямо в StorageBackend …).

Дизайн-решения
──────────────
1. Все классы – контекст-менеджеры; достаточно писать::

       with CsvSource(csv) as src, HDF5Writer("out.h5") as wr:
           gen.run(src, wr)

2. Метаданные могут возвращаться
   • поэлементно (List[dict])
   • единым скалярным dict (распространяется на все элементы)
   • None (ничего).
3. Встроенный `tqdm`-progress-bar включается флагом `show_progress`.
4. Используется стандартный `logging`, ссылка хранится в
   `self.logger`, доступна во всех методах генератора.
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
    Tuple,
    Union,
)

import torch

from mwlab.io.touchstone import TouchstoneData

# ─────────────────────────────────────────────────────────────────────────────
#  Type aliases
# ─────────────────────────────────────────────────────────────────────────────
GeneratorOutput = Union[
    TouchstoneData,
    torch.Tensor,
    Mapping[str, Any],           # «сырые» данные в произвольном формате
]

# meta может быть
#   • None
#   • один общий словарь
#   • список словарей длиной = batch
MetaLike = Union[None, Mapping[str, Any], Sequence[Mapping[str, Any]]]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ParamSource – откуда брать параметры
# ─────────────────────────────────────────────────────────────────────────────
class ParamSource(AbstractContextManager, ABC):
    """
    Итератор (и, опционально, менеджер ресурсов) для чтения точек параметров.

    *По умолчанию* все методы-хуки (`reserve`, `mark_done`, `fail`) ничего
    не делают – этого достаточно для простых in-memory списков.
    При распределённом или fault-tolerant запуске наследник переопределяет
    нужные вызовы (запись статуса точки в CSV/БД/Redis …).
    """

    # ––––– интерфейс итератора –––––
    @abstractmethod
    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Возвращает *бесконечный* или конечный поток параметров."""
        ...

    # длина может быть неизвестна → вернуть None
    def __len__(self) -> int | None:  # noqa: D401
        return None

    # ––––– опциональные вызовы для распределённых сценариев –––––
    def reserve(self, n: int) -> None:
        """
        Сообщить источнику, что прямо сейчас планируется обработать *n* точек.
        Используется для атомарного «захвата» кусочка работы.
        """
        pass

    def mark_done(self, params: Mapping[str, Any]) -> None:
        """Точка успешно обработана."""
        pass

    def fail(self, params: Mapping[str, Any], exc: Exception) -> None:
        """Точка завершилась исключением *exc*."""
        pass

    # ––––– контекст-менеджер –––––
    # большинству источников ничего не нужно закрывать
    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. BaseWriter – куда складывать результат
# ─────────────────────────────────────────────────────────────────────────────
class BaseWriter(AbstractContextManager, ABC):
    """
    “Приёмник” (sink) для результатов генератора.

    •   `outputs` – список объектов любого поддерживаемого типа
        (TouchstoneData, Tensor, ...).
    •   `params_batch` – ровно те же словари параметров,
        что подавались генератору.
    •   `meta` – либо None, либо общий dict, либо список dict'ов.
        Writer решает, что с ними делать: писать в header .sNp,
        в HDF5-атрибуты, игнорировать…
    """

    @abstractmethod
    def write(
        self,
        outputs: Sequence[GeneratorOutput],
        params_batch: Sequence[Mapping[str, Any]],
        meta: MetaLike = None,
    ) -> None: ...

    # логировать ошибку (по желанию)
    def log_error(self, params_batch, exc):  # noqa: D401
        pass

    # для буферизованных Writer'ов можно перегрузить flush
    def flush(self):  # noqa: D401
        pass

    # контекст-менеджер
    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        self.flush()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. DataGenerator – ядро
# ─────────────────────────────────────────────────────────────────────────────
class DataGenerator(ABC):
    """
    Базовый класс конкретных генераторов.

    Пользователь обычно переопределяет **только** `generate_batch`.
    При необходимости – `preprocess` и/или `postprocess`.

    * Атрибут `batch_size` можно задать через конструктор.
    * Логгер доступен как `self.logger`.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        batch_size: int = 1,
        logger: logging.Logger | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = int(batch_size)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    # ---------------------------------------------------------------- hooks
    def preprocess(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Опциональная валидация / нормализация входных параметров."""
        return params

    @abstractmethod
    def generate_batch(
        self,
        params_batch: Sequence[Mapping[str, Any]],
    ) -> Tuple[Sequence[GeneratorOutput], MetaLike]:
        """Главная работа: params → данные (+ meta)."""
        ...

    def postprocess(
        self,
        outputs: Sequence[GeneratorOutput],
        meta: MetaLike,
        params_batch: Sequence[Mapping[str, Any]],
    ) -> Tuple[Sequence[GeneratorOutput], Sequence[Mapping[str, Any]]]:
        """
        Финальная обработка перед записью.
        ↳ Возвращаем гарантированно *список* meta той же длины,
        что и outputs.
        """
        if meta is None:
            meta_seq: Sequence[Mapping[str, Any]] = [{} for _ in outputs]
        elif isinstance(meta, Mapping):
            meta_seq = [meta] * len(outputs)
        else:
            meta_seq = list(meta)
            if len(meta_seq) != len(outputs):
                raise ValueError("meta длины batch не совпадает с outputs")
        return outputs, meta_seq

    # ---------------------------------------------------------------- public run
    def run(
        self,
        source: ParamSource,
        writer: BaseWriter,
        *,
        show_progress: bool = True,
    ):
        """
        Высокоуровневый цикл:

            for batch in source:
                preprocess → generate_batch → postprocess → writer.write
        """
        iterator: Iterable[Mapping[str, Any]] = source
        total = len(source) if hasattr(source, "__len__") else None

        # progress-bar (tqdm подключается, если есть и если включён)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, total=total, desc="DataGen")
            except ModuleNotFoundError:
                self.logger.debug("tqdm не установлен – прогрессбар выключен")

        with source, writer:
            batch: list[Mapping[str, Any]] = []

            def _commit():
                """Обработать накопленный batch."""
                if not batch:
                    return
                try:
                    clean = [self.preprocess(p) for p in batch]
                    outs, meta = self.generate_batch(clean)
                    outs, meta = self.postprocess(outs, meta, clean)
                    writer.write(outs, clean, meta)
                    for p in clean:
                        source.mark_done(p)
                except Exception as e:
                    self.logger.exception("Batch failed")
                    for p in batch:
                        source.fail(p, e)
                    writer.log_error(batch, e)
                finally:
                    batch.clear()

            # основной цикл
            for params in iterator:
                if not batch:
                    # захватываем «porцию» заранее, если нужно
                    source.reserve(self.batch_size)
                batch.append(params)

                if len(batch) >= self.batch_size:
                    _commit()

            # хвост
            _commit()

        writer.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Утилита (может пригодиться)
# ─────────────────────────────────────────────────────────────────────────────
def batched(iterable: Iterable[Any], n: int) -> Iterator[Tuple[Any, ...]]:
    """
    Разбивает любой итерируемый объект на чанки по *n* элементов.
    Последний чанк может быть короче.
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


# ─────────────────────────────────────────────────────────────────────────────
# Re-export
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "GeneratorOutput",
    "MetaLike",
    "ParamSource",
    "BaseWriter",
    "DataGenerator",
    "batched",
]
