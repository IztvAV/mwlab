# mwlab/data_gen/runner.py
"""
Управляющий цикл генерации данных и тонкая обёртка GenRunner
============================================================

Модуль связывает три абстракции из :mod:`mwlab.data_gen.base`:

    ParamSource  →  DataGenerator  →  Writer

Ключевые сущности
-----------------
* :func:`run_pipeline` — референсная (однопоточная) функция-оркестратор.
  Идёт по источнику параметров, пакетирует по `batch_size`, последовательно
  вызывает у Source хук-методы `reserve / mark_done / mark_failed`, у Generator
  — `preprocess_batch / preprocess / generate`, у Writer — `write / flush`.
* :class:`GenRunner` — тонкая обёртка над :func:`run_pipeline`, хранящая
  конфигурацию запуска (размер батча, прогрессбар, логгер, колбэк).

Колбэк `on_batch_end`
---------------------
Опциональный аргумент `on_batch_end` (и у :func:`run_pipeline`, и у
:class:`GenRunner`) позволяет подписаться на завершение **каждого** коммита
батча — как успешного, так и завершившегося ошибкой. Это удобно для
онлайн-метрик, логирования, телеметрии и т.п.

**Сигнатура:**
    `on_batch_end(payload: dict) -> None`

**Структура `payload`:**
    {
        "ok_delta": int,          # сколько точек успешно записано этим коммитом
        "fail_delta": int,        # сколько точек помечено failed этим коммитом
        "processed_total": int,   # накопительный счётчик успешных точек (с начала запуска)
        "failed_total": int,      # накопительный счётчик неуспешных точек
        "ids": list[str],         # __id всех точек текущего батча (после preprocess*)
        "exception": Exception | None,  # исключение для неуспешного коммита, иначе None
    }

**Гарантии и поведение:**
* Колбэк вызывается **ровно один раз** на коммит батча — и при успехе, и при ошибке.
* На ранних ошибках (например, в `preprocess*`) в `ids` передаются исходные `__id`
  элементов батча.
* Исключения, брошенные внутри колбэка, **не прерывают** пайплайн (они
  подавляются внутри раннера); используйте собственный логгер при необходимости.
* Колбэк должен быть **быстрым** и не выполнять длительных блокирующих операций,
  чтобы не тормозить обработку батчей.

Примеры использования `on_batch_end`
------------------------------------

1) Простой логирующий колбэк:

    ```python
    import logging
    log = logging.getLogger("datagen")

    def on_end(payload: dict):
        if payload["exception"] is None:
            log.info("OK: +%d (total=%d), ids=%s",
                     payload["ok_delta"], payload["processed_total"], payload["ids"])
        else:
            log.warning("FAIL: +%d (failed_total=%d), err=%r, ids=%s",
                        payload["fail_delta"], payload["failed_total"],
                        payload["exception"], payload["ids"])
    ```

2) Подсчёт производительности (точек/сек) и вывод каждые N батчей:

    ```python
    import time
    N = 20
    t0 = time.perf_counter()
    counters = {"batches": 0}

    def on_end(payload: dict):
        counters["batches"] += 1
        if counters["batches"] % N == 0:
            dt = time.perf_counter() - t0
            rate = payload["processed_total"] / max(dt, 1e-9)
            print(f"[stats] processed={payload['processed_total']} "
                  f"failed={payload['failed_total']} "
                  f"rate={rate:.1f} samples/s")
    ```

3) Сбор агрегированных метрик для последующей визуализации:

    ```python
    history = {"ok": [], "fail": []}

    def on_end(payload: dict):
        history["ok"].append(payload["processed_total"])
        history["fail"].append(payload["failed_total"])

    # ... после окончания пайплайна:
    # plt.plot(history["ok"]); plt.plot(history["fail"])
    ```

Замечания по интеграции pre-hooks
---------------------------------
Перед вызовом `reserve()` выполняются:
1) `generator.preprocess_batch(batch)` — не должен менять длину батча;
2) `generator.preprocess(item)` для каждого элемента — рекомендуется **не менять**
   ключ `"__id"`. После этих хуков раннер проверяет, что внутри батча нет
   дубликатов `__id`, и только затем вызывает `source.reserve(ids)`.

Совместимость
-------------
API совместим с прежними версиями. Если `on_batch_end` не передан, поведение
полностью соответствует классической однопоточной ссылочной реализации.
"""


from __future__ import annotations

import contextlib
import logging
import signal
import sys
from typing import Optional, Callable, Dict, Any

from .base import Batch, DataGenerator, ParamSource, Writer

# ---------------------------------------------------------------------------
# Internal helper: graceful CTRL‑C handling
# ---------------------------------------------------------------------------
class _GracefulInterruptHandler:  # pragma: no cover
    """Context‑manager, который переводит SIGINT в Python‑исключение один раз.

    В кроссплатформенных сценариях (Windows) SIGINT обрабатывается иначе, но
    для локального Development достаточно такой реализации.
    """

    def __init__(self):
        self._orig_handler = None
        self.interrupted = False

    def __enter__(self):
        self._orig_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        signal.signal(signal.SIGINT, self._orig_handler)  # type: ignore[arg-type]
        return False  # не подавляем исключения

    # -------------------- handler
    def _handle(self, signum, frame):  # noqa: D401
        if self.interrupted:
            # второй Ctrl‑C – выходим немедленно
            sys.exit(130)
        self.interrupted = True
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# публичная функция run_pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: ParamSource,
    generator: DataGenerator,
    writer: Writer,
    *,
    batch_size: int = 1,
    progress: bool = True,
    logger: Optional[logging.Logger] = None,
    on_batch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> dict[str, int]:
    """Запускает полный цикл генерации.

    Parameters
    ----------
    source : ParamSource
        Итератор параметров.  Должен быть *уже* создан, настройки – через его
        собственный конструктор.
    generator : DataGenerator
        Любой наследник, реализующий `generate(batch)`.
    writer : Writer
        Куда складывать результат.
    batch_size : int, default 1
        Сколько точек передаём генератору за один вызов.
    progress : bool, default True
        Показывать ли прогрессбар (используется *tqdm* при наличии).
    logger : logging.Logger | None
        Свой логгер; если None, создаётся «DataGenRunner».
     on_batch_end : callable | None
        Необязательный колбэк, вызывается ПОСЛЕ каждой фиксации батча
        (успех или ошибка). Получает dict с мини-статистикой и списком id.

    Returns
    -------
    dict
        Мини‑статистика: ``{"processed": N_ok, "failed": N_fail}``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    log = logger or logging.getLogger("DataGenRunner")
    processed = failed = 0

    # Используем ExitStack, чтобы гарантировать закрытие всех ресурсов
    with contextlib.ExitStack() as stack, _GracefulInterruptHandler():
        stack.enter_context(source)
        stack.enter_context(generator)
        stack.enter_context(writer)

        # Подготавливаем прогрессбар (теперь источник уже инициализирован).
        total = None
        iterator = iter(source)
        pbar = None
        if progress:
            try:
                from tqdm.auto import tqdm  # noqa: WPS433 (runtime import)
                try:
                    total = len(source)  # может бросить, напр. NotImplementedError
                except Exception:
                    total = None
                iterator = tqdm(iterator, total=total, desc="DataGen")
                pbar = iterator
            except ModuleNotFoundError:  # pragma: no cover
                log.debug("tqdm не найден – прогрессbar отключён")
                progress = False

        batch: Batch = []  # накапливаем текущий пакет параметров

        def _commit():  # локальная функция обработки пачки
            nonlocal processed, failed, batch
            if not batch:
                return

            # 0) исходные ids для случая раннего фейла до резервирования
            orig_ids = [p.get("__id") for p in batch]

            # 1) предварительная обработка батча и отдельных элементов
            try:
                # Важно: preprocess* НЕ должны менять "__id" (рекомендация/контракт)
                pre_batch = generator.preprocess_batch(batch)
                if len(pre_batch) != len(batch):
                    raise ValueError(
                        "preprocess_batch должен сохранять длину батча: "
                        f"{len(pre_batch)} != {len(batch)}"
                    )
                pre_items = [generator.preprocess(p) for p in pre_batch]

                # 2) собираем ids и проверяем дубликаты
                ids = [p["__id"] for p in pre_items]
                if len(set(ids)) != len(ids):
                    raise ValueError("Обнаружены дубликаты '__id' внутри батча")

            except Exception as exc:  # ранний фейл на preprocess*
                # помечаем как failed по ИСХОДНЫМ id
                safe_ids = [str(i) for i in orig_ids if i is not None]
                try:
                    source.mark_failed(safe_ids, exc)
                finally:
                    failed += len(batch)
                    if progress:
                        iterator.update(len(batch))  # type: ignore[attr-defined]
                    # колбэк об окончании батча (ошибка)
                    if on_batch_end is not None:
                        with contextlib.suppress(Exception):
                            on_batch_end({
                                "ok_delta": 0,
                                "fail_delta": len(batch),
                                "processed_total": processed,
                                "failed_total": failed,
                                "ids": safe_ids,
                                "exception": exc,
                            })
                    batch = []
                return

            # 3) резервируем именно эти ids (после успешного preprocess*)
            source.reserve(ids)
            try:
                outputs, meta = generator.generate(pre_items)

                # --- валидация длин ---
                if len(outputs) != len(pre_items) or len(meta) != len(pre_items):
                    raise ValueError(
                        "Generator returned lengths: "
                        f"outputs={len(outputs)}, meta={len(meta)}, params={len(pre_items)}",
                    )

                writer.write(outputs, meta, pre_items)
                source.mark_done(ids)
                processed += len(pre_items)

                # колбэк об окончании батча (успех)
                if on_batch_end is not None:
                    with contextlib.suppress(Exception):
                        on_batch_end({
                            "ok_delta": len(pre_items),
                            "fail_delta": 0,
                            "processed_total": processed,
                            "failed_total": failed,
                            "ids": ids,
                            "exception": None,
                        })
            except Exception as exc:  # noqa: BLE001
                # Добавляем идентификаторы батча в лог для удобной диагностики
                log.exception("Batch failed (ids=%s) – записываю в Source.mark_failed()", ids)
                source.mark_failed(ids, exc)
                failed += len(pre_items)
                # колбэк об окончании батча (ошибка)
                if on_batch_end is not None:
                    with contextlib.suppress(Exception):
                        on_batch_end({
                            "ok_delta": 0,
                            "fail_delta": len(pre_items),
                            "processed_total": processed,
                            "failed_total": failed,
                            "ids": ids,
                            "exception": exc,
                        })
            finally:
                batch = []  # очищаем для следующего набора

        # ---------------------- основной цикл ----------------------
        try:
            for params in iterator:
                batch.append(params)
                if len(batch) >= batch_size:
                    _commit()
            # хвост
            _commit()
            writer.flush()
        finally:
            # Явно закрываем прогресс-бар (если он есть), чтобы корректно завершать вывод
            if progress and pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.close()

    return {"processed": processed, "failed": failed}

# ---------------------------------------------------------------------------
# Thin wrapper: GenRunner
# ---------------------------------------------------------------------------

class GenRunner:
    """Объект‑обёртка вокруг :func:`run_pipeline`.

    Сохраняет конфигурацию (batch_size, progress, logger) и позволяет запускать
    один и тот же раннер на разных тройках *(Source, Generator, Writer)* без
    повторения аргументов.
    """

    def __init__(
        self,
        *,
        batch_size: int = 1,
        progress: bool = True,
        logger: Optional[logging.Logger] = None,
        on_batch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = batch_size
        self.progress = progress
        self.logger = logger
        self.on_batch_end = on_batch_end

    # -------------------------------------------------------- call
    def __call__(
        self,
        source: ParamSource,
        generator: DataGenerator,
        writer: Writer,
    ) -> dict[str, int]:
        """Запускает пайплайн с сохранёнными настройками."""
        return run_pipeline(
            source,
            generator,
            writer,
            batch_size=self.batch_size,
            progress=self.progress,
            logger=self.logger,
            on_batch_end=self.on_batch_end,
        )

    # -------------------------------------------------------- repr
    def __repr__(self):  # pragma: no cover
        return (
            f"GenRunner(batch_size={self.batch_size}, "
            f"progress={self.progress}, logger={self.logger}, "
            f"on_batch_end={self.on_batch_end})"
        )


# ---------------------------------------------------------------------------
# __all__ для публичного импорта
# ---------------------------------------------------------------------------
__all__ = ["run_pipeline", "GenRunner"]
