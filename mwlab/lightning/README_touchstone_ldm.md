# mwlab.lightning.touchstone_ldm

Модуль `touchstone_ldm.py` предоставляет LightningDataModule-обёртку для работы с Touchstone-данными — файлов формата *.sNp, используемых в ВЧ и СВЧ‑моделировании (например, S‑параметры фильтров, антенн и пр.).

Он основан на `TouchstoneTensorDataset` и обеспечивает полную поддержку типового ML‑пайплайна: разбиение на train/val/test, масштабирование признаков и целей, predict‑режим, работу с метаданными и поддержку обратной постановки задачи.

---

## TouchstoneLDataModule

```python
class TouchstoneLDataModule(L.LightningDataModule)
```

### Основное предназначение

- Чтение Touchstone-данных из каталогов, HDF5-файлов и произвольных backend-ов (`StorageBackend`)
- Разбиение на поднаборы train/val/test (по долям или фиксированному числу примеров)
- Масштабирование входов и выходов с помощью `scaler_in`, `scaler_out` (с авто-обучением по train)
- Возможность постановки обратной задачи: предсказывать X по Y (`swap_xy=True`)
- Режим `predict` с возвратом `meta` для декодирования в `TouchstoneData`
- Методы `get_dataset(...)` и `get_dataloader(...)` для удобного доступа к наборам данных

---

## Аргументы конструктора

| Параметр | Тип | Описание |
|----------|-----|----------|
| `source` | `str` / `Path` / `StorageBackend` | Путь к *.sNp, .h5-файлу или backend-объект |
| `codec` | `TouchstoneCodec` | Кодек для преобразования TouchstoneData ↔ тензоры |
| `batch_size` | `int` | Размер батча |
| `num_workers` | `int` | Число воркеров при загрузке данных |
| `pin_memory` | `bool` | Использовать pinned memory при загрузке |
| `val_ratio` | `float` | Доля валидационного набора |
| `test_ratio` | `float` | Доля тестового набора |
| `max_samples` | `int` / `None` | Ограничение общего числа примеров |
| `seed` | `int` | Фиксация генератора для воспроизводимости |
| `swap_xy` | `bool` | Обратная задача (Y → X) |
| `cache_size` | `int` / `None` | Размер LRU-кеша |
| `scaler_in` / `scaler_out` | `nn.Module` / `None` | Скейлеры признаков и выходов |
| `base_ds_kwargs` | `dict` / `None` | Дополнительные аргументы в `TouchstoneTensorDataset` |

---

## Основные методы

### `setup(stage: str | None)`

- `'fit'` или `None`: создаёт разбиение и, при необходимости, обучает скейлеры
- `'validate'`, `'test'`: требуют предварительный вызов `setup('fit')`
- `'predict'`: инициализирует датасет с `return_meta=True`

---

## Методы доступа к данным

### `get_dataset(split="train", meta=False)`

Возвращает `Dataset` для одного из поднаборов:

- `split ∈ {"train", "val", "test", "full"}`
- если `meta=True`, возвращаются дополнительные словари с метаданными

**Примеры:**

```python
train_ds = ldm.get_dataset("train")
test_ds = ldm.get_dataset("test", meta=True)
```

### `get_dataloader(split="train", meta=False, shuffle=None)`

Создаёт `DataLoader` для выбранного поднабора:

- `shuffle` по умолчанию `True` только для `train`
- если `meta=True`, батчи будут вида `(x, y, metas)`

**Примеры:**

```python
val_loader = ldm.get_dataloader("val")
pred_loader = ldm.get_dataloader("test", meta=True, shuffle=False)
```

---

## Стандартные методы Lightning

- `train_dataloader()` / `val_dataloader()` / `test_dataloader()` / `predict_dataloader()`

---

## Пример использования

```python
from mwlab.lightning.touchstone_ldm import TouchstoneLDataModule
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.nn.scalers import StdScaler, MinMaxScaler

codec = TouchstoneCodec.from_dataset(...)

ldm = TouchstoneLDataModule(
    source="FilterData/FilterBank",
    codec=codec,
    batch_size=32,
    val_ratio=0.2,
    test_ratio=0.1,
    scaler_in=StdScaler(dim=0),
    scaler_out=MinMaxScaler(dim=(0, 2)),
)

ldm.setup("fit")
train_loader = ldm.get_dataloader("train")
```

---

## Особенности

- Автоматическая подгонка скейлеров по тренировочной выборке
- Совместимость с `BaseLModule` и `BaseLMWithMetrics`
- Возможность ограничить размер датасета (`max_samples`) для быстрой отладки
- Обратимая постановка задачи (`swap_xy=True`)
- Предикт-режим готов к декодированию с `codec.decode(...)`)
