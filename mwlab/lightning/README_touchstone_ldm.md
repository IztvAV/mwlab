# mwlab.lightning.touchstone_ldm

Модуль `touchstone_ldm.py` предоставляет LightningDataModule-обертку для работы с Touchstone-данными — файлов формата *.sNp, используемых в ВЧ и СВЧ-моделировании (например, S-параметры фильтров, антенн и пр.).

Основан на `TouchstoneTensorDataset` и обеспечивает полную поддержку типового ML-пайплайна: разбиение на train/val/test, скейлинг, predict-режим, поддержку метаданных.

---

## TouchstoneLDataModule

```python
class TouchstoneLDataModule(L.LightningDataModule)
```

### Основное предназначение

- Поддержка чтения Touchstone-данных из различных backend-ов (`FileBackend`, `HDF5Backend`)
- Разбиение на тренировочную, валидационную и тестовую части
- Предобработка входов и выходов с помощью скейлеров
- Возможность постановки обратной задачи (swap X ↔ Y)
- Режим предсказания с возвратом `meta` для декодирования в `TouchstoneData`

---

## Аргументы конструктора

| Параметр | Тип               | Описание |
|----------|-------------------|----------|
| `source` | `str`/`Path`/`StorageBackend` | Путь к каталогу с *.sNp, .h5-файлу или объекту backend-а |
| `codec` | `TouchstoneCodec` | Кодек для преобразования данных в тензоры |
| `batch_size` | `int`             | Размер батча |
| `val_ratio` | `float`           | Доля валидационного набора |
| `test_ratio` | `float`           | Доля тестового набора |
| `max_samples` | `int`/`None` | Ограничение по количеству примеров |
| `swap_xy` | `bool`            | Обратная задача (Y→X) |
| `scaler_in` | `nn.Module`/`None` | Скейлер признаков |
| `scaler_out` | `nn.Module`/`None` | Скейлер выходов |
| `cache_size` | `int`/`None` | Размер LRU-кеша |
| `base_ds_kwargs` | `dict`/`None` | Аргументы для базового `TouchstoneDataset` |

---

## Методы

### `setup(stage: str)`

Инициализирует подвыборки набора данных:

- `'fit'` → train/val/test + подгонка скейлеров
- `'validate'`, `'test'` → требует предварительного вызова `'fit'`
- `'predict'` → отдельный датасет с `return_meta=True`

### `train_dataloader() / val_dataloader() / test_dataloader() / predict_dataloader()`

Возвращают  `DataLoader` для соответствующих (под)наборов данных.

---

## Режим `predict`

- В режиме `predict`, датасет автоматически создается с `return_meta=True`
- Коллатор `collate_fn` сохраняет `meta` в виде списка словарей
- Это позволяет `LightningModule.predict_step()` декодировать результат в `TouchstoneData`

---

## Пример использования

```python
from mwlab.lightning.touchstone_ldm import TouchstoneLDataModule
from mwlab.codecs.touchstone_codec import TouchstoneCodec

# создаем codec по датасету
codec = TouchstoneCodec.from_dataset(...)

# создаем LDataModule
ldm = TouchstoneLDataModule(
    source="Data/FilterBank",
    codec=codec,
    batch_size=32,
    val_ratio=0.2,
    test_ratio=0.1,
    scaler_in=StdScaler(dim=0),
    scaler_out=MinMaxScaler(dim=(0, 1)),
)

ldm.setup("fit")
train_loader = ldm.train_dataloader()
```

---

## Особенности

- Скейлеры автоматически обучаются (`fit()`) на тренировочной части
- `predict_dataloader()` возвращает `(X, Y, meta)` — готово к декодированию
- Совместим с `BaseLModule` и `BaseLMWithMetrics` из `mwlab.lightning`
