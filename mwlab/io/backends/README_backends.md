# 🗃️ `mwlab.io.backends`

Модуль `mwlab.io.backends` предоставляет унифицированный интерфейс для хранения и загрузки данных формата Touchstone через абстракцию `StorageBackend`. Поддерживает работу как с наборами отдельных файлов (`FileBackend`), так и с монолитным контейнером (`HDF5Backend`).

---

## 🔑 Интерфейс: `StorageBackend`

Базовый абстрактный класс с минимальным контрактом:

```python
class StorageBackend:
    def __len__(self) -> int: ...
    def read(self, idx: int) -> TouchstoneData: ...
    def append(self, ts: TouchstoneData) -> None: ...
```

---

## 📁 `FileBackend` – набор *.sNp‑файлов

Простой backend, работающий с директориями, содержащими Touchstone-файлы (`*.s2p`, `*.sNp`, ...).

### Пример:

```python
from mwlab.io.backends import FileBackend

backend = FileBackend(root="Data/Filter12")
print(len(backend))         # количество файлов
ts = backend.read(0)        # TouchstoneData
```

---

## 📦 `HDF5Backend` – контейнерный формат

Позволяет хранить все `TouchstoneData` в одном `.h5`‑файле с поддержкой:

- чтения и дозаписи (`mode='r' | 'a' | 'w'`);
- структурированного хранения: `/samples/0`, `/samples/1`, ...;
- пользовательских параметров — в `attrs`;
- метаданных (`unit`, `s_def`, `z0`, `comments`, ...);
- поддержки `swmr` (safe concurrent read).

### Пример создания и записи:

```python
from mwlab.io.backends import HDF5Backend
from mwlab.io.touchstone import TouchstoneData
import skrf as rf
import numpy as np

with HDF5Backend("train.h5", mode="w") as backend:
    f = rf.Frequency(1, 10, 5, "GHz")
    s = np.zeros((5, 2, 2), dtype=complex)
    net = rf.Network(frequency=f, s=s)
    ts = TouchstoneData(net, params={"w": 1.2, "gap": 0.3})
    backend.append(ts)
```

### Пример чтения:

```python
with HDF5Backend("train.h5", mode="r") as backend:
    ts = backend.read(0)
    print(ts.params)
    print(ts.network)
```

### SWMR режим

- При `mode='r'` автоматически активируется `swmr=True` для безопасного параллельного чтения;
- Метод `.refresh()` (внутренний) обновляет список доступных записей без закрытия файла.

---

## 📌 Совместимость с TouchstoneDataset

Любой backend можно передать в `TouchstoneDataset`:

```python
from mwlab.datasets.touchstone_dataset import TouchstoneDataset

ds = TouchstoneDataset(HDF5Backend("train.h5", "r"))
```

---

## 🧩 Расширяемость

Можно реализовать свой backend, унаследовавшись от `StorageBackend`:

```python
class MyBackend(StorageBackend):
    def __len__(self): ...
    def read(self, idx): ...
    def append(self, ts): ...
```

---

## 📁 Структура данных в HDF5

```
/samples/0/
    ├── s            # (F, P, P) complex64
    ├── f            # (F,) float64
    ├── unit         # ascii
    ├── s_def        # ascii
    ├── z0           # (F, P) complex64
    ├── comments     # (N,) uint8 (если есть)
    └── attrs[...]   # параметры (w, gap, ...) в виде атрибутов
```

---

## 🧪 Покрытие тестами

`HDF5Backend` и `FileBackend` тестируются через:
- загрузку и сохранение;
- round-trip `TouchstoneData`;
- сохранность параметров и сетки;
- поведение в `TouchstoneDataset`.