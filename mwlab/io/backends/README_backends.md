
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

backend = FileBackend(root="FilterData/Filter12")
print(len(backend))         # количество файлов
ts = backend.read(0)        # TouchstoneData
```

---

## 📦 `HDF5Backend` – контейнерный формат

Позволяет хранить все `TouchstoneData` в одном `.h5`‑файле с поддержкой:

- режимов чтения и записи (`mode='r' | 'a' | 'w'`);
- хранения записей как подгрупп: `/samples/0`, `/samples/1`, ...;
- общей или индивидуальной частотной сетки (`/common_f`, `/samples/X/f`);
- произвольных пользовательских параметров — в `attrs`;
- метаданных (`unit`, `s_def`, `z0`, `comments`, ...);
- безопасного параллельного чтения (SWMR);
- опционального **режима in_memory**, превращающего HDF5 в кэш‑backend.

---

### Режимы работы

- `mode="r"` — только чтение; активирует SWMR (если `in_memory=False`);
- `mode="w"` — перезаписывает существующий файл;
- `mode="a"` — дозапись в конец;
- `in_memory=True` — загружает все записи в оперативную память при инициализации (только для чтения).

---

### Пример записи:

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

---

### Пример чтения:

```python
with HDF5Backend("train.h5", mode="r") as backend:
    ts = backend.read(0)
    print(ts.params)
    print(ts.network)
```

---

### Пример in-memory режима:

```python
# Загружает ВСЕ записи при старте, далее работает из RAM
backend = HDF5Backend("train.h5", mode="r", in_memory=True)
print(len(backend))
ts = backend.read(0)
```

---

## 📁 Структура данных в HDF5

```
/samples/0/
    ├── s            # (F, P, P) complex64 – S‑матрица
    ├── f            # (F,) float64        – индивидуальная частотная сетка (опц.)
    ├── unit         # ascii ('Hz', 'GHz', ...)
    ├── s_def        # ascii ('S', 'T', ...)
    ├── z0           # (P,) float64        – опорные сопротивления
    ├── comments     # bytes               – UTF‑8 строки через 
 (опц.)
    └── attrs[...]   # пользовательские параметры (в атрибутах группы)
/common_f            # (F,) float64        – общая частотная сетка (если есть)
```

---

## 🧪 Покрытие тестами

`HDF5Backend` и `FileBackend` тестируются через:

- запись и последующее чтение (`append()` → `read()`);
- проверку точности восстановления сетки и параметров;
- работу с `TouchstoneDataset`;
- поддержку `swmr`‑режима и `refresh()`;
- отдельные тесты для `in_memory=True`:
  - загрузка всех записей;
  - запрет на `append`;
  - ускоренное чтение без обращения к файлу.

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
