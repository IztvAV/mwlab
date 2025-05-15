
# 🗃️ `mwlab.io.backends`

Модуль `mwlab.io.backends` предоставляет унифицированный интерфейс для хранения и загрузки данных формата Touchstone через абстракцию `StorageBackend`. Поддерживает работу как с наборами отдельных файлов (`FileBackend`), монолитным контейнером (`HDF5Backend`), а также с данными, загруженными в оперативную память (`RAMBackend`,`SyntheticBackend`).

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

---

## 🧠 In‑Memory backend'ы: `RAMBackend`, `SyntheticBackend`

Эти backend'ы работают полностью в оперативной памяти и идеально подходят для:

- тестирования и синтетики;
- работы без дискового ввода-вывода;
- генерации обучающих выборок "на лету".

---

### 🧾 `RAMBackend` – готовый список TouchstoneData

Позволяет вручную собрать или загрузить список данных в память.

```python
from mwlab.io.backends import RAMBackend
from mwlab.io.touchstone import TouchstoneData

backend = RAMBackend()
backend.append(TouchstoneData.load("sample.s2p"))

# Или сразу из списка:
backend2 = RAMBackend([TouchstoneData.load(p) for p in Path("data").glob("*.s2p")])

# Сохранение/загрузка через pickle
backend2.dump_pickle("data.pkl")
restored = RAMBackend.load_pickle("data.pkl")
```

---

### 🧬 `SyntheticBackend` – генерация TouchstoneData на лету

Позволяет сгенерировать данные по функции `factory(idx)`.

```python
from mwlab.io.backends import SyntheticBackend

def coupling_matrix_factory(i: int) -> TouchstoneData:
    return synthesize_filter(i, order=6)

# Генерация с кешированием всех значений
backend_syn = SyntheticBackend(
        length=100_000,
        factory=coupling_matrix_factory,  # передаем ссылку на функцию
        cache=True,                       # сохраняем все в кэш
        workers=8                         # распараллеливаем генерацию данных 
)

# LRU-кеш на N последних вызовов
online = SyntheticBackend(
    length=10000,
    factory=lambda i: synthesize_filter(i),
    cache=256  # хранит только 256 последних
)
```

Варианты `cache`:
- `True` → предгенерация всех записей;
- `int` → LRU-кеш на N последних;
- `False` → без кеша, каждый вызов — новая генерация.

**Важно:** `SyntheticBackend` не поддерживает `load_pickle()`.


------

------

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
