# 📡 `mwlab.io.touchstone`

Модуль `mwlab.io.touchstone` предоставляет класс `TouchstoneData` — удобный контейнер для работы с S‑параметрами, построенный поверх `skrf.Network`. Поддерживает загрузку/сохранение Touchstone‑файлов и сериализацию в формат NumPy‑словарей.

---

## 🧱 Структура

### `TouchstoneData` содержит:

- `network` — объект `skrf.Network`
- `params` — словарь пользовательских параметров (`dict[str, float|str]`)
- `path` — путь к исходному файлу (если задан)

---

## 📥 Загрузка из файла

```python
from mwlab.io.touchstone import TouchstoneData

ts = TouchstoneData.load("Data/sample.s2p")
print(ts.params)     # параметры из "! Parameters = {...}"
print(ts.network)    # skrf.Network
```

---

## 💾 Сохранение в файл

```python
ts.save("out_dir/new_sample.s2p")
```

При этом будет добавлена строка `! Parameters = {...}` в начало файла.

---

## 📦 Создание с нуля

```python
import skrf as rf
import numpy as np

f = rf.Frequency(1, 10, 5, "GHz")
s = np.zeros((5, 2, 2), dtype=complex)
net = rf.Network(frequency=f, s=s)

ts = TouchstoneData(network=net, params={"w": 1.2, "gap": 0.3})
```

---

## 🔁 Сериализация в NumPy-словарь

```python
dct = ts.to_numpy()
```

Ключи:

- `"s"` – S‑матрица: `(F, P, P)` complex64
- `"f"` – частоты: `(F,)` float64
- `"meta/unit"`, `"meta/z0"`, `"meta/s_def"` — метаданные сети
- `"meta/comments"` — комментарии (если есть)
- `"param/<key>"` — пользовательские параметры

Можно использовать для хранения в HDF5, LMDB и др.

---

## ↩️ Восстановление из словаря

```python
ts2 = TouchstoneData.from_numpy(dct)
```

---

## 🧠 Извлечение параметров

Поддерживаются параметры из строки:

```text
! Parameters = {w=1.2; gap=0.3}
```

Они извлекаются как:

- из `network.comments`
- либо напрямую из файла (`_params_from_file`)

---

## 🧪 Пример использования с HDF5

```python
d = ts.to_numpy()

import h5py
with h5py.File("sample.h5", "w") as f:
    for k, v in d.items():
        f.create_dataset(k, data=v)

# чтение обратно
with h5py.File("sample.h5", "r") as f:
    d2 = {k: f[k][()] for k in f}
    ts2 = TouchstoneData.from_numpy(d2)
```

---

## 🧪 Тестирование

Покрывается:

- загрузка и сохранение Touchstone‑файлов;
- извлечение параметров из комментариев;
- сериализация и восстановление через `.to_numpy()` / `.from_numpy()`;
- round-trip корректность (`s`, `f`, `params`, `unit`, ...).