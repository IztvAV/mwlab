# 📚 `mwlab.datasets`

Модуль `mwlab.datasets` содержит удобные классы для подготовки и загрузки S‑параметров и параметров X в формате PyTorch-датасетов. Поддерживает:

- работу с `TouchstoneData` из любых backend‑ов;
- трансформации (S / X);
- кэширование и переупорядочивание X ↔ Y;
- преобразование в тензоры (`TouchstoneTensorDataset`);
- поддержку `meta`‑информации.

---

## 🧱 Состав

| Класс                      | Назначение                                                |
|----------------------------|------------------------------------------------------------|
| `TouchstoneDataset`        | Базовый датасет (X: dict, Y: skrf.Network)                |
| `TouchstoneTensorDataset`  | Расширение: выдаёт пары `(X_t, Y_t)` (тензоры)             |
| `_CachedDataset` (mixin)   | Кэширование + meta + swap XY                              |

---

## 🧪 Пример (сырой датасет)

```python
from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.io.backends import FileBackend
from mwlab.transforms import S_Crop, X_SelectKeys

ds = TouchstoneDataset(
    FileBackend("Data/Filter12"),
    x_keys=["w", "gap"],
    x_tf=X_SelectKeys(["w", "gap"]),
    s_tf=S_Crop(1e9, 10e9)
)

x, s = ds[0]
print(x)       # {'w': ..., 'gap': ...}
print(s)       # skrf.Network
```

---

## 🔁 Пример (тензорный датасет)

```python
from mwlab.datasets.touchstone_tensor_dataset import TouchstoneTensorDataset
from mwlab.codecs.touchstone_codec import TouchstoneCodec

# создаём базовый датасет
base = TouchstoneDataset("Data/Filter12")
codec = TouchstoneCodec.from_dataset(base)

ds = TouchstoneTensorDataset(
    source="Data/Filter12",
    codec=codec,
    swap_xy=False,
    return_meta=True,
    cache_size=512
)

x_t, y_t, meta = ds[0]
print(x_t.shape)  # (Dx,)
print(y_t.shape)  # (C, F)
print(meta["params"])
```

---

## 🔁 Поведение `swap_xy`

Параметр `swap_xy=True` позволяет легко переключаться между задачами:

- `False`: `(x, y)` → прямое моделирование (X → S);
- `True`: `(y, x)` → обратная задача (S → X).

---

## 🧠 `return_meta`

Если включено, возвращается `(x, y, meta)`:

- `meta["params"]` — исходные параметры X;
- `meta["orig_path"]` — путь к исходному файлу;
- `meta["unit"]`, `z0`, `comments`, ...

---

## 🗃️ Кэширование (`_CachedDataset`)

Поддерживается LRU-кэш по индексам:

```python
ds = TouchstoneTensorDataset(..., cache_size=512)
```

- Автоматически отключается внутри PyTorch DataLoader (`get_worker_info()`);
- Эффективен при многократном доступе к одним и тем же элементам.

---

## 🧪 Юнит-тесты

Покрываются:

- корректность длины и форм `x`, `y`;
- работа с `swap_xy` и `return_meta`;
- поведение LRU‑кэша;
- совместимость с `TouchstoneCodec`;
- NaN для отсутствующих параметров;
- round-trip encode/decode.