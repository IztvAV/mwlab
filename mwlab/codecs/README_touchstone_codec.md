# 📦 mwlab.codecs — TouchstoneCodec

Модуль `mwlab.codecs.touchstone_codec` предоставляет `TouchstoneCodec` — универсальный преобразователь данных Touchstone (.sNp) в тензоры PyTorch и обратно.
Он используется для подготовки данных к машинному обучению и для восстановления сетей из предсказаний модели.

---

## 🚀 Возможности

- Преобразование `TouchstoneData` → `(x_tensor, y_tensor, meta)` и обратно;
- Отдельное кодирование/декодирование X (`encode_x`, `decode_x`) и S (`encode_s`, `decode_s`);
- Автоматическая генерация Codec по `TouchstoneDataset` (`from_dataset`);
- Гибкие стратегии резервного копирования S‑матрицы (`backup_mode`);
- Полная поддержка `pickle`‑сериализации (`dumps()`, `loads()`);
- Совместимость с любыми PyTorch‑пайплайнами.
---

## ⚙️ Стратегии `backup_mode`

| Режим     | Описание                                                        |
|-----------|-----------------------------------------------------------------|
| `none`    | Ничего не сохраняется, непокрытые компоненты → `0+0j`          |
| `missing` | Сохраняются только отсутствующие `Sij` в `meta['s_missing']`   |
| `full`    | Сохраняется вся `S`‑матрица в `meta['s_backup']`               |

Пример:
```python
codec = TouchstoneCodec(
    x_keys=["w", "gap"],
    y_channels=["S1_1.real"],
    freq_hz=freq_vec,
    backup_mode="full",
)
```

---

## 🛠 Примеры использования

### 1. Автоматическая генерация из датасета

```python
from mwlab.datasets import TouchstoneDataset
from mwlab.codecs import TouchstoneCodec

ds = TouchstoneDataset("Data/Filter12")
codec = TouchstoneCodec.from_dataset(ds, components=["real", "imag", "gd"])
print(codec)
# TouchstoneCodec(x_keys=2, y_channels=12[gd,imag,real], freq_pts=256, ports=2)
```

### 2. Кодирование `TouchstoneData`

```python
from mwlab.io import TouchstoneData

ts = TouchstoneData(net, params={"w": 1.0, "gap": 2.0})
x_t, y_t, meta = codec.encode(ts)   # y_t содержит real / imag / gd
```

### 3. Обратное декодирование

```python
ts_restored = codec.decode(y_pred=y_t, meta=meta)
```

### 4. Частичное кодирование

```python
x_tensor = codec.encode_x({"w": 1.0, "gap": 2.0})

s_tensor, meta_s = codec.encode_s(net)   # содержит GD‑каналы, если заданы
net_rec = codec.decode_s(s_tensor, meta_s)
```

### 5. Сериализация / десериализация

```python
with open("codec.pkl", "wb") as f:
    f.write(codec.dumps())

with open("codec.pkl", "rb") as f:
    codec2 = TouchstoneCodec.loads(f.read())
```

---

## 📐 Форматы тензоров

| Объект         | Форма          | Описание                            |
|----------------|----------------|-------------------------------------|
| `x_t`          | `(Dx,)`        | Параметры из X‑пространства         |
| `y_t`          | `(C, F)`       | S‑каналы в частотной области        |
| `meta`         | `dict`         | Метаданные для реконструкции сети   |

---

## 🎯 Формат `y_channels`

Каждый канал описывается как `S<i>_<j>.<part>`, где:

* `i`, `j` — номера портов, начиная с `1`;
* `part` — компонент: `real`, `imag`, `mag`, `db`, `deg`, **`gd`** (group delay).

Примеры:

```python
y_channels = [
    "S1_1.real", "S1_1.imag", "S1_1.gd",   # S11
    "S2_1.db",   "S2_1.deg",               # S21
    "S1_2.mag"                            # S12
]
```

*Компоненты можно миксовать произвольно; форма `y_t` остаётся `(C, F)`.*

---

## 🧠 Особенности

* Поддержка `NaN` в X‑параметрах;
* Автоматическое выравнивание частотной сетки (`force_resample`);
* Восстановление `unit`, `z0`, `s_def`, `comments`;
* Групповая задержка (`gd`) игнорируется при `decode()` — она не нужна для восстановления `S`;
* Полная обратимость при `backup_mode="full"`.

---

## 🧪 Тесты

`pytest`‑набор покрывает:

* Прямой и обратный проход (`encode`/`decode`), включая `gd`;
* Все режимы `backup_mode`;
* encode_x / decode_x / encode_s / decode_s;
* Проверку частотной сетки и ресэмплинга;
* Сериализацию / десериализацию (`dumps`, `loads`);
* Корректную обработку `z0`, `unit`, `comments`, `params`.
