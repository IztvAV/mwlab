# 📦 mwlab.codecs — TouchstoneCodec

Модуль `mwlab.codecs.touchstone_codec` предоставляет `TouchstoneCodec` — универсальный преобразователь данных Touchstone (.sNp) в тензоры PyTorch и обратно. Он используется для подготовки данных к машинному обучению, а также восстановления параметров и сетей из предсказаний модели.

---

## 🚀 Возможности

- Преобразование `TouchstoneData` → `(x_tensor, y_tensor, meta)`;
- Обратное восстановление `(y_pred, meta)` → `TouchstoneData`;
- Отдельное кодирование/декодирование по X (`encode_x`, `decode_x`);
- Отдельное кодирование/декодирование по S (`encode_s`, `decode_s`);
- Автоматическая генерация Codec из `TouchstoneDataset` (`from_dataset`);
- Гибкие стратегии резервного копирования S-матрицы (`backup_mode`);
- Полная поддержка `pickle`-сериализации (`dumps()`, `loads()`);
- Совместимость с PyTorch‑пайплайнами.

---

## ⚙️ Стратегии `backup_mode`

Если `y_channels` не покрывает всю `S`‑матрицу, можно указать одну из стратегий:

| Режим     | Описание                                                       |
|-----------|----------------------------------------------------------------|
| `none`    | Ничего не сохраняется, непокрытые компоненты → `0+0j`         |
| `missing` | Сохраняются только отсутствующие `Sij` в `meta['s_missing']`  |
| `full`    | Сохраняется вся `S`‑матрица в `meta['s_backup']`              |

Пример:
```python
codec = TouchstoneCodec(
    x_keys=["w", "gap"],
    y_channels=["S1_1.real"],
    freq_hz=freq_vec,
    backup_mode="full"
)
```

---

## 🛠 Примеры использования

### 1. Автоматическая генерация из датасета

```python
from mwlab.datasets import TouchstoneDataset
from mwlab.codecs import TouchstoneCodec

ds = TouchstoneDataset("FilterData/Filter12")
codec = TouchstoneCodec.from_dataset(ds, components=["real", "imag"])
print(codec)
# TouchstoneCodec(x_keys=2, y_channels=8[imag,real], freq_pts=256, ports=2)
```

### 2. Кодирование `TouchstoneData`

```python
from mwlab.io import TouchstoneData

ts = TouchstoneData(net, params={"w": 1.0, "gap": 2.0})
x_t, y_t, meta = codec.encode(ts)
```

### 3. Обратное декодирование

```python
ts_restored = codec.decode(y_pred=y_t, meta=meta)
```

### 4. Частичное кодирование (только X или S)

```python
x_tensor = codec.encode_x({"w": 1.0, "gap": 2.0})
s_tensor, meta = codec.encode_s(net)
```

```python
params = codec.decode_x(x_tensor)
net = codec.decode_s(s_tensor, meta)
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

- `i`, `j` — номера портов (нумерация с 1);
- `part` — компонент: `real`, `imag`, `mag`, `db`, `deg`.

Пример:
```python
y_channels = [
    "S1_1.real", "S1_1.imag",
    "S2_1.db", "S2_1.deg",
    "S1_2.mag"
]
```

---

## 🧠 Особенности

- Поддержка `NaN` в параметрах `X`;
- Автоматическое приведение частотной сетки (`force_resample`);
- Восстановление сети с `unit`, `z0`, `s_def`, `comments`;
- Гибкое управление пропущенными компонентами S‑матрицы;
- Полная обратимость при `backup_mode="full"`.

---

## 🧪 Тесты и надежность

Полный набор тестов на `pytest` покрывает:

- Прямой и обратный проход (`encode`/`decode`);
- Режимы `backup_mode`: none / missing / full;
- Отдельные encode_x / decode_x / encode_s / decode_s;
- Ошибки при несовпадении частот (без ресемплинга);
- Проверку сериализации / десериализации (`dumps`, `loads`);
- Восстановление `z0`, `unit`, `comments`, `params`.

---

## 📎 Связанные модули

- `mwlab.io.touchstone.TouchstoneData`
- `mwlab.datasets.touchstone_dataset.TouchstoneDataset`
- `mwlab.transforms.s_transforms`