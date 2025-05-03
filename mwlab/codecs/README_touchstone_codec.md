# 📦 mwlab.codecs

Модуль `mwlab.codecs` содержит `TouchstoneCodec` — ключевой компонент библиотеки MWLab, который позволяет преобразовывать данные формата Touchstone в тензоры PyTorch и обратно.

## 🚀 Возможности

- Преобразование `TouchstoneData` → `(x_tensor, y_tensor, meta)`;
- Обратное восстановление `(y_pred, meta)` → `TouchstoneData`;
- Автоматическая генерация Codec из `TouchstoneDataset` (`from_dataset()`);
- Поддержка pickle-сериализации/десериализации (`dumps()`, `loads()`);
- Гибкая поддержка частичных S‑матриц (`Sij.part`);
- Совместимость с PyTorch‑пайплайнами (в т.ч. `TouchstoneTensorDataset`).

---

## 🛠 Пример использования

### 1. Автоматическая генерация кодека из датасета

```python
from mwlab.datasets.touchstone_dataset import TouchstoneDataset
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.transforms.s_transforms import S_Crop

ds = TouchstoneDataset(
    "Data/Filter12",
    x_keys=["w", "gap"],
    s_tf=S_Crop(1e9, 10e9),
)

codec = TouchstoneCodec.from_dataset(ds, components=["real", "imag"])
print(codec)
# TouchstoneCodec(x_keys=2, y_channels=8[imag,real], freq_pts=256, ports=2)
```

### 2. Кодирование примера в тензоры

```python
x, net = ds[0]
from mwlab.io.touchstone import TouchstoneData

ts = TouchstoneData(net, params=x)
x_t, y_t, meta = codec.encode(ts)

print(x_t.shape)   # (Dx,)
print(y_t.shape)   # (C, F)
```

### 3. Декодирование обратно в TouchstoneData

```python
ts_rec = codec.decode(y_pred=y_t, meta=meta)
print(ts_rec.network)
```

### 4. Сохранение и загрузка кодека

```python
# Сохраняем
with open("codec.pkl", "wb") as f:
    f.write(codec.dumps())

# Загружаем
with open("codec.pkl", "rb") as f:
    codec2 = TouchstoneCodec.loads(f.read())
```

---

## 🎯 Как формировать `y_channels`

`y_channels` — это список каналов, которые будут сериализованы из S‑матрицы.

Формат: `S<i>_<j>.<part>`, где:
- `<i>` и `<j>` — индексы портов (нумерация с 1);
- `<part>` — компонент, один из:

| Компонент | Описание                |
|-----------|-------------------------|
| `real`    | Действительная часть    |
| `imag`    | Мнимая часть            |
| `mag`     | Амплитуда |S|           |
| `db`      | Амплитуда в дБ          |
| `deg`     | Фаза в градусах         |

Примеры:
```python
y_channels = [
    "S1_1.real", "S1_1.imag",    # только диагональ
    "S2_1.db", "S2_1.deg",       # полная фаза в dB/deg
    "S1_2.mag"                   # амплитуда одного элемента
]
```

**Важно**: если `y_channels` не покрывает всю S‑матрицу, остальные элементы будут обнулены при `decode()`, но оригинальная матрица может быть сохранена в `meta["s_backup"]`.

---

## 📐 Формы тензоров

| Объект         | Форма          | Описание                            |
|----------------|----------------|-------------------------------------|
| `x_t`          | `(Dx,)`        | Вектор параметров                   |
| `y_t`          | `(C, F)`       | C каналов, F частотных точек        |
| `y_pred`       | `(C, F)`       | Предсказание модели                 |
| `meta`         | `dict`         | Метаданные для восстановления сети |

---

## 🧠 Умные особенности

- Поддержка `NaN` в параметрах X;
- Интерполяция частотной сетки при `force_resample=True`;
- Работа с нестандартным `z0` (скаляр/вектор);
- Чтение неполной информации через `s_backup`.

---

## 🧪 Юнит‑тесты

Полностью покрыт через `pytest`, включая:
- `encode()`/`decode()` на всех компонентах (`real`, `imag`, `db`, `deg`, `mag`);
- Частичную S-матрицу (`Sij`);
- Обратную совместимость `pickle`;
- Ресемплирование и восстановление `z0`, `comments`, `s_def` и др.