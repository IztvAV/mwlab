# 🔀 `mwlab.transforms`

Модуль `mwlab.transforms` содержит набор трансформаций для работы с:

- `x`‑параметрами (`X_...`) — словари скалярных параметров;
- `s`‑параметрами (`S_...`) — объекты `skrf.Network`.

Эти трансформы можно комбинировать, применять к датасетам или использовать для аугментации данных.

---

## 📦 Структура

- `x_transforms.py` – трансформы над скалярными параметрами (`dict`);
- `s_transforms.py` – трансформы над S‑матрицами (`skrf.Network`);
- `TComposite` – композиция нескольких трансформ.

---

## 🧠 Применение трансформ

```python
from mwlab.transforms import X_SelectKeys, S_Crop, S_AddNoise, TComposite

# X: оставляем только ключи 'w' и 'gap'
x_tf = X_SelectKeys(["w", "gap"])

# S: обрезаем диапазон частот и добавляем шум
s_tf = TComposite([
    S_Crop(1e9, 10e9),
    S_AddNoise(sigma_db=0.05, sigma_deg=0.5)
])
```

Эти трансформы можно передавать в `TouchstoneDataset(...)`.

---

## 🔧 `X_SelectKeys`

Оставляет только нужные ключи в словаре параметров.

```python
X_SelectKeys(["w", "gap"])(
    {"w": 1.2, "l": 0.9, "gap": 0.1}
)
# → {"w": 1.2, "gap": 0.1}
```

---

## 🔁 `TComposite`

Композиция трансформ: `x = tf(x)` применяется последовательно.

```python
TComposite([tf1, tf2, tf3])(x)
```

---

## ⚙️ Часто используемые `S_...` трансформы

| Трансформ              | Описание                                       |
|------------------------|------------------------------------------------|
| `S_Crop`               | Обрезка диапазона частот                       |
| `S_Resample`           | Интерполяция сетки частот                      |
| `S_AddNoise`           | Добавление шума (амплитуда + фаза)            |
| `S_PhaseShiftDelay`    | Общий фазовый сдвиг Δτ                         |
| `S_PhaseShiftAngle`    | Смещение фазы на фиксированный угол            |
| `S_Z0Shift`            | Сдвиг опорного импеданса                      |
| `S_DeReciprocal`       | Нарушение взаимности S[i,j] ≠ S[j,i]          |
| `S_Ripple`             | Синусоидальная ripple по частоте              |
| `S_MagSlope`           | Линейный наклон амплитуды по частоте          |

---

## 🧪 Пример аугментации

```python
from mwlab.transforms import TComposite, S_Crop, S_AddNoise, S_PhaseShiftDelay

s_tf = TComposite([
    S_Crop(1e9, 10e9),
    S_AddNoise(sigma_db=(0.05, 0.02), sigma_deg=0.5),
    S_PhaseShiftDelay(tau_ps=(-10, 5)),
])
```

---

## ⚠️ Распределения как параметры

Большинство трансформов принимают:

- фиксированное число: `0.1`
- нормальное распределение: `(μ, σ)`
- callable: `lambda: ...`

Пример:

```python
S_AddNoise(sigma_db=(0.0, 0.02))  # N(0, 0.02)
```

---

## 📦 Экспортируемые классы

```python
from mwlab.transforms import (
    S_Crop, S_Resample, S_AddNoise, S_PhaseShiftDelay,
    S_PhaseShiftAngle, S_DeReciprocal, S_Z0Shift,
    S_Ripple, S_MagSlope, X_SelectKeys, TComposite
)
```

---

## 🧪 Юнит-тесты

Для всех трансформов доступны юнит-тесты:
- Проверка сохранения формы `skrf.Network`;
- Проверка корректности частотной сетки после `S_Crop`, `S_Resample`;
- Проверка статистических свойств (`S_AddNoise`, `S_Z0Shift`, ...);
- Проверка поведения трансформ-композиции (`TComposite`).