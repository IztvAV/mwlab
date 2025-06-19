# mwlab.nn.scalers

Модуль `mwlab/nn/scalers.py` предоставляет **три** готовых класса
масштабирования данных для PyTorch‑моделей:

| Класс           | Алгоритм                     | Устойчивость к выбросам |
|-----------------|------------------------------|-------------------------|
| `StdScaler`     | (x − mean) / std             | низкая                  |
| `MinMaxScaler`  | линейное [min,max] → [a,b]   | низкая                  |
| `RobustScaler`  | (x − median) / IQR           | **высокая**             |

Все скейлеры наследуются от общего `_Base`, работают на любом `device`/`dtype`,
поддерживают агрегирование статистик по произвольным осям (`dim`) и
корректно сохраняются через `state_dict()`.

---

## Быстрый старт

```python
import torch
from mwlab.nn.scalers import RobustScaler  # или StdScaler, MinMaxScaler

x = torch.randn(256, 64)
x[::30] *= 20                               # имитируем выбросы

scaler = RobustScaler(dim=0)               # агрегируем статистики по features
scaler.fit(x)

z = scaler(x)           # нормализованные данные
x_rec = scaler.inverse(z)
assert torch.allclose(x, x_rec, atol=1e-5)
```

---

## 1. Базовый класс `_Base`

```python
class _Base(nn.Module)
```

* **Параметры конструктора**
  * `dim` — int | Sequence[int] | None. Оси для агрегации статистик (по‑умолчанию 0).
  * `eps` — float. Защита от деления на 0 (default = 1e‑12).

* **Методы**
  * `fit(data, dim=None)` — вычисляет статистики.
  * `forward(x)` — применяет трансформацию.
  * `inverse(z)` — обратное преобразование.

---

## 2. StdScaler

Нормализация по среднему и стандартному отклонению:

```python
z = (x - mean) / std
```

```python
scaler = StdScaler(dim=(0, 2), unbiased=False)
scaler.fit(x)
z = scaler(x)
x_back = scaler.inverse(z)
```

---

## 3. MinMaxScaler

Линейное преобразование в диапазон `[a, b]`:

```python
y = (x - min) / (max - min) * (b - a) + a
```

```python
scaler = MinMaxScaler(dim=0, feature_range=(-1.0, 1.0))
scaler.fit(x)
y = scaler(x)
x_back = scaler.inverse(y)
```

---

## 4. **RobustScaler** 

Устойчивое к выбросам масштабирование по медиане и интерквантильному
размаху (IQR):

```python
z = (x - median) / (Q_high - Q_low)
```

*По умолчанию `quantile_range = (25, 75)`, но можно задать любой.*

```python
scaler = RobustScaler(dim=0, quantile_range=(10, 90))
scaler.fit(x)
z = scaler(x)
x_back = scaler.inverse(z)
```

### Почему «robust»

* Медиана и IQR не «едут», даже если 5–10 % точек экстремальны.
* Подходит для каналов с резкими всплесками (например, group delay).

---

## 5. Сериализация

Любой скейлер можно сохранять вместе с моделью:

```python
torch.save({
    "model": model.state_dict(),
    "scaler": scaler.state_dict(),
}, "checkpoint.pt")

# Загрузка
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])

scaler = RobustScaler()      # тот же класс
scaler.load_state_dict(checkpoint["scaler"])
```

---

## 6. Советы по применению

1. Для **стационарных** распределений без выбросов используйте `StdScaler`.
2. Если важно жёстко ограничить диапазон — берите `MinMaxScaler`.
3. При **пикирующих** величинах (delay, ratio dB, power) лучший выбор — `RobustScaler`.
