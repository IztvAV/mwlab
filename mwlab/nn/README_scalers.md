# mwlab.nn.scalers

Модуль `scalers.py` предоставляет классы нормализации данных для моделей PyTorch. Скейлеры реализованы как `torch.nn.Module` и поддерживают агрегирование статистик по произвольным осям, работу на любом `device`/`dtype`, а также совместимы с пайплайнами Lightning и TorchScript.

## Возможности

- Нормализация по среднему и стандартному отклонению (`StdScaler`)
- Приведение к заданному диапазону (`MinMaxScaler`)
- Поддержка агрегирования по любым измерениям (`dim`)
- Автоматическая трансформация на `fit()` / `forward()` / `inverse()`
- Совместимость с `state_dict()` / `load_state_dict()`

---

## Базовый класс: `_Base`

```python
class _Base(nn.Module)
```

Абстрактный базовый класс. Определяет базовую логику нормализации и буферов. Пользователю напрямую не нужен, но используется в `StdScaler` и `MinMaxScaler`.

**Параметры конструктора:**

- `dim`: `int | Sequence[int] | None` – Оси для агрегации статистик. По умолчанию — 0.
- `eps`: `float`, по умолчанию `1e-12` – Число, предотвращающее деление на 0.

**Методы:**

- `fit(data, dim=None)` – вычисляет статистики по данным
- `forward(x)` – применяет прямое преобразование
- `inverse(z)` – обратное преобразование

---

## StdScaler

```python
class StdScaler(_Base)
```

Нормализует входной тензор по формуле:

```text
z = (x - mean) / std
```

### Пример:

```python
scaler = StdScaler(dim=0)
scaler.fit(data)
z = scaler(data)
x = scaler.inverse(z)
```

## MinMaxScaler

```python
class MinMaxScaler(_Base)
```

Масштабирует данные линейно в диапазон `[a, b]`:

```text
y = (x - min) / (max - min) * (b - a) + a
```

### Пример:

```python
scaler = MinMaxScaler(dim=0, feature_range=(-1.0, 1.0))
scaler.fit(data)
y = scaler(data)
x = scaler.inverse(y)
```

**Дополнительные параметры:**

- `feature_range`: `Tuple[float, float]`, по умолчанию `(0.0, 1.0)`

**Буферы:**

- `data_min`
- `data_range`

---

## Типовой сценарий использования

```python
import torch
from mwlab.nn.scalers import StdScaler

data = torch.randn(100, 16)
scaler = StdScaler(dim=0)
scaler.fit(data)

# Прямое преобразование
z = scaler(data)

# Обратное преобразование
restored = scaler.inverse(z)

assert torch.allclose(data, restored, atol=1e-5)
```

---

## Особенности

- Поддержка трансформаций на батчах, спектрах и многомерных тензорах.
- `fit()` не обучаемый (параметры не требуют градиента).
- Совместим с `state_dict()` – можно сохранять и загружать вместе с моделью.
- Подходит для использования в PyTorch Lightning, включая `predict_step`.
