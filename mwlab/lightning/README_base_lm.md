# mwlab.lightning.base_lm & base_lm_with_metrics

Модули `BaseLModule` и `BaseLMWithMetrics` реализуют обобщенную логику обучения Lightning‑моделей в задачах MWLab — с поддержкой как прямой, так и обратной регрессии, скейлинга, декодирования, инференса и логирования метрик.

---

## BaseLModule

```python
class BaseLModule(L.LightningModule)
```

### Основные возможности

- Прямая (`X → Y`) и обратная (`Y → X`) регрессия — через `swap_xy`
- Интеграция с `TouchstoneCodec` (автодекодирование, сериализация)
- Поддержка входных и выходных скейлеров (`scaler_in`, `scaler_out`)
- Методы инференса:
  - `predict_s(dict) → Network` — X → S
  - `predict_x(Network) → dict` — S → X

---

### Аргументы конструктора

| Параметр         | Описание                                                                 |
|------------------|--------------------------------------------------------------------------|
| `model`          | PyTorch-модель (`nn.Module`)                                              |
| `swap_xy`        | Решать обратную задачу (Y→X)                                              |
| `auto_decode`    | Автоматически декодировать `TouchstoneData` в `predict_step`             |
| `codec`          | Экземпляр `TouchstoneCodec`, используется при предсказании                |
| `scaler_in`      | Скейлер для нормализации X (должен быть `nn.Module`)                      |
| `scaler_out`     | Скейлер для нормализации Y                                                |
| `loss_fn`        | Функция потерь (по умолчанию `nn.MSELoss`)                               |
| `optimizer_cfg`  | Конфигурация оптимизатора (`{"name": "Adam", "lr": 1e-3}` и т.п.)         |
| `scheduler_cfg`  | Конфигурация планировщика обучения (например, ReduceLROnPlateau)          |

---

### Методы

| Метод                      | Назначение                                                        |
|---------------------------|-------------------------------------------------------------------|
| `forward(x)`              | Прямой проход (с учетом `scaler_in`)                              |
| `training_step`           | Этап обучения                                                     |
| `validation_step`         | Этап валидации                                                    |
| `test_step`               | Этап тестирования                                                 |
| `predict_step`            | Предсказание с учетом типа задачи и возможности автодекодирования |
| `predict_s(dict)`         | Прямая модель: X → S                                              |
| `predict_x(Network)`      | Обратная модель: S → X                                            |
| `configure_optimizers()`  | Настройка оптимизатора и LR‑scheduler                             |
| `state_dict()`            | Сохраняет модель + codec + скейлеры                               |
| `load_state_dict()`       | Восстанавливает все из checkpoint‑словаря                         |

---

### Самодостаточная загрузка

Вы можете загрузить готовую модель через:

```python
model = BaseLModule.load_from_checkpoint("best.ckpt", model=model_core)
```

При этом не требуется вручную передавать `codec=...` или `scaler_*` — они сериализуются вместе с моделью.

---

## BaseLMWithMetrics

```python
class BaseLMWithMetrics(BaseLModule)
```

Наследник `BaseLModule`, добавляющий логирование метрик (через `torchmetrics`).

### Аргументы

| Параметр | Описание                                                   |
|----------|------------------------------------------------------------|
| `metrics` | `MetricCollection` или `dict` с именованными метриками     |

Если не указано, по умолчанию используются:

- `MeanSquaredError`
- `MeanAbsoluteError`
- `R2Score`

---

### Поведение

- Метрики логируются на `validation_step` и `test_step`
- Метрики автоматически клонируются с префиксами `val_` и `test_`
- Все скейлеры и `TouchstoneCodec` сохраняются и восстанавливаются из checkpoint

---

## Пример использования

```python
from mwlab.lightning import BaseLMWithMetrics
from mwlab.nn.scalers import StdScaler, MinMaxScaler
from torchmetrics import MetricCollection, R2Score
import torch.nn as nn

metrics = MetricCollection({
    "mse": torchmetrics.MeanSquaredError(),
    "r2": R2Score()
})

model = BaseLMWithMetrics(
    model=MyNet(),
    codec=my_codec,
    scaler_in=my_scaler_in,
    scaler_out=my_scaler_out,
    metrics=metrics,
    optimizer_cfg={"name": "Adam", "lr": 1e-3},
)
```

---

## Инференс (предсказания)

```python
# Прямая модель (X → S)
net = model.predict_s({"a": 1.0, "b": 2.0})

# Обратная модель (S → X)
params = model.predict_x(my_rf_network)
```

---

## Интеграция

Модули интегрируются с `TouchstoneLDataModule` и `Trainer`:

```python
trainer = Trainer(max_epochs=20)
trainer.fit(model=lit_model, datamodule=ldm)
predictions = trainer.predict(model=lit_model, datamodule=ldm)
```

---
