# mwlab.lightning.base_lm & base_lm_with_metrics

Два связанных модуля, реализующих базовый `LightningModule` для обучения моделей в MWLab. Поддерживают прямую и обратную задачи (X → Y / Y → X), скейлинг, логирование метрик и декодирование в TouchstoneData.

---

## BaseLModule

```python
class BaseLModule(L.LightningModule)
```

Базовый Lightning-модуль для задачи регрессии с использованием Touchstone-данных. Поддерживает forward/backward, обучение и предсказания.

### Ключевые возможности

- Поддержка `swap_xy`: обучение на Y→X вместо X→Y
- Скейлинг входов и выходов (`scaler_in`, `scaler_out`)
- Декодирование предсказаний через `TouchstoneCodec.decode()`
- Поддержка checkpoint-совместимого `state_dict`

---

### Аргументы конструктора

| Параметр | Описание                                                                   |
|----------|----------------------------------------------------------------------------|
| `model` | PyTorch-модель (`nn.Module`)                                                 |
| `swap_xy` | `bool`, решать обратную задачу                                             |
| `auto_decode` | `bool`, в методе `predict()` автоматически декодировать в `TouchstoneData`|
| `codec` | `TouchstoneCodec` для `decode()`                                               |
| `scaler_in` / `scaler_out` | `nn.Module` с методами `fit`, `forward`, `inverse` |
| `loss_fn` | функция потерь (по умолчанию MSELoss)                                      |
| `optimizer_cfg` | словарь `{"name": ..., ...}`                                               |
| `scheduler_cfg` | словарь (опционально)                                                      |

---

### Методы

- `forward(x)` — применяет скейлер и модель
- `training_step / validation_step / test_step`
- `predict_step` — с авто-декодированием или возвратом тензоров/словаря
- `configure_optimizers()` — создает оптимайзер и `scheduler`
- `state_dict()` / `load_state_dict()` — сохраняет codec и скейлеры

---

## Пример

```python
module = BaseLModule(
    model=MyModel(),
    codec=codec,
    scaler_in=StdScaler(dim=0),
    scaler_out=MinMaxScaler(dim=(0, 1)),
    optimizer_cfg={"name": "Adam", "lr": 1e-3}
)
```

---

## BaseLMWithMetrics

```python
class BaseLMWithMetrics(BaseLModule)
```

Наследует `BaseLModule`, добавляя логирование метрик (`torchmetrics`).

### Аргумент

| Параметр | Описание |
|----------|----------|
| `metrics` | `dict` или MetricCollection (`mse`, `mae`, и т.п.) |

**По умолчанию добавлены метрики:**

- `MeanSquaredError`
- `MeanAbsoluteError`

---

### Поведение

- В `validation_step` и `test_step` автоматически логируются все заданные метрики
- Метрики клонируются с префиксами `val_` и `test_`
- Поддерживает те же скейлеры и `auto_decode`

---

## Пример

```python
from torchmetrics import MetricCollection, R2Score

metrics = MetricCollection({
    "mse": torchmetrics.MeanSquaredError(),
    "r2": R2Score()
})

module = BaseLMWithMetrics(
    model=MyModel(),
    codec=codec,
    metrics=metrics,
    scaler_in=StdScaler(),
    scaler_out=MinMaxScaler()
)
```

---

## Совместимость

- Идеально интегрируется с `TouchstoneLDataModule`
- Поддерживает predict-декодирование в TouchstoneData
- Все скейлеры и метрики сохраняются внутри checkpoint

---

## Сценарий использования

```python
trainer = Trainer(max_epochs=10)
trainer.fit(model=module, datamodule=ldm)

predictions = trainer.predict(model=module, datamodule=ldm)
```

---

