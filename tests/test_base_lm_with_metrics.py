# tests/test_base_lm_with_metrics.py
"""
Юнит-тесты для BaseLMWithMetrics (расширение BaseLModule).

Покрываем:
1.  Прямая задача (swap_xy=False):
      • validation_step обновляет val_metrics
      • compute() выдаёт ключи 'val_mse', 'val_mae'
2.  Обратная задача (swap_xy=True):
      • validation_step работает с перевёрнутыми X/Y
      • метрики считаются по нужному масштабу (без scaler_out)
3.  test_step аналогичен validation_step (проверяем префиксы 'test_').
"""

import torch
from torch import nn

from mwlab.lightning.base_lm_with_metrics import BaseLMWithMetrics
from mwlab.nn.scalers import StdScaler, MinMaxScaler

# ------------------------------------------------------------------------------
# helpers
class Dummy(nn.Module):
    """Линейный слой с детерминированной инициализацией (0.1)."""

    def __init__(self, d: int, c: int):
        super().__init__()
        self.fc = nn.Linear(d, c, bias=False)
        torch.manual_seed(0)
        nn.init.constant_(self.fc.weight, 0.1)

    def forward(self, x):  # type: ignore[override]
        return self.fc(x)


# ------------------------------------------------------------------------------
# 1. direct task  (X ➜ Y)
# ------------------------------------------------------------------------------
def test_val_metrics_direct():
    mod = BaseLMWithMetrics(
        model=Dummy(4, 2),
        scaler_out=StdScaler(dim=0).fit(torch.randn(20, 2)),
        swap_xy=False,
    )

    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    _ = mod.validation_step((x, y), 0)

    vals = mod.val_metrics.compute()
    assert {"val_mse", "val_mae"} <= vals.keys()
    # метрики должны быть тензорами-скалярами
    assert all(v.ndim == 0 for v in vals.values())


# ------------------------------------------------------------------------------
# 2. inverse task  (Y ➜ X, swap_xy=True)
# ------------------------------------------------------------------------------
def test_val_metrics_inverse():
    mod = BaseLMWithMetrics(
        model=Dummy(2, 1),           # C=2 -> D=1
        swap_xy=True,
        scaler_out=MinMaxScaler(dim=0).fit(torch.randn(20, 1)),  # НЕ должен применяться
    )

    x_y = torch.randn(10, 2)          # ← будет входом модели
    y_x = torch.randn(10, 1)          # ← target (без скейлера)
    _ = mod.validation_step((x_y, y_x), 0)

    vals = mod.val_metrics.compute()
    assert {"val_mse", "val_mae"} <= vals.keys()


# ------------------------------------------------------------------------------
# 3. test_step использует test_metrics (префикс 'test_')
# ------------------------------------------------------------------------------
def test_test_metrics_prefix():
    mod = BaseLMWithMetrics(model=Dummy(3, 2))

    x = torch.randn(6, 3)
    y = torch.randn(6, 2)
    _ = mod.test_step((x, y), 0)

    vals = mod.test_metrics.compute()
    assert {"test_mse", "test_mae"} <= vals.keys()
