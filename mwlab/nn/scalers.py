# mwlab/nn/scalers.py
"""
Скейлеры для тензоров PyTorch.

Данный модуль реализует несколько вариантов нормализации / масштабирования данных,
аналогичных трансформерам из scikit-learn, но с поддержкой:

* произвольного выбора осей для вычисления статистик,
* корректной работы с любыми dtype / device,
* хранения параметров в buffer (совместимость с .to(), state_dict и т. п.).

Доступные классы
----------------
* StdScaler
    Стандартизация по среднему и стандартному отклонению:
        forward(x) = (x - mean) / std
        inverse(z) = z * std + mean

* MinMaxScaler
    Линейное масштабирование данных в заданный диапазон [a, b]:
        forward(x) = (x - data_min) / (data_max - data_min) * (b - a) + a
        inverse(y) = (y - a) / (b - a) * (data_max - data_min) + data_min
    Примечание: чувствителен к выбросам.


* RobustScaler
    Масштабирование, устойчивое к выбросам (аналог sklearn.RobustScaler).
    Центрирование по медиане и деление на межквартильный размах (IQR):
        forward(x) = (x - median) / IQR,  IQR = Q_high - Q_low
        inverse(z) = z * IQR + median

* QuantileMinMaxScaler
    Робастное масштабирование в [a, b] по выбранным квантилям.
    Вместо min/max берутся q_low и q_high (например 1% и 99%),
    опционально возможен клиппинг хвостов:
         forward(x) = (clip(x, q_low, q_high) - q_low) / (q_high - q_low) * (b - a) + a
        inverse(y) = (y - a) / (b - a) * (q_high - q_low) + q_low
    Подходит, когда нужен строгий диапазон [0, 1] при наличии выбросов.

Общее
-----
- Все классы — torch.nn.Module с буферами статистик; поддерживают fit → forward → inverse чейнинг.
- Параметр ``dim`` задаёт оси для агрегации статистики; ``None`` означает «не агрегируем».
- Все редукции выполняются с ``keepdim=True`` и потому статистики broadcast-совместимы с входом.
- Корректно работают с любыми ``device``/``dtype`` входных данных; используется ``eps`` для защиты от деления на ноль.
- Параметры по умолчанию: ``dim=0``, ``eps=1e-12``; для MinMax/QuantileMinMax — ``feature_range=(0.0, 1.0)``,
  для RobustScaler — ``quantile_range=(25.0, 75.0)``, для QuantileMinMax — обычно ``(1.0, 99.0)``.

Пример работы dim
-----------------
Пусть размер тензора (batch, channels, freq_points), где:
    - batch        — индекс отдельного примера (например, touchstone-файла),
    - channels     — разные каналы (S11real, S21real, ...),
    - freq_points  — точки частотной сетки.

dim=0:
    Аггрегируем по батчу. Статистики считаются отдельно для каждого канала и каждой частоты.
    (32, 4, 256) → (4, 256)

dim=(0, 2):
    Аггрегируем по батчу и по частотам. Статистики остаются только по каналам.
    (32, 4, 256) → (4, 1) или (4,)

"""

from __future__ import annotations

import torch
from typing import Sequence, Tuple

# --------------------------------------------------------------------------- #
#                                helpers                                      #
# --------------------------------------------------------------------------- #
def _norm_dim(arg: int | Sequence[int] | None) -> Tuple[int, ...]:
    """
    Приводит аргумент `dim` к кортежу целых:
        None      -> ()
        int       -> (int,)
        list/tuple -> tuple(arg)

    Raises
    ------
    TypeError
        Если `arg` не int | Sequence[int] | None.
    """
    if arg is None:
        return ()
    if isinstance(arg, int):
        return (arg,)
    if isinstance(arg, (list, tuple)):
        return tuple(int(d) for d in arg)
    raise TypeError("dim must be int | Sequence[int] | None")


def _update_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    """
    Безопасно обновляет buffer. Если форма / устройство поменялись —
    перерегистрируем буфер, иначе делаем in-place copy_.
    """
    if not hasattr(module, name) or getattr(module, name).shape != value.shape:
        module.register_buffer(name, value)
    else:
        getattr(module, name).copy_(value)

# --------------------------------------------------------------------------- #
#                                base class                                   #
# --------------------------------------------------------------------------- #
class _Base(torch.nn.Module):
    """
    Базовый класс скейлеров.
    Позволяет агрегировать статистику по произвольным измерениям.
    """
    DEFAULT_EPS: float = 1e-12  # защита от деления на 0 (можно переопределить в init)

    def __init__(self, dim: Sequence[int] | int | None = 0, eps: float | None = None):
        super().__init__()
        self.default_dim: Tuple[int, ...] = _norm_dim(dim)
        self.eps = float(eps) if eps is not None else self.DEFAULT_EPS

    # ── internal helpers ────────────────────────────────────────────────────
    @staticmethod
    def _reduce(t: torch.Tensor, fn, dim: Tuple[int, ...]) -> torch.Tensor:
        """Применяет fn последовательно по измерениям из dim (в порядке убывания)."""
        if not dim:
            return t
        for d in sorted(dim, reverse=True):
            res = fn(t, dim=d, keepdim=True)
            t = res[0] if isinstance(res, tuple) else res
        return t

    def _cast_like(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Приводит dtype / device к ref."""
        return t.to(dtype=ref.dtype, device=ref.device)

    # ── public API (дочерние классы должны определить: fit, forward, inverse) ─
    def fit(self, data: torch.Tensor, dim: Sequence[int] | int | None = None): ...  # noqa: D401,E701  (abstract stub)

# --------------------------------------------------------------------------- #
#                                StdScaler                                    #
# --------------------------------------------------------------------------- #
class StdScaler(_Base):
    """
    Стандартизация:

        forward(x) = (x - mean) / std
        inverse(z) = z * std + mean
    """
    def __init__(self, dim: Sequence[int] | int | None = 0, eps: float | None = None, *, unbiased: bool = False):
        super().__init__(dim, eps)
        self.unbiased = bool(unbiased)

        # сохраняем init-параметры для корректной сериализации
        self._init_kwargs = {
            "dim": dim,
            "eps": eps,
            "unbiased": unbiased
        }

        # «пустые» буферы-заглушки; реальные значения появятся после fit()
        self.register_buffer("mean", torch.tensor(0.))
        self.register_buffer("std", torch.tensor(1.))

    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit(self, data: torch.Tensor, dim: Sequence[int] | int | None = None):
        dims = _norm_dim(dim) if dim is not None else self.default_dim

        mean = self._reduce(data, torch.mean, dims)
        std = self._reduce(data, lambda x, dim, keepdim: torch.std(x, dim=dim, keepdim=keepdim, unbiased=self.unbiased),
                           dims).clamp_min(self.eps)

        # приводим device / dtype к data
        mean, std = self._cast_like(mean, data), self._cast_like(std, data)

        _update_buffer(self, "mean", mean.detach())
        _update_buffer(self, "std", std.detach())
        return self

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def extra_repr(self) -> str:  # для красивого print(module)
        return f"dim={self.default_dim}, unbiased={self.unbiased}, eps={self.eps}"

# --------------------------------------------------------------------------- #
#                               MinMaxScaler                                  #
# --------------------------------------------------------------------------- #
class MinMaxScaler(_Base):
    """
    Линейное масштабирование в диапазон [a, b].

    Прямое и обратное преобразования:
        forward(x) = (x - data_min) / data_range * (b - a) + a
        inverse(y) = (y - a) / (b - a) * data_range + data_min

    Параметры
    ----------
    dim : int | Sequence[int] | None
        Измерения, по которым считается min/max. Если None — не агрегируем.
    feature_range : Tuple[float, float]
        Целевой диапазон. По умолчанию [0.0, 1.0].
    eps : float, optional
        Малое значение, предотвращающее деление на ноль.
    """

    def __init__(
        self,
        dim: Sequence[int] | int | None = 0,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        eps: float | None = None,
    ):
        super().__init__(dim, eps)

        # ── Валидация диапазона ─────────────────────────────────────────────
        if not isinstance(feature_range, (tuple, list)) or len(feature_range) != 2:
            raise TypeError("feature_range must be a tuple/list of two floats")
        if not all(isinstance(v, (int, float)) for v in feature_range):
            raise TypeError("feature_range must contain numeric values")

        a, b = map(float, feature_range)
        if b <= a:
            raise ValueError("feature_range must satisfy max > min")

        # сохраняем init-параметры для корректной сериализации
        self._init_kwargs = {
            "dim": dim,
            "feature_range": feature_range,
            "eps": eps,
        }

        self.min_val = a
        self.max_val = b

        # ── Буферы, заполняемые после fit() ─────────────────────────────────
        self.register_buffer("data_min", torch.tensor(0.))
        self.register_buffer("data_range", torch.tensor(1.))

    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit(self, data: torch.Tensor, dim: Sequence[int] | int | None = None):
        dims = _norm_dim(dim) if dim is not None else self.default_dim

        d_min = self._reduce(data, torch.min, dims)
        d_max = self._reduce(data, torch.max, dims)
        d_range = (d_max - d_min).clamp_min(self.eps)

        d_min, d_range = self._cast_like(d_min, data), self._cast_like(d_range, data)

        _update_buffer(self, "data_min", d_min.detach())
        _update_buffer(self, "data_range", d_range.detach())
        return self

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_std = (x - self.data_min) / self.data_range
        return x_std * (self.max_val - self.min_val) + self.min_val

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        scale = max(self.max_val - self.min_val, self.eps)
        y_std = (y - self.min_val) / scale
        return y_std * self.data_range + self.data_min

    def extra_repr(self) -> str:
        return f"dim={self.default_dim}, range=({self.min_val}, {self.max_val}), eps={self.eps}"


# --------------------------------------------------------------------------- #
#                               RobustScaler                                  #
# --------------------------------------------------------------------------- #
class RobustScaler(_Base):
    r"""
    Масштабирование, устойчивое к выбросам (аналог ``sklearn.RobustScaler``).

    Формулы
    -------
    .. math::

        \text{forward}(x) = \frac{x - \text{median}}{\text{IQR}},\qquad
        \text{IQR}=Q_\text{high}-Q_\text{low}

    Параметры
    ----------
    dim : int | Sequence[int] | None
        Измерения, по которым считаются квантиль и медиана. ``None`` — не агрегируем.
    quantile_range : Tuple[float, float], default=(25.0, 75.0)
        Нижний и верхний процентили **в процентах** (0–100].
    eps : float, optional
        Защита от деления на ноль.
    """

    def __init__(
        self,
        dim: Sequence[int] | int | None = 0,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        eps: float | None = None,
    ):
        super().__init__(dim, eps)

        # ── валидация диапазона квантилей ────────────────────────────────
        if (
            not isinstance(quantile_range, (tuple, list))
            or len(quantile_range) != 2
            or not all(isinstance(q, (int, float)) for q in quantile_range)
        ):
            raise TypeError("quantile_range must be tuple/list of two floats")
        q_low, q_high = map(float, quantile_range)
        if not (0.0 <= q_low < q_high <= 100.0):
            raise ValueError("quantile_range must satisfy 0 ≤ low < high ≤ 100")

        self.q_low = q_low / 100.0
        self.q_high = q_high / 100.0

        # сохраняем init-параметры для корректной сериализации
        self._init_kwargs = {
            "dim": dim,
            "quantile_range": quantile_range,
            "eps": eps,
        }

        # ── буферы, наполняемые после fit() ───────────────────────────────
        self.register_buffer("center", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(1.0))

    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit(self, data: torch.Tensor, dim: Sequence[int] | int | None = None):
        """
        Вычисляет медиану и интерквантильный размах по указанным измерениям.
        Возвращает ``self`` для чейнинга.
        """
        dims = _norm_dim(dim) if dim is not None else self.default_dim

        # --- медиана ------------------------------------------------------
        median = self._reduce(
            data,
            lambda x, dim, keepdim: torch.median(x, dim=dim, keepdim=keepdim).values,
            dims,
        )

        # --- квантиль low / high ------------------------------------------
        q_low = self._reduce(
            data,
            lambda x, dim, keepdim: torch.quantile(
                x, self.q_low, dim=dim, keepdim=keepdim
            ),
            dims,
        )
        q_high = self._reduce(
            data,
            lambda x, dim, keepdim: torch.quantile(
                x, self.q_high, dim=dim, keepdim=keepdim
            ),
            dims,
        )

        iqr = (q_high - q_low).clamp_min(self.eps)

        # приводим device / dtype к data
        median, iqr = self._cast_like(median, data), self._cast_like(iqr, data)

        _update_buffer(self, "center", median.detach())
        _update_buffer(self, "scale", iqr.detach())
        return self

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Преобразует данные к устойчивому масштабу."""
        return (x - self.center) / self.scale

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Обратное преобразование из робаст-масштаба в оригинальный."""
        return z * self.scale + self.center

    def extra_repr(self) -> str:
        low = self.q_low * 100.0
        high = self.q_high * 100.0
        return f"dim={self.default_dim}, q_range=({low}%, {high}%), eps={self.eps}"


# --------------------------------------------------------------------------- #
#                               QuantileMinMaxScaler                                  #
# --------------------------------------------------------------------------- #
class QuantileMinMaxScaler(_Base):
    """
    Робастное масштабирование в [a, b] по квантилям.

    Идея: low- и high-квантили исходных данных считаются опорными точками,
    их линейно переводим в a и b (по умолчанию 0 и 1). Хвосты можно
    опционально клиппировать, чтобы *гарантировать* диапазон [a, b].

        forward(x) = clip(x, q_low, q_high) - если clip=True
                     затем (x - q_low) / (q_high - q_low) * (b - a) + a
        inverse(y) = (y - a) / (b - a) * (q_high - q_low) + q_low

    Параметры
    ----------
    dim : int | Sequence[int] | None
        Оси для агрегирования квантилей. None — без агрегации.
    quantile_range : (float, float), default=(1.0, 99.0)
        Нижний/верхний процентили в процентах (0–100], 0 ≤ low < high ≤ 100.
    feature_range : (float, float), default=(0.0, 1.0)
        Целевой диапазон (a, b), где b > a.
    clip : bool, default=True
        Если True, значения ниже/выше опорных квантилей клиппируются,
        так что forward(x) ∈ [a, b] строго.
    eps : float | None
        Защита от деления на ноль.
    """

    def __init__(
        self,
        dim: Sequence[int] | int | None = 0,
        quantile_range: Tuple[float, float] = (1.0, 99.0),
        feature_range: Tuple[float, float] = (0.0, 1.0),
        clip: bool = True,
        eps: float | None = None,
    ):
        super().__init__(dim, eps)

        # валидация квантилей
        if (
            not isinstance(quantile_range, (tuple, list))
            or len(quantile_range) != 2
            or not all(isinstance(q, (int, float)) for q in quantile_range)
        ):
            raise TypeError("quantile_range must be tuple/list of two floats")
        q_low, q_high = map(float, quantile_range)
        if not (0.0 <= q_low < q_high <= 100.0):
            raise ValueError("quantile_range must satisfy 0 ≤ low < high ≤ 100")

        self.q_low = q_low / 100.0
        self.q_high = q_high / 100.0

        # валидация целевого диапазона
        if (
            not isinstance(feature_range, (tuple, list))
            or len(feature_range) != 2
            or not all(isinstance(v, (int, float)) for v in feature_range)
        ):
            raise TypeError("feature_range must be a tuple/list of two floats")
        a, b = map(float, feature_range)
        if b <= a:
            raise ValueError("feature_range must satisfy max > min")
        self.min_val = a
        self.max_val = b

        self.clip = bool(clip)

        # для сериализации
        self._init_kwargs = {
            "dim": dim,
            "quantile_range": quantile_range,
            "feature_range": feature_range,
            "clip": clip,
            "eps": eps,
        }

        # буферы (заполняются в fit)
        self.register_buffer("data_min", torch.tensor(0.0))   # q_low
        self.register_buffer("data_range", torch.tensor(1.0)) # q_high - q_low

    @torch.no_grad()
    def fit(self, data: torch.Tensor, dim: Sequence[int] | int | None = None):
        dims = _norm_dim(dim) if dim is not None else self.default_dim

        q_lo = self._reduce(
            data,
            lambda x, dim, keepdim: torch.quantile(x, self.q_low, dim=dim, keepdim=keepdim),
            dims,
        )
        q_hi = self._reduce(
            data,
            lambda x, dim, keepdim: torch.quantile(x, self.q_high, dim=dim, keepdim=keepdim),
            dims,
        )

        d_min = q_lo
        d_range = (q_hi - q_lo).clamp_min(self.eps)

        d_min, d_range = self._cast_like(d_min, data), self._cast_like(d_range, data)
        _update_buffer(self, "data_min", d_min.detach())
        _update_buffer(self, "data_range", d_range.detach())
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_adj = x
        if self.clip:
            data_max = self.data_min + self.data_range
            x_adj = torch.clamp(x_adj, min=self.data_min, max=data_max)
        x_std = (x_adj - self.data_min) / self.data_range
        return x_std * (self.max_val - self.min_val) + self.min_val

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        scale = max(self.max_val - self.min_val, self.eps)
        y_std = (y - self.min_val) / scale
        return y_std * self.data_range + self.data_min

    def extra_repr(self) -> str:
        low = self.q_low * 100.0
        high = self.q_high * 100.0
        return f"dim={self.default_dim}, q_range=({low}%, {high}%), range=({self.min_val}, {self.max_val}), clip={self.clip}, eps={self.eps}"
