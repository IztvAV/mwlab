# mwlab/opt/objectives/network_like.py
"""
mwlab.opt.objectives.network_like
================================

Duck-typed протоколы "сетей" и частотной оси для подсистемы objectives
и лёгкие view-контейнеры для частоты и S-параметров с ленивым доступом к NumPy.

Зачем это нужно
---------------
Ранее подсистема была жёстко привязана к `skrf.Network`. Для оптимизационных задач
(особенно глобальная оптимизация/penalty/yield) это часто приводит к деградации
производительности из-за многократного создания "тяжёлых" объектов Network.

Этот модуль вводит:
- `NetworkLike` и `FrequencyLike` протоколы (минимальный контракт),
- `FrequencyView` и `SParamView` лёгкие контейнеры, которые:
  * принимают NumPy или Torch,
  * лениво приводят данные к NumPy,
  * могут кэшировать вычисленные массивы (`abs`, `log10`),
  * совместимы по интерфейсу с `skrf.Network` настолько, чтобы существующие селекторы
    (использующие `net.frequency.f`, `net.s`, `net.s_mag`, `net.s_db`) работали без правок.

Ключевые моменты реализации
---------------------------
1) `SParamView` реализует атрибут `frequency` с полем `.f`, а также `s/s_mag/s_db`.
   Это делает его "совместимым" с вашим `NetworkLike` протоколом и старым кодом селекторов.
2) Конвертация Torch -> NumPy:
   - обязательно делаем `detach()`, если `requires_grad=True` (иначе `.numpy()` упадёт),
   - переносим на CPU при необходимости,
   - приводим к contiguous при необходимости,
   - используем zero-copy только для CPU+contiguous.
3) `s_db` по умолчанию вычисляется как `20*log10(|s|)`. Это может давать `-inf`
   при нулевых значениях. Для оптимизаций это иногда нежелательно, поэтому добавлен
   опциональный параметр `db_floor` (в dB) или `mag_eps` (минимальная амплитуда).
   По умолчанию поведение "физически прямое" — без клипов.

Дополнительно
-------------
Добавлен лёгкий фабричный хелпер make_netlike(...), чтобы в других подсистемах
(например, surrogates) был единый “источник истины” по созданию NetworkLike.
"""

from __future__ import annotations

import importlib.util
from typing import Optional, Protocol, Union

import numpy as np

# -----------------------------------------------------------------------------
# Опциональная поддержка torch (без жёсткой зависимости).
# -----------------------------------------------------------------------------
_TORCH_SPEC = importlib.util.find_spec("torch")
if _TORCH_SPEC is not None:
    import torch  # type: ignore
else:
    torch = None  # type: ignore


# Тип входных массивов: либо numpy, либо torch.Tensor (если torch установлен).
ArrayLike = Union[np.ndarray, "torch.Tensor"]


# =============================================================================
# Протоколы (минимальные интерфейсы) для совместимости без skrf
# =============================================================================

class FrequencyLike(Protocol):
    """Минимальный интерфейс частотной оси, совместимый с rf.Frequency.

    Требование: атрибут `.f` — 1-D массив частот в Гц (или в тех единицах,
    в которых селектор ожидает частоты; у skrf это обычно Гц).
    """
    f: np.ndarray


class NetworkLike(Protocol):
    """Минимальный интерфейс сети, совместимый с rf.Network.

    Требования:
    - `frequency.f` : 1-D массив частот
    - `s`           : массив S-параметров (обычно комплексный)
    - `s_mag`       : |s|
    - `s_db`        : 20*log10(|s|)

    Примечание: форма `s` может быть (nf, nport, nport) — как в skrf.
    Селекторы обычно сами берут нужный (m,n) и затем приводят к 1-D.
    """
    frequency: FrequencyLike
    s: np.ndarray
    s_mag: np.ndarray
    s_db: np.ndarray


# =============================================================================
# Внутренние утилиты конвертации в NumPy
# =============================================================================

def _as_numpy(arr: ArrayLike, *, name: str) -> np.ndarray:
    """Преобразовать входной массив в NumPy.

    Поддерживаем:
    - np.ndarray -> возвращаем как есть (zero-copy)
    - torch.Tensor -> преобразуем к NumPy

    Политика zero-copy:
    - CPU + contiguous + (не требует grad) -> `.numpy()` без копии.
    - Иначе:
      * если requires_grad=True -> `detach()`
      * если CUDA -> `cpu()`
      * если non-contiguous -> `contiguous()`
      * затем `.numpy()` (это будет копия, если требовались преобразования)

    Важно:
    - `.numpy()` у torch не работает для CUDA-тензоров.
    - `.numpy()` также не работает для тензоров, участвующих в графе градиента
      (`requires_grad=True`) без detach.
    """
    if isinstance(arr, np.ndarray):
        return arr

    if torch is not None and isinstance(arr, torch.Tensor):
        t = arr

        # Если тензор "трекает" градиенты, надо отсоединить от графа.
        # Иначе torch выбросит ошибку при попытке `.numpy()`.
        if getattr(t, "requires_grad", False):
            t = t.detach()

        # Перенос на CPU, если тензор на GPU
        if t.device.type != "cpu":
            t = t.cpu()

        # Приведение к contiguous, иначе `.numpy()` может вернуть view с лишними
        # сложностями или просто скопировать/ошибиться в зависимости от версии.
        if not t.is_contiguous():
            t = t.contiguous()

        # После этих шагов `.numpy()` безопасен.
        return t.numpy()

    raise TypeError(f"{name} must be np.ndarray or torch.Tensor")


# =============================================================================
# Лёгкие view-контейнеры
# =============================================================================

class FrequencyView:
    """Ленивый view частот с опциональным кэшированием.

    Хранит исходный массив частоты (numpy или torch), и по запросу `f`
    выдаёт NumPy-массив.

    Параметры
    ---------
    f_src : ArrayLike
        Источник частот.
    cache : bool
        Если True — NumPy-результат кэшируется в `_f_np`.
        Если False — конвертация будет выполняться каждый доступ.
        Для оптимизаций обычно выгодно cache=True.
    """

    def __init__(self, f_src: ArrayLike, *, cache: bool = True) -> None:
        self.f_src = f_src
        self.cache = bool(cache)
        self._f_np: Optional[np.ndarray] = None

    @property
    def f(self) -> np.ndarray:
        """Частота как NumPy-массив."""
        if self.cache:
            if self._f_np is None:
                self._f_np = _as_numpy(self.f_src, name="f")
            return self._f_np
        return _as_numpy(self.f_src, name="f")


class SParamView:
    """Ленивый view для S-параметров и производных mag/db массивов.

    Ключевая цель: соответствовать `NetworkLike` (как `skrf.Network`), но быть
    лёгким и быстрым для множественных вычислений критериев.

    Что именно совместимо
    ---------------------
    - `self.frequency.f` (как в skrf: net.frequency.f)
    - `self.s`
    - `self.s_mag`
    - `self.s_db`

    Дополнительно:
    - `self.freq` — удобный алиас на `self.frequency.f` (часто удобно в отладке)

    Параметры
    ---------
    s : ArrayLike
        S-параметры (обычно complex).
    freq : ArrayLike
        Частота (обычно 1-D).
    cache : bool
        Кэшировать ли NumPy-представления и вычисленные массивы.
    mag_eps : float | None
        Если задано, то при вычислении s_db амплитуда клипуется снизу:
            mag = max(|s|, mag_eps)
        Это защищает от -inf в log10, что полезно в оптимизации.
        По умолчанию None — без клипа.
    db_floor : float | None
        Альтернатива mag_eps: задать минимальный уровень в dB.
        Тогда mag_eps будет вычислен как 10^(db_floor/20).
        Пример: db_floor=-300 -> mag_eps=10^(-15).
        Если заданы и mag_eps и db_floor, приоритет у mag_eps.
    """

    def __init__(
        self,
        s: ArrayLike,
        freq: ArrayLike,
        *,
        cache: bool = True,
        mag_eps: Optional[float] = None,
        db_floor: Optional[float] = None,
    ) -> None:
        self._s_src = s

        # ВАЖНОЕ ИЗМЕНЕНИЕ относительно прежней версии:
        # Теперь есть атрибут `frequency` с полем `.f`,
        # чтобы старые селекторы вида `net.frequency.f` работали без правок.
        self.frequency = FrequencyView(freq, cache=cache)

        self.cache = bool(cache)

        # Кэш NumPy S-параметров и производных
        self._s_np: Optional[np.ndarray] = None
        self._s_mag_np: Optional[np.ndarray] = None
        self._s_db_np: Optional[np.ndarray] = None

        # Настройки для защиты от -inf в dB (опционально)
        self._mag_eps = float(mag_eps) if mag_eps is not None else None
        if self._mag_eps is None and db_floor is not None:
            # Переводим уровень db_floor (dB) в минимальную амплитуду
            self._mag_eps = float(10.0 ** (float(db_floor) / 20.0))

    @property
    def freq(self) -> np.ndarray:
        """Удобный алиас: частоты как NumPy-массив."""
        return self.frequency.f

    @property
    def s(self) -> np.ndarray:
        """S-параметры как NumPy-массив."""
        if self.cache:
            if self._s_np is None:
                self._s_np = _as_numpy(self._s_src, name="s")
            return self._s_np
        return _as_numpy(self._s_src, name="s")

    @property
    def s_mag(self) -> np.ndarray:
        """|S| (амплитуда)."""
        if self.cache:
            if self._s_mag_np is None:
                self._s_mag_np = np.abs(self.s)
            return self._s_mag_np
        return np.abs(self.s)

    @property
    def s_db(self) -> np.ndarray:
        """20*log10(|S|) в dB.

        По умолчанию может быть -inf при |S|=0.
        Если задан `mag_eps`, то используем клип снизу, чтобы избежать -inf:
            mag = max(|S|, mag_eps)
        """
        if self.cache and self._s_db_np is not None:
            return self._s_db_np

        mag = self.s_mag
        if self._mag_eps is not None:
            # Клип снизу для численной устойчивости
            mag = np.maximum(mag, self._mag_eps)

        out = 20.0 * np.log10(mag)

        if self.cache:
            self._s_db_np = out
        return out


def make_netlike(
    s: ArrayLike,
    freq_hz: ArrayLike,
    *,
    cache: bool = True,
    mag_eps: Optional[float] = None,
    db_floor: Optional[float] = None,
    validate: bool = False,
) -> SParamView:
    """
    Единая фабрика NetworkLike-объекта (SParamView).

    По умолчанию validate=False, чтобы:
    - не форсировать конвертацию torch->numpy в “горячих” циклах,
    - не тратить время на проверки формы.

    Если validate=True:
    - проверяем, что freq 1-D,
    - s имеет форму (nf, nport, nport),
    - nf совпадает с длиной freq.
    Эти проверки могут триггерить ленивую конвертацию в NumPy.
    """
    net = SParamView(s, freq_hz, cache=cache, mag_eps=mag_eps, db_floor=db_floor)
    if validate:
        f = np.asarray(net.frequency.f)
        ss = np.asarray(net.s)
        if f.ndim != 1:
            raise ValueError("make_netlike: freq_hz должен быть 1-D")
        if ss.ndim != 3:
            raise ValueError("make_netlike: s должен быть 3-D массивом (nf, nport, nport)")
        if ss.shape[0] != f.shape[0]:
            raise ValueError(f"make_netlike: nf в s ({ss.shape[0]}) не совпадает с len(freq) ({f.shape[0]})")
        if ss.shape[1] != ss.shape[2]:
            raise ValueError("make_netlike: s должен иметь квадратные портовые измерения (nport, nport)")
    return net


__all__ = [
    "ArrayLike",
    "FrequencyLike",
    "NetworkLike",
    "FrequencyView",
    "SParamView",
    "make_netlike",
]
