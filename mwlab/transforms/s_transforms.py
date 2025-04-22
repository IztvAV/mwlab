# transforms/s_transforms.py
import torch
import numpy as np
import skrf

class S_Crop:
    """
    Обрезает сеть по диапазону частот [f_start, f_stop].

    Параметры
    ----------
    f_start : float
        Начальная частота в единицах *unit*.
    f_stop  : float
        Конечная частота в единицах *unit*.
    unit : str, optional
        Единица измерения ('Hz', 'kHz', 'MHz', 'GHz').
        Если None — используется `network.frequency.unit`.
    """
    def __init__(self, f_start: float, f_stop: float, unit: str | None = None):
        if f_start >= f_stop:
            raise ValueError("f_start должен быть меньше f_stop")
        self.f_start = f_start
        self.f_stop = f_stop
        self.unit = unit  # может быть None

    def __call__(self, network: skrf.Network) -> skrf.Network:
        unit = self.unit or network.frequency.unit  # <‑‑ ключевая строка
        return network.cropped(f_start=self.f_start,
                               f_stop=self.f_stop,
                               unit=unit)


class S_Resample:
    """Интерполирует сеть по новому количеству точек или вектору частот.

    Параметры:
        freq_or_n (int или array-like или Frequency):
            целевое число точек или вектор частот.
            Если передан array-like, значения интерпретируются в тех же единицах,
            что и network.frequency.unit.
        **kwargs: дополнительные аргументы для scipy.interpolate.interp1d().
    """
    def __init__(self, freq_or_n, **kwargs):
        self.freq_or_n = freq_or_n
        self.kwargs = kwargs

    def __call__(self, network: skrf.network.Network) -> skrf.network.Network:
        # Создаем копию, чтобы не менять исходный объект
        ntwk = network.copy()
        # Определяем аргумент для resample
        arg = self.freq_or_n
        # Если array-like, и не Frequency и не int, конвертируем в Гц
        if isinstance(arg, (list, tuple, np.ndarray)):
            arr = np.asarray(arg)
            # единица исходной сети
            unit = network.frequency.unit or 'Hz'
            mult = skrf.frequency.Frequency.multiplier_dict[unit.lower()]
            # переводим все в Гц
            arr_hz = arr * mult
            arg = arr_hz
        elif isinstance(arg, skrf.Frequency):
            arg = arg
        elif isinstance(arg, int):
            pass
        else:
            raise TypeError(f"Unsupported type for freq_or_n: {type(arg)}")
        # Вызываем inplace-интерполяцию
        ntwk.resample(arg, **self.kwargs)
        return ntwk
