# transforms/s_transforms.py
import torch
import numpy as np
import skrf

class S_Crop:
    """Обрезает сеть по диапазону частот [f_start, f_stop].

    Параметры:
        f_start (float): начальная частота в единицах unit.
        f_stop (float): конечная частота в единицах unit.
        unit (str, optional): единица измерения частоты ('GHz','MHz' и т.д.).
                              Если None, используется unit сети.
    """
    def __init__(self, f_start: float, f_stop: float, unit: str = None):
        self.f_start = f_start
        self.f_stop = f_stop
        self.unit = unit

    def __call__(self, network: skrf.network.Network) -> skrf.network.Network:
        # Используем метод cropped для создания новой сети без изменения исходной
        return network.cropped(f_start=self.f_start, f_stop=self.f_stop, unit=self.unit)


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


class S_ToTensor:
    """Преобразует s-матрицу сети в тензор PyTorch с каналами real/imag.

    На вход: Network
    На выход: тензор формы (2, F, P, P)
    """
    def __call__(self, network: skrf.network.Network) -> torch.Tensor:
        # Получаем комплексную s-матрицу и разделяем на real/imag
        s = network.s.astype(np.complex64)
        real = s.real.astype(np.float32)
        imag = s.imag.astype(np.float32)
        # Собираем тензор формы (2, F, P, P): [канал real; канал imag]
        return torch.from_numpy(np.stack([real, imag], axis=0))

