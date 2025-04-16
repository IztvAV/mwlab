import numpy as np
from pathlib import Path
from mwlab.fileio import load_touchstone, save_touchstone
import matplotlib.pyplot as plt

class TouchstoneFile:
    """
    Класс, представляющий Touchstone-файл (sNp) с S-параметрами.

    Позволяет загружать, сохранять, преобразовывать и визуализировать S-данные.
    """

    def __init__(self, freqs, s_data, format='RI', unit='HZ', impedance=50, annotations=None):
        self.freq = np.asarray(freqs)
        self.s = np.asarray(s_data)
        self.format = format.upper()
        self.unit = unit.upper()
        self.impedance = impedance
        self.annotations = annotations or []

        if self.s.ndim != 3 or self.s.shape[1] != self.s.shape[2]:
            raise ValueError("Ожидается массив S-параметров размерности (N, n_ports, n_ports)")

    @classmethod
    def from_data(cls, freqs, s_data, format='RI', unit='HZ', impedance=50, annotations=None):
        """
        Создаёт объект TouchstoneFile из массивов данных (без файла).

        Параметры
        ----------
        freqs : np.ndarray
            Массив частот (в Гц)
        s_data : np.ndarray
            Массив S-параметров (shape = [N, n_ports, n_ports])
        format : str
            Формат данных: 'RI', 'MA', 'DB'
        unit : str
            Единицы измерения частоты: 'HZ', 'MHZ', 'GHZ'
        impedance : float
            Характеристическое сопротивление
        annotations : list[str]
            Комментарии, которые попадут в файл при сохранении

        Возвращает
        ----------
        TouchstoneFile
        """
        return cls(
            freqs=np.asarray(freqs),
            s_data=np.asarray(s_data),
            format=format,
            unit=unit,
            impedance=impedance,
            annotations=annotations or []
        )

    @classmethod
    def load(cls, filepath):
        """
        Загружает Touchstone-файл и создаёт объект класса.
        """
        freqs, s_data, meta, annotations = load_touchstone(filepath, annotations=True)
        return cls(freqs, s_data,
                   format=meta.get('format', 'RI'),
                   unit=meta.get('unit', 'HZ'),
                   impedance=meta.get('impedance', 50),
                   annotations=annotations)

    def save(self, filepath, format=None, unit=None, float_fmt="%.9f"):
        """
        Сохраняет текущий объект в Touchstone-файл.

        Можно указать формат и единицу измерения, отличающиеся от исходных.
        """
        save_touchstone(
            filepath,
            self.freq,
            self.s,
            format=format or self.format,
            unit=unit or self.unit,
            annotations=self.annotations,
            float_fmt=float_fmt,
            impedance=self.impedance
        )

    @property
    def n_ports(self):
        return self.s.shape[1]

    @property
    def n_freqs(self):
        return self.freq.shape[0]

    def to_format(self, target_format: str) -> 'TouchstoneFile':
        """
        Возвращает копию объекта, в котором данные представлены в другом формате ('RI', 'MA', 'DB').
        """
        target_format = target_format.upper()
        if target_format == self.format:
            return self.copy()

        if self.format != 'RI':
            raise ValueError("Конвертация возможна только из формата 'RI'. Перезагрузите файл в RI-формате.")

        if target_format == 'MA':
            s_new = np.abs(self.s) * np.exp(1j * np.angle(self.s))
        elif target_format == 'DB':
            magnitude = 20 * np.log10(np.abs(self.s))
            phase = np.angle(self.s, deg=True)
            s_new = magnitude * np.exp(1j * np.deg2rad(phase))
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")

        return TouchstoneFile(
            self.freq.copy(),
            s_new,
            format=target_format,
            unit=self.unit,
            impedance=self.impedance,
            annotations=self.annotations.copy()
        )

    def copy(self):
        """
        Возвращает глубокую копию объекта.
        """
        return TouchstoneFile(
            self.freq.copy(),
            self.s.copy(),
            format=self.format,
            unit=self.unit,
            impedance=self.impedance,
            annotations=self.annotations.copy()
        )

    def plot_s_db(self, element: str = "S21"):
        """
        Визуализация модуля выбранного элемента S-матрицы в децибелах.

        Пример: S21, S11, S12
        """

        if len(element) != 3 or not element[0].upper() == 'S':
            raise ValueError("Неверный формат элемента, ожидается строка вида 'S21'.")

        i, j = int(element[1]) - 1, int(element[2]) - 1
        if i >= self.n_ports or j >= self.n_ports:
            raise IndexError("Выход за пределы матрицы S.")

        s_elem = self.s[:, i, j]
        magnitude_db = 20 * np.log10(np.abs(s_elem))

        freq_ghz = self.freq / 1e9 if self.unit == 'HZ' else self.freq
        plt.plot(freq_ghz, magnitude_db)
        plt.xlabel("Частота (GHz)")
        plt.ylabel(f"|{element}| [dB]")
        plt.title(f"{element} vs Frequency")
        plt.grid(True)
        plt.show()
