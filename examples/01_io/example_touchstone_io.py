"""
Пример демонстрации класса TouchstoneFile:
- Загрузка S-параметров из файла Sprms.s2p
- Просмотр метаинформации
- Преобразование формата данных
- Визуализация
- Сохранение в новом формате
"""

from mwlab.touchstone import TouchstoneFile
import matplotlib.pyplot as plt

# Загрузка Touchstone-файла
ts = TouchstoneFile.load("Sprms.s2p")

# Вывод основной информации
print(f"Частотных точек: {ts.n_freqs}")
print(f"Число портов: {ts.n_ports}")
print(f"Формат данных: {ts.format}")
print(f"Единицы частоты: {ts.unit}")
print(f"Импеданс: {ts.impedance} Ом")

# Визуализация S11, S21
ts.plot_s_db("S11")
ts.plot_s_db("S21")
ts.plot_s_db("S22")

#x = ts.freq
#y = ts.s[:,1,1]

#plt.plot(x, abs(y))
#plt.grid(True)
#plt.show()

# Преобразование в формат dB+angle
ts_db = ts.to_format("DB")

# Сохранение преобразованного файла
ts_db.save("Sprms_db_format.s2p", format="DB", unit="MHZ")
print("Файл сохранён как Sprms_db_format.s2p в формате DB (MHz)")
