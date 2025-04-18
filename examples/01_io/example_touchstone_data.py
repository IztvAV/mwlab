"""
Демонстрация ключевых возможностей TouchstoneData.

Требования:
    pip install scikit-rf numpy

Перед запуском убедитесь, что файл Sprms.s2p лежит рядом
с этим скриптом.
"""
from pathlib import Path
import numpy as np
import skrf as rf
from mwlab.io.touchstone import TouchstoneData

EXAMPLE = Path(__file__).with_name("Sprms.s2p")


# --------------------------------------------------------------------------- #
# 1)  Чтение готового Touchstone‑файла                                         #
# --------------------------------------------------------------------------- #
ts = TouchstoneData.load(EXAMPLE)
print("------ исходный файл ------")
print("Путь:         ", ts.path)
print("Параметры:    ", ts.params)
print("Частотные точки:", len(ts.network.f), "\n")

# --------------------------------------------------------------------------- #
# 2)  Обновляем параметры и сохраняем под новым именем                         #
# --------------------------------------------------------------------------- #
print("Обновляем w -> w+0.5  и пишем в  'Sprms_mod.s2p'")
ts.params["w"] = ts.params.get("w", 0.0) + 0.5
ts.save("Sprms_mod.s2p")          # новая строка Parameters будет записана

# --------------------------------------------------------------------------- #
# 3)  Создаём TouchstoneData из результатов симуляции (CST → Python)          #
# --------------------------------------------------------------------------- #
print("\n------ создаём сеть «с нуля» ------")
freq = rf.Frequency(2, 4, 201, unit="GHz")
# пример: простая идентичность S11=‑20 дБ, S12=‑3 дБ
s = np.zeros((201, 2, 2), dtype=np.complex64)
s[:, 0, 0] = 10 ** (-20 / 20) * np.exp(1j * 0)   # |S11|=‑20 дБ
s[:, 1, 0] = 10 ** (-3 / 20)  * np.exp(1j * 0)   # |S21|=‑3 дБ
net = rf.Network(frequency=freq, s=s, z0=50)

ts_new = TouchstoneData(
    network = net,
    params  = {"filter_order": 3, "fc": 3.0}
)
ts_new.save("my_synth.s2p")
print("Создан файл my_synth.s2p  с параметрами:", ts_new.params)

# --------------------------------------------------------------------------- #
# 4)  Быстрый просмотр/обработка через scikit‑rf                              #
# --------------------------------------------------------------------------- #
net_mod = rf.Network("Sprms_mod.s2p")
print("\nПотери прохода (|S21| в дБ) в    Sprms_mod.s2p:")
print(net_mod.s21.s_db)
