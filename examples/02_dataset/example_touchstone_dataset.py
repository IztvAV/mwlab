"""
Демонстрация работы с TouchstoneDataset и TouchstoneDatasetAnalyzer

Требования:
    Перед началом работы убедитесь, что установлены следующие пакеты:
    - **MWLab** (включая transforms и анализатор)
    - **scikit-rf**
    - **seaborn**, **matplotlib**, **pandas**, **xarray**

Папка с данными:
    Для демонстрации используется директория `Data/Filter12/`, содержащая *.s2p файлы.
"""

import numpy as np
from pathlib import Path
from mwlab.datasets import TouchstoneDataset
from mwlab.utils.analysis import TouchstoneDatasetAnalyzer
from mwlab.transforms import TComposite
from mwlab.transforms.x_transforms import X_SelectKeys
from mwlab.transforms.s_transforms import S_Crop, S_Resample
import skrf as rf
import matplotlib.pyplot as plt

"""
    Воспользуемся готовым набором Touchstone-файлов из библиотеки **MWLab**,  
    содержащим результаты моделирования S-параметров СВЧ-фильтра 12-го порядка,  
    полученные для различных значений параметров электромагнитной модели.
"""

DATA_DIR = Path(__file__).parents[2] / "Data" / "Filter12"

# --------------------------------------------------------------------------- #
# 1. Доступ к набору данных и его анализ                                      #
# --------------------------------------------------------------------------- #
"""
    ### Загрузка набора Touchstone-файлов и инициализация анализатора
    В библиотеке **MWLab** предусмотрен специальный класс `TouchstoneDataset`,  
    предназначенный для загрузки набора Touchstone-файлов из указанной директории.
    
    Каждый файл содержит результаты расчета S-параметров и значения параметров электромагнитной модели.  
    Класс `TouchstoneDataset` предоставляет удобный интерфейс для доступа к этим данным и дальнейшего анализа.
"""
# Загружаем набор Touchstone-файлов
ds = TouchstoneDataset(source=DATA_DIR)

print(f"Загружено файлов: {len(ds)}")
print("Пример параметров из первого файла:")
print(ds[0][0], "\n")

"""
    Другой полезный инструмент библиотеки **MWLab** — класс `TouchstoneDatasetAnalyzer`,  
    предназначенный для удобного анализа и визуализации набора Touchstone-файлов.
    
    Он позволяет группировать данные, строить графики, выполнять сравнение и другие операции,  
    что особенно полезно при работе с большими объёмами моделирования.
"""
# Создаем объект анализатора на основе загруженного набора
analyzer = TouchstoneDatasetAnalyzer(ds)

"""
    ### Извлечение таблицы параметров из набора
    С помощью метода `get_params_df()` можно получить сводную таблицу параметров (в виде `pandas.DataFrame`),  
    содержащую значения всех параметров электромагнитной модели из каждого Touchstone-файла в наборе.
    
    Такую таблицу удобно использовать для:
      - предварительного анализа структуры данных,
      - фильтрации и группировки по параметрам,
      - построения графиков зависимости характеристик от параметров модели,
      - подготовки данных к обучению моделей машинного обучения.

    Ниже выводятся первые строки таблицы параметров.
"""
# Получаем таблицу параметров из всех файлов набора
df = analyzer.get_params_df()

print("Несколько строк из таблицы параметров:")
print(df.head())

"""
    ### Сводная статистика по параметрам модели
    Метод `summarize_params()` возвращает таблицу с базовой статистикой по всем числовым параметрам в датасете:  
    средние значения (`mean`), стандартное отклонение (`std`), минимумы (`min`), максимумы (`max`), количество 
    незаданных значений (`nan`), а также признак того, изменяется ли параметр в данном наборе данных (`is_constant`).
    
    Это позволяет быстро оценить структуру параметров и выявить отсутствующие или неизменяющиеся значения.
"""

# Получаем сводную статистику по параметрам (средние, мин, макс, std и т.д.)
summary = analyzer.summarize_params()

print("\nСводная статистика по параметрам:")
print(summary.T)  # Транспонируем для удобства отображения

"""
    ### Определение изменяемых параметров
    С помощью метода `get_varying_keys()` можно получить список параметров,  
    значения которых **меняются** от одного Touchstone-файла к другому.
    
    Это полезно для того, чтобы:
      - выделить только варьируемые параметры (например, для построения графиков зависимости),
      - исключить постоянные параметры из анализа,
      - упростить визуализацию многомерных данных.
"""
# Получаем список параметров, которые изменяются в наборе данных
varying = analyzer.get_varying_keys()

print("\nИзменяемые параметры:", varying)

"""
    ### Визуализация распределения параметров
    
    С помощью метода `plot_param_distributions()` можно построить графики распределения значений параметров.  
    Это позволяет оценить диапазоны и форму распределения каждого параметра —  
    например, равномерное, нормальное распределение или наличие выбросов.
    
    На графике ниже отображены гистограммы для первых трех изменяемых параметров, выявленных ранее.
"""

# Построим распределения первых трех изменяемых параметров
fig = analyzer.plot_param_distributions(varying[0:3])
fig.suptitle("Распределения параметров", y=1.02)
plt.show()

"""
    ### S-параметры: сводная статистика
    Аналогично методу `summarize_params()`, метод `summarize_s_components()` возвращает сводную таблицу,  
    содержащую базовую статистику для всех компонентов S-параметров, усредненную по всем объектам в датасете  
    и по частотным точкам. В таблице приводятся следующие показатели:
      - средние значения (`mean`),
      - стандартное отклонение (`std`),
      - минимумы (`min`),
      - максимумы (`max`),
      - количество незаданных значений (`nan`),
      - признак того, изменяется ли S-компонента в данном наборе данных (`is_constant`).
"""
# Получаем сводную статистику по всем компонентам S-параметров
# (усреднение по объектам и по частоте: среднее, std, min, max, nan и т.д.)
summary_s = analyzer.summarize_s_components()

print("\nСводная статистика по S-параметрам:")
print(summary_s.T)  # Транспонируем для более удобного отображения

"""
    ### Визуализация статистик по S-параметрам
    Метод `plot_s_stats()` позволяет построить графики статистик S-параметров по всем файлам набора.  
    Можно выбрать порт входа и выхода, интересующую метрику (`'db'`, `'mag'`, `'deg'`)  
    и список отображаемых статистик, таких как:
      - среднее (`mean`),  
      - стандартное отклонение (`std`),  
      - минимум (`min`),  
      - максимум (`max`).
"""
# Пример визуализации статистик S‑параметра S21 (модуль в децибелах)
fig = analyzer.plot_s_stats(
    port_out=2, port_in=1, metric='db', stats=['mean', 'std', 'min', 'max']
)
fig.suptitle("Статистика по S21 (дБ)", y=1.02)
plt.show()

# Пример визуализации статистик S‑параметра S11 (модуль в абсолютных значениях)
fig = analyzer.plot_s_stats(
    port_out=1, port_in=1, metric='mag', stats=['mean', 'std']
)
fig.suptitle("Статистика по S11 (модуль)", y=1.02)
plt.show()

# Пример визуализации статистик S‑параметра S11 (фаза в градусах)
fig = analyzer.plot_s_stats(
    port_out=1, port_in=1, metric='deg', stats=['mean', 'std']
)
fig.suptitle("Статистика по S11 (фаза)", y=1.02)
plt.show()


# --------------------------------------------------------------------------- #
# 2. Трансформация параметров: использование `X_SelectKeys`                   #
# --------------------------------------------------------------------------- #

"""
    ### Отбор параметров с помощью `X_SelectKeys`
    Параметры (`X`) электромагнитной модели из набора данных можно предварительно отобрать (трансформировать) "на лету" 
    перед дальнейшей обработкой. Для этого в **MWLab** предусмотрен объект `X_SelectKeys`, который позволяет оставить 
    только нужные параметры — например, только те, что изменяются в датасете.
    
    В этом случае при создании объекта `X_SelectKeys` в конструктор передаётся список ключей (названий параметров), 
    которые нужно оставить в данных.
"""
# Найдем список изменяемых параметров
varying = analyzer.get_varying_keys()

# Используем его как аргумент для X-преобразования (отбор только изменяемых параметров)
x_tf = X_SelectKeys(varying)

# Создаем новый датасет с примененной X-трансформацией
ds_x = TouchstoneDataset(source=DATA_DIR, x_tf=x_tf)

# Инициализируем анализатор на основе преобразованного датасета
analyzer_x = TouchstoneDatasetAnalyzer(ds_x)

"""
    ### Анализ параметров после X-преобразования
    
    После применения `X_SelectKeys` в наборе остаются только отобранные параметры.  
    Методы `summarize_params()` и `plot_param_distributions()` можно использовать повторно,  
    чтобы пересчитать статистику и построить гистограммы уже по фильтрованному множеству параметров.
    
    Это удобно для проверки корректности выбора параметров и для фокусного анализа только тех величин,  
    которые действительно варьируются в модели.
"""
# Получаем статистику по параметрам после применения X_SelectKeys
summary_x = analyzer_x.summarize_params()

print("\nСтатистика по параметрам после преобразования X_SelectKeys:")
print(summary_x.T)

# Визуализируем распределения оставшихся (отфильтрованных) параметров
fig = analyzer_x.plot_param_distributions()
fig.suptitle("Распределения отфильтрованных параметров модели", y=1.02)
plt.show()

# --------------------------------------------------------------------------- #
# 3. Трансформация S-параметров: использование `S_Crop`                       #
# --------------------------------------------------------------------------- #

"""
    ### Ограничение частотного диапазона с помощью `S_Crop`
    S-параметры Touchstone-файлов могут охватывать широкий частотный диапазон,  
    но в практическом анализе часто важно сфокусироваться только на рабочей полосе.  
    Класс `S_Crop` позволяет задать нужные границы частот и автоматически ограничить (обрезать) данные.
    
    Следующий пример демонстрирует, как ограничить частотный диапазон центральными 50% от исходного.
"""
# Определим частотный диапазон по первому элементу датасета
net0 = ds[0][1]
f_start, f_stop = net0.f[0], net0.f[-1]

# Задаем новый диапазон: оставим центральные 50%
f1 = f_start + 0.25 * (f_stop - f_start)
f2 = f_stop  - 0.25 * (f_stop - f_start)

# Получаем единицы измерения и масштаб для отображения
unit = net0.frequency.unit or "Hz"
mult = rf.Frequency.multiplier_dict[unit.lower()]

print(f"\nОграничим диапазон: с {f1 / mult:.3f} до {f2 / mult:.3f} {unit}")

# Создаем S-преобразователь с заданными границами
s_tf_crop = S_Crop(f_start=f1, f_stop=f2)

# Пересоздаем датасет и анализатор с новым S-преобразованием
ds_crop = TouchstoneDataset(source=DATA_DIR, s_tf=s_tf_crop)
analyzer_crop = TouchstoneDatasetAnalyzer(ds_crop)

"""
    ### Визуализация результата применения `S_Crop`
    После применения `S_Crop` частотный диапазон данных сужается до заданных границ.  
    На графике ниже показано, как изменилась статистика коэффициента передачи **S21**  
    в результате обрезки частотной области.
"""
# Визуализируем, как изменился S21 (коэффициент передачи) после обрезки частотного диапазона
fig = analyzer_crop.plot_s_stats(port_out=2, port_in=1)
fig.suptitle("S21 после применения S_Crop (обрезка диапазона)", y=1.02)
plt.show()


# --------------------------------------------------------------------------- #
# 4. Трансформация S-параметров: использование `S_Resample`                   #
# --------------------------------------------------------------------------- #

"""
    ### Вариант 1. Пересэмплирование по числу точек
    Если требуется просто изменить количество точек в частотной сетке,  
    можно передать желаемое число в конструктор `S_Resample`.  
    В этом случае сетка будет автоматически построена равномерно в пределах исходного диапазона.
"""
# Вариант 1: пересэмплируем до 128 равномерно распределенных точек
s_tf_resample1 = S_Resample(128)

ds_resample1 = TouchstoneDataset(source=DATA_DIR, s_tf=s_tf_resample1)
analyzer_resample1 = TouchstoneDatasetAnalyzer(ds_resample1)

fig = analyzer_resample1.plot_s_stats(port_out=2, port_in=1)
fig.suptitle("S21 после S_Resample (128 точек)", y=1.02)
plt.show()

"""
    ### Вариант 2. Явное задание частотной сетки
    Вместо указания количества точек можно вручную задать новую частотную сетку  
    с помощью объекта `Frequency` из библиотеки `scikit-rf`.  
    Это дает полный контроль над диапазоном и шагом дискретизации.
"""
# Вариант 2: задаем новую частотную сетку вручную (12.7–12.76 ГГц, 64 точки)
f = rf.Frequency(12.7, 12.76, 64, unit="GHz")
s_tf_resample2 = S_Resample(f)

ds_resample2 = TouchstoneDataset(source=DATA_DIR, s_tf=s_tf_resample2)
analyzer_resample2 = TouchstoneDatasetAnalyzer(ds_resample2)

fig = analyzer_resample2.plot_s_stats(port_out=2, port_in=1)
fig.suptitle("S21 после S_Resample (частоты 12.7–12.76 ГГц, 64 точки)", y=1.02)
plt.show()

# --------------------------------------------------------------------------- #
# 5. Комбинирование трансформаций: использование `TComposite`                 #
# --------------------------------------------------------------------------- #
"""
    Класс `TComposite` позволяет объединить несколько преобразований в единую цепочку,  
    которая будет последовательно применяться к данным.
    
    Это особенно удобно, когда нужно выполнить несколько этапов обработки подряд —  
    например, сначала обрезать частотный диапазон, а затем пересэмплировать сетку,  
    не создавая вручную вложенные вызовы преобразователей.
    
    В следующем примере мы последовательно применим два преобразования:
    1. **Обрезка частотного диапазона** с помощью `S_Crop`,
    2. **Пересэмплирование** на фиксированное число точек с помощью `S_Resample`.
"""
# Задаем составное преобразование:
# сначала S_Crop, затем S_Resample до 128 точек
s_tf_composite = TComposite([
    S_Crop(f_start=f1, f_stop=f2),
    S_Resample(128)
])

# Создаём новый датасет и анализатор с объединённым S-преобразованием
ds_composite = TouchstoneDataset(source=DATA_DIR, s_tf=s_tf_composite)
analyzer_composite = TouchstoneDatasetAnalyzer(ds_composite)

# Визуализируем результат (например, S21)
fig = analyzer_composite.plot_s_stats(port_out=2, port_in=1)
fig.suptitle("S21 после TComposite: S_Crop + S_Resample", y=1.02)
plt.show()