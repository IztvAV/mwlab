# 📊 `mwlab.utils.analysis`

Модуль `TouchstoneDatasetAnalyzer` предназначен для **анализa датасета на основе Touchstone‑файлов**.

Позволяет изучать статистику как по параметрам (`params`), так и по S‑матрицам (`skrf.Network`). Использует `pandas`, `xarray`, `seaborn` для анализа и визуализации.

---

## 🚀 Быстрый старт

```python
from mwlab.datasets import TouchstoneDataset
from mwlab.utils.analysis import TouchstoneDatasetAnalyzer

ds = TouchstoneDataset("Data/Filter12")
an = TouchstoneDatasetAnalyzer(ds)
```

---

## 📋 Анализ параметров (`params`)

### 📑 Таблица параметров
```python
df = an.get_params_df()
```

### 📊 Сводная статистика
```python
an.summarize_params()
```

### 🧮 Только изменяемые параметры
```python
an.get_varying_keys()
```

### 🖼️ Гистограммы параметров
```python
an.plot_param_distributions()
```

---

## 🔬 Анализ S‑матриц

### 📉 Сводная статистика по компонентам S-матрицы
```python
an.summarize_s_components()
```

---

## 📦 Экспорт данных

- `.export_params_csv("params.csv")` — сохраняет таблицу параметров в CSV
- `.export_s_netcdf("s_stats.nc")` — сохраняет mean/std/min/max по S‑матрице

---

## 📉 Построение графиков

### Среднее и ±1σ по частоте:

```python
an.plot_s_stats(port_out=1, port_in=1, metric="db")
```

Можно выбрать метрику: `"db"`, `"mag"`, `"deg"`.

