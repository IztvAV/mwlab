# Подсистема `mwlab.opt.sensitivity`

Анализ чувствительности — это первый шаг к пониманию того, **какие параметры** действительно управляют поведением вашего СВЧ-устройства и куда стоит направить усилия по оптимизации и контролю допуска.  
Подпакет `mwlab.opt.sensitivity` закрывает три классических метода:

| Метод | Назначение | Когда применять |
|-------|-----------|-----------------|
| **Morris (elementary effects)** | Быстрый скрининг 100 + параметров; выделить «ключевые» | самый первый запуск |
| **Sobol (Saltelli sampling)** | Точные доли вклада \(S_1, S_T\) + (опц.) пары \(S_{ij}\) | 10 – 30 параметров после скрининга |
| **Active Subspace** | Линейные комбинации параметров (главные направления) | когда d ≫ 2 и хотите упростить задачу |

Под капотом используются:

* собственный высокоуровневый фасад `SensitivityAnalyzer`;
* библиотека **SALib** (опциональная зависимость) — для Морриса и Соболя;
* чистый NumPy / автоградиенты — для Active Subspace;
* Seaborn-стиль для графиков.

---

## Установка

```bash
pip install mwlab                  # базовая библиотека
pip install mwlab[analysis]        # добавляет SALib ≥ 1.4
```

> Если `SALib` не установлен, будут работать **только** методы Active Subspace;  
>   Моррис/Соболь аккуратно сообщат о нехватке зависимости.

---

## Быстрый старт

```python
from mwlab.opt.design.space import DesignSpace, ContinuousVar
from mwlab.opt.surrogates import NNSurrogate        # готовая NN‑модель
from mwlab.opt.objectives.specification import Specification
from mwlab.opt.objectives.base import BaseCriterion
from mwlab.opt.objectives import SMagSelector, MaxAgg, LEComparator

# 1. Пространство параметров
space = DesignSpace({
    "gap": ContinuousVar(-100e-6, 100e-6),
    "w"  : ContinuousVar(900e-6, 1100e-6),
})

# 2. Загружаем surrogate
sur = NNSurrogate.load("models/vco_surrogate.pt")

# 3. Техническое задание
crit_s11 = BaseCriterion(
    selector   = SMagSelector(1, 1, band=(2.0, 2.3), db=True),
    aggregator = MaxAgg(),
    comparator = LEComparator(-22, unit="dB"),
    name="S11"
)
spec = Specification([crit_s11])

# 4. Анализатор
from mwlab.opt.sensitivity import SensitivityAnalyzer
sa = SensitivityAnalyzer(surrogate=sur, design_space=space, specification=spec)

# 5. Morris
df_morris = sa.morris(N=20)

# 6. Sobol
top_vars = df_morris["mu_star"].nlargest(10).index
df_sobol = sa.sobol(params=top_vars, n_base=2048)

# 7. Active Subspace
lam, W = sa.active_subspace(k=3, n_samples=5000)
print(lam)
```

---

## API‑справка

| Метод | Аргументы | Возвращает |
|-------|-----------|------------|
| `morris(N=20, sampler="sobol", plot=True)` | `N` — траекторий | `DataFrame` `['mu_star','sigma']` |
| `sobol(params="auto", n_base=1024, second_order=False)` | `params` — список или "auto" | `DataFrame` `['S1','ST']` (+ `attrs['S2_pairs']`) |
| `active_subspace(k=5, n_samples=5000, method="fd")` | | `(λ, W)` |

---

## Дальнейшая работа

* Поддержка градиентов surrogate для Active Subspace.  
* Реализация Fast GSA / FGSA.  
* Учет коррелированных входов (OpenTURNS).

---

## Обратная связь

Создавайте issue или PR в GitHub‑репозитории MWLab. Чем подробнее описание, тем быстрее интегрируем!
