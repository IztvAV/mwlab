# MWLAB Objectives Serde Contract (draft)

> Контракт сериализации/десериализации (serde) для подсистемы целей/ограничений `mwlab.opt.objectives`  
> Форматы: **YAML** и **JSON**  
> Версия: **__**

## 1. Назначение и контекст

Этот документ описывает **канонический формат** хранения и загрузки технического задания (ТЗ) и спецификаций для оптимизационных задач в `mwlab`, представленных как:

- `Specification` — набор критериев, объединённых логическим **AND**;
- `Criterion` — композиция компонентов:

```
Selector -> Transform -> Aggregator -> Comparator
```

Где:
- **Selector** извлекает частотную кривую `(freq, vals)` из `skrf.Network` и задаёт *исходные* единицы (`freq_unit`, `value_unit`);
- **Transform** (опционально) предобрабатывает кривую `(freq, vals)` (band, сглаживание, производные, GD и т.п.);
- **Aggregator** сворачивает кривую в скаляр (float);
- **Comparator** интерпретирует скаляр как pass/fail и возвращает штраф `penalty >= 0`.

**Serde-контракт** гарантирует:
- воспроизводимость цепочек вычислений,
- прозрачность инженерных соглашений (ничего «неявного» в Selector),
- устойчивость к ошибкам смешения единиц,
- возможность контроля совместимости и валидности *до* вычислений.

---

## 2. Термины и нормативные слова

В документе используются слова:

- **ДОЛЖНО** / **MUST** — обязательное требование
- **СЛЕДУЕТ** / **SHOULD** — настоятельная рекомендация
- **МОЖЕТ** / **MAY** — допустимый вариант

---

## 3. Общие требования к данным

### 3.1. Разрешённые типы в `params`

Все параметры компонентов (`params`) ДОЛЖНЫ быть сериализуемыми в YAML/JSON типами:

- `null`, `bool`, `int`, `float`, `str`,
- `list`,
- `dict` (строки в качестве ключей).

Дополнительно поддерживаются **комплексные числа** в JSON-совместимом каноническом виде:

```yaml
{ "__complex__": [<re>, <im>] }
```

где `<re>` и `<im>` — конечные (`finite`) числа (не NaN и не Inf).

**Запрещены**: callables (функции/лямбды), классы, объекты NumPy (на dump приводятся к python-типам), ссылки на модули, неявные выражения.

### 3.2. Запрет NaN/Inf в параметрах

Пользовательские параметры (`defaults`, `criteria[*]...params`, `weight`, `band`, `limit`, и т.п.)
**НЕ ДОЛЖНЫ** содержать `NaN` или `Inf` (включая YAML-литералы вида `.nan`, `.inf`, `nan`, `inf`).

Loader ДОЛЖЕН отклонять такие документы с понятной ошибкой и путём до поля.


### 3.3. Registry и `type`

Каждый компонент в файле идентифицируется по строковому `type`, который:
- ДОЛЖЕН существовать в соответствующем registry (`register_selector/transform/aggregator/comparator`);
- ДОЛЖЕН быть **каноническим алиасом** (см. §9) и трактуется **case-sensitive**.

---

## 4. Заголовок документа

Каждый YAML/JSON документ ДОЛЖЕН содержать поля:

```yaml
format: mwlab.spec
version: 1
name: "spec"        # опционально; если отсутствует — используется "spec"
```

Правила:
- `format` фиксирован и равен `mwlab.spec`;
- `version` фиксирован и равен `1` для этого контракта;
- `name` — человекочитаемое имя спецификации (для логов/отчётов).

---

## 5. Структура Specification

### 5.1. Поля Specification

Минимальная структура:

```yaml
format: mwlab.spec
version: 1
name: "my_spec"
criteria:
  - ...
```

Дополнительно допускается поле `defaults` (см. §5.2).

`criteria` ДОЛЖНО быть **непустым** списком.

### 5.2. `defaults` (опционально)

`defaults` — дефолтные параметры, подмешиваемые в `params` компонентов при загрузке, если параметр не указан явно.

Рекомендуемая структура:

```yaml
defaults:
  criterion:   { weight: 1.0 }
  selector:    { validate: true, freq_unit: "GHz" }
  transform:   { validate: true }
  aggregator:  { validate: true }
  comparator:  { }
```

Правила merge:
- merge выполняется **только по отсутствующим ключам** (явные параметры в критерии всегда сильнее дефолтов);
- merge выполняется отдельно по группам: `criterion/selector/transform/aggregator/comparator`.

---

## 6. Структура Criterion

Каждый элемент `criteria` ДОЛЖЕН иметь вид:

```yaml
- name: "S11 return loss"
  weight: 1.0                 # опционально, default=1.0
  selector:   <ComponentSpec> # обязательно
  transform:  <TransformSpec> # опционально (identity, если нет)
  aggregator: <ComponentSpec> # обязательно
  comparator: <ComponentSpec> # обязательно
  meta: { ... }               # опционально (любые сериализуемые данные)
```

Требования:
- `name` ДОЛЖНО быть уникальным в пределах `criteria` (иначе конфликт ключей в `report`);
- `weight` ДОЛЖЕН быть конечным числом, СЛЕДУЕТ требовать `weight > 0`;
- `selector/aggregator/comparator` обязательны;
- отсутствие `transform` трактуется как **identity** (ничего не делать).

---

## 7. Универсальная спецификация компонента (ComponentSpec)

### 7.1. Каноническая форма

Любой компонент (Selector/Aggregator/Comparator и одиночный Transform) задаётся как:

```yaml
type: "<registry_type>"
params:
  <ключ>: <значение>
  ...
```

Пример (Selector):

```yaml
selector:
  type: SMagSelector
  params:
    m: 1
    n: 1
    db: true
    freq_unit: "GHz"
    band: [1.0, 2.0]
    validate: true
```

### 7.2. Строгая проверка параметров

Загрузчик (loader) СЛЕДУЕТ делать строгим:
- неизвестные параметры в `params` → **ошибка** (UnknownParam),
- отсутствующие обязательные параметры конструктора → **ошибка** (MissingParam),
- неверные типы/диапазоны → **ошибка** (InvalidValue).

---

## 8. Спецификация Transform (TransformSpec)

Поле `transform` в Criterion МОЖЕТ быть задано тремя способами.

### 8.1. Одиночный Transform (ComponentSpec)

```yaml
transform:
  type: FiniteTransform
  params: { mode: drop }
```

### 8.2. Канонический Compose (рекомендуемый стандарт)

```yaml
transform:
  type: Compose
  params:
    validate: true
    transforms:
      - { type: FiniteTransform, params: { mode: drop } }
      - { type: SmoothPointsTransform, params: { n_pts: 9 } }
      - { type: DerivativeTransform, params: { method: diff, basis: Hz } }
```

Требования:
- `transforms` ДОЛЖНО быть списком `ComponentSpec`;
- `compose` СЛЕДУЕТ использовать как **каноническое** представление цепочек.

### 8.3. Синтаксический сахар: список transforms

Допускается:

```yaml
transform:
  - { type: FiniteTransform, params: { mode: drop } }
  - { type: SmoothPointsTransform, params: { n_pts: 9 } }
```

Правило:
- loader преобразует такой список в канонический `Compose`.

---

## 9. Канонические `type` и алиасы registry

### 9.1. Канонический алиас

Один и тот же класс может регистрироваться под несколькими алиасами:

- пример: `register_selector(("SMagSelector", "s_mag", "sdb", "smag")))`

Для сериализации/файлового формата выбирается **ровно один канонический алиас**.

**Правило (канон):**
- канонический алиас = **первый** алиас в регистрации, либо строка, если передана одна.

Это обеспечивает:
- стабильный формат файлов,
- возможность поддерживать «старые имена» как синонимы в loader.

### 9.2. Поведение при неизвестном `type`

Неизвестный `type` ДОЛЖЕН приводить к ошибке, содержащей:
- путь в документе (например, `criteria[2].selector.type`),
- категорию (`selector/transform/...`),
- список известных типов или ближайшие совпадения.

---

## 10. Единицы и соглашения (critical)

### 10.1. Единицы частоты (`freq_unit`)

Канонические значения: `"Hz"`, `"kHz"`, `"MHz"`, `"GHz"` (без чувствительности к регистру при загрузке).

Loader ДОЛЖЕН нормализовать через `normalize_freq_unit()`.

### 10.2. Семантические единицы значений (`value_unit`)

`value_unit` задаётся селектором и является **семантической меткой** (не обязательно строгой SI-единицей).

Примеры: `"db"`, `"lin"`, `"rad"`, `"deg"`, `"complex"`, `"ns"`, `"s"`, `""`.

### 10.3. Band у Selector

Если у селектора есть `band`, то:
- `band` интерпретируется **в единицах `selector.freq_unit`**;
- применяется как **маска по существующим точкам сетки** (без интерполяции граничных точек).

Если нужна строгая полоса с включением границ — используйте `BandTransform(include_edges=true)`.

### 10.4. Параметры Transform с единицами

Для Transform, где параметры имеют частотный смысл, используются поля `*_unit`:

- `BandTransform`: `band_unit`
- `SmoothApertureTransform` и `ApertureSlopeTransform`: `fw_unit`
- `ValueAtAgg`: `f0_unit`

Правило:
- если `*_unit` не задан, параметр трактуется в текущих единицах частоты;
- если задан, loader/выполнение unit-aware должны корректно привести величину к текущему `freq_unit`.

---

## 11. Unit-aware исполнение и dry-валидация цепочки

### 11.1. Режим вычисления

В вычислительном проходе критерия СЛЕДУЕТ использовать unit-aware API:

- `Transform.apply(freq, vals, freq_unit=..., value_unit=...)`
- `Aggregator.aggregate(freq, vals, freq_unit=..., value_unit=...)`

Это важно для корректной физики (например, производные в Hz, GD, апертуры).

### 11.2. Dry-валидация при загрузке (без `rf.Network`)

Loader СЛЕДУЕТ выполнять проверку совместимости цепочки **до вычислений**, используя:

- `selector.freq_unit` и `selector.value_unit` как стартовые единицы,
- `transform.out_value_unit(...)` для протаскивания `value_unit` по цепочке,
- `aggregator.out_value_unit(...)` при необходимости,
- атрибуты `expects_value_unit` / `expects_freq_unit` у компонентов, если они заданы.

Если `expects_value_unit` — список/кортеж, то допустим любой из перечисленных.

### 11.3. Ошибки конфигурации unit-aware

Примеры ошибок, которые СЛЕДУЕТ ловить заранее:
- `GroupDelayTransform` требует `value_unit` ∈ {"rad","deg"}.
- `UpIntAgg/LoIntAgg/RippleIntAgg` с `basis="Hz"` требует `aggregate(..., freq_unit=...)`, а не `__call__`.
- `Compose.apply` требует, чтобы все transforms имели `apply`.

---

## 12. Политики NaN/Inf и пустых данных

Serde-контракт должен сохранять параметры политик как часть `params`, например:
- Transform: `FiniteTransform(mode=...)`, `on_empty=...`
- Aggregator: `finite_policy`, `on_empty`, `complex_mode`
- Comparator: `finite_policy`, `non_finite_penalty`

**Важно:** семантика этих политик определяется реализацией классов; serde лишь фиксирует, как это хранится.

---

## 13. Сериализация (dump) — требования

### 13.1. Общие правила dump
При dump Specification в dict (и далее в YAML/JSON):

- всегда писать `format/version`,
- всегда писать канонический `type`,
- все значения приводить к python-примитивам (например, `np.float64 -> float`),
- не записывать несериализуемые объекты.

### 13.2. Минимизация vs полнота
Разрешены два режима dump:

1) **Полный (explicit)** — писать все параметры.
2) **Компактный (diff-from-default)** — писать только параметры, отличающиеся от дефолтов конструктора.

Контракт допускает оба режима. Для устойчивости CI/диффов СЛЕДУЕТ выбрать один режим как стандарт проекта.

---

## 14. Диагностика ошибок (рекомендуемый стандарт)

Ошибка загрузки/валидации СЛЕДУЕТ включать:

- `path` (пример: `criteria[0].transform.params.transforms[2].params.basis`)
- `kind` (например: `UnknownType`, `UnknownParam`, `MissingParam`, `InvalidValue`, `UnitMismatch`)
- человекочитаемое сообщение
- (опционально) `hints`: список подсказок (например, близкие `type` по Levenshtein)

---

## 15. Пример полного YAML

Ниже пример спецификации из двух критериев:

```yaml
format: mwlab.spec
version: 1
name: "demo_spec"

defaults:
  criterion:  { weight: 1.0 }
  selector:   { validate: true, freq_unit: "GHz" }
  transform:  { validate: true }
  aggregator: { validate: true }
  comparator: { }

criteria:
  - name: "S11 (dB) <= -10 in 1..2 GHz"
    weight: 1.0
    selector:
      type: SMagSelector
      params: { m: 1, n: 1, db: true, band: [1.0, 2.0] }
    transform:
      type: SignTransform
      params: { sign: -1 }
    aggregator:
      type: MaxAgg
      params: { finite_policy: omit, on_empty: raise, complex_mode: raise }
    comparator:
      type: LEComparator
      params: { limit: 10.0, unit: "db", finite_policy: fail, non_finite_penalty: 1.0 }

  - name: "Group delay ripple (ns) in 1..2 GHz"
    selector:
      type: PhaseSelector
      params: { m: 2, n: 1, unwrap: true, unit: rad, band: [1.0, 2.0] }
    transform:
      type: Compose
      params:
        transforms:
          - { type: GroupDelayTransform, params: { out_unit: ns } }
          - { type: FiniteTransform, params: { mode: drop } }
    aggregator:
      type: RippleAgg
      params: { finite_policy: omit, on_empty: raise }
    comparator:
      type: LEComparator
      params: { limit: 1.0, unit: "ns" }
```

---

## 16. Примечания по проектированию loader/dumper (не часть формата)

Это не требования контракта, но практические рекомендации:

1) **Factory по registry**: `make_component(kind, type, params)`  
2) **Интроспекция конструктора** для строгой проверки параметров (`inspect.signature`)  
3) **Нормализация единиц** на входе (`normalize_freq_unit`, lower/strip для value_unit)  
4) **Миграции версий**: `version -> миграция -> v1-структура -> build`  
5) **Канонизация при dump**: всегда приводить к compose и каноническим `type`.

---