# MWLAB Objectives Serde Contract (ver 1)

**Область:** `mwlab.opt.objectives`  
**Форматы:** YAML и JSON (одна и та же модель данных)  
**Идентификатор формата:** `mwlab.spec`  
**Версия контракта:** `1`

---

## 0. Нормативные слова

- **ДОЛЖНО** — обязательное требование.
- **СЛЕДУЕТ** — рекомендация высокой важности.
- **МОЖЕТ** — допустимый вариант.

---

## 1. Каноническая модель

Спецификация — это AND-набор критериев.

Каждый критерий — композиция:

```
Selector -> Transform -> Aggregator -> Comparator
```

**Serde-контракт** описывает только *конфигурацию* этих объектов и правил её проверки/канонизации.  
Он **не** описывает формат отчёта/результатов (это задача `CriterionResult` и `Specification.report()`).

---

## 2. Типы данных и запреты

### 2.1 Допустимые типы значений в документе

Любое значение в документе (включая `meta`, `defaults`, `params`) **ДОЛЖНО** быть представимо следующими JSON/YAML-типами:

- `null`, `bool`, `int`, `float`, `str`,
- `list` (элементы — тоже допустимые типы),
- `dict` с **строковыми ключами** (значения — тоже допустимые типы).

### 2.2 Запрет NaN/Inf в параметрах

Параметры пользователя **НЕ ДОЛЖНЫ** содержать `NaN`/`Inf` (включая YAML-литералы `.nan/.inf`, `nan/inf`).

**Loader ДОЛЖЕН** отклонять документ с ошибкой, содержащей путь до поля.

> Примечание: NaN/Inf *в данных кривых* допустимы и обрабатываются Transform/Aggregator/Comparator-политиками; запрет относится только к *параметрам конфигурации*.

### 2.3 Комплексные числа в параметрах

Если требуется сериализовать комплексное число (например, `fill_value`, `ref` и т.п.), используется **единственный канонический JSON-совместимый вид**:

```yaml
{ "__complex__": [<re>, <im>] }
```

- `<re>`, `<im>` **ДОЛЖНЫ** быть конечными float (не NaN/Inf).
- Ключ `"__complex__"` **зарезервирован**.

### 2.4 Запрещённые значения

В документе запрещены:

- callables (функции/лямбды),
- ссылки на классы/модули,
- NumPy-объекты (на dump приводятся к python-примитивам),
- любые не-JSON/YAML-совместимые расширения (например, YAML tags для python-объектов).

---

## 3. Заголовок документа (обязателен)

Каждый документ **ДОЛЖЕН** иметь:

```yaml
format: mwlab.spec
version: 1
name: "spec"        # опционально, default="spec"
criteria: [...]
```

- `format` **ДОЛЖЕН** быть ровно `"mwlab.spec"`.
- `version` **ДОЛЖЕН** быть `1`.

Дополнительно **МОЖЕТ** быть:

- `defaults` (см. §5),
- `meta` (произвольные сериализуемые данные на уровне спецификации).

---

## 4. Структура Specification

```yaml
format: mwlab.spec
version: 1
name: "my_spec"
defaults: {...}     # опционально
meta: {...}         # опционально
criteria:
  - <CriterionSpec>
  - ...
```

Требования:

- `criteria` **ДОЛЖНО** быть **непустым** списком.
- Все `criteria[*].name` **ДОЛЖНЫ** быть **уникальны** (иначе конфликт ключей в `report`).

---

## 5. Defaults (опционально) и правила merge

`defaults` — дефолты, подмешиваемые при загрузке **только по отсутствующим ключам**  
(явно заданные параметры всегда сильнее).

Рекомендуемая структура:

```yaml
defaults:
  criterion:   { weight: 1.0 }
  selector:    { validate: true, freq_unit: "GHz" }
  transform:   { validate: true }
  aggregator:  { validate: true }
  comparator:  { }
```

Правила:

- merge выполняется **по группам**: `criterion/selector/transform/aggregator/comparator`;
- merge выполняется **только “missing keys”** (никакого deep-merge внутрь вложенных структур, кроме обычного dict-merge по ключам на текущем уровне);
- `defaults` не меняют `type`, только параметры/поля.

---

## 6. CriterionSpec (обязательная структура элемента criteria)

```yaml
- name: "..."
  weight: 1.0                 # опционально, default=1.0
  selector:   <ComponentSpec> # обязательно
  transform:  <TransformSpec> # опционально (identity, если отсутствует/ null)
  aggregator: <ComponentSpec> # обязательно
  comparator: <ComponentSpec> # обязательно
  meta: { ... }               # опционально
```

Требования:

- `weight` **ДОЛЖЕН** быть конечным числом; **СЛЕДУЕТ** требовать `weight > 0` (ошибка конфигурации, если ≤ 0).
- Если `transform` отсутствует или равен `null` — трактуется как **identity**.

---

## 7. ComponentSpec (универсальная форма компонента)

Любой **Selector / Transform (одиночный) / Aggregator / Comparator** задаётся так:

```yaml
type: "<registry_type>"
params: { ... }   # опционально, default={}
```

### 7.1 `type` и registry

- `type` **ДОЛЖЕН** существовать в соответствующем registry (`register_selector`, `register_transform`, `register_aggregator`, `register_comparator`).
- `type` трактуется **case-sensitive**.
- При загрузке **ДОЛЖНЫ** приниматься **все алиасы** из registry (для обратной совместимости).
- При dump **ДОЛЖЕН** использоваться **канонический алиас** (см. §10).

### 7.2 Строгая проверка `params`

Loader **ДОЛЖЕН** быть строгим:

- неизвестный параметр → ошибка `UnknownParam`,
- отсутствие обязательного параметра конструктора → `MissingParam`,
- неверный тип/диапазон → `InvalidValue`.

Проверка выполняется как минимум:

- интроспекцией `__init__` сигнатуры,
- плюс валидацией самим конструктором класса (исключения поднимаются наружу и оборачиваются в `InvalidValue` с путём).

---

## 8. TransformSpec (3 формы)

### 8.1 Одиночный Transform

```yaml
transform:
  type: FiniteTransform
  params: { mode: "drop" }
```

### 8.2 Канонический Compose

```yaml
transform:
  type: Compose
  params:
    validate: true
    transforms:
      - { type: FiniteTransform, params: { mode: "drop" } }
      - { type: SmoothPointsTransform, params: { n_pts: 9 } }
```

Требования:

- `params.transforms` **ДОЛЖНО** быть списком `ComponentSpec` (каждый элемент — Transform).
- Пустой список `transforms: []` трактуется как identity (**СЛЕДУЕТ** предупреждать, но не падать).

### 8.3 Синтаксический сахар: список transforms

Допускается:

```yaml
transform:
  - { type: FiniteTransform, params: { mode: "drop" } }
  - { type: SmoothPointsTransform, params: { n_pts: 9 } }
```

Правило:

- Loader **ДОЛЖЕН** канонизировать это в `Compose(...)` при построении объектов.

---

## 9. Единицы и “прозрачность” семантики

### 9.1 `freq_unit`

Канонические значения: `"Hz"`, `"kHz"`, `"MHz"`, `"GHz"`.

- Loader **ДОЛЖЕН** нормализовать единицы через `normalize_freq_unit()` (регистр/пробелы игнорируются).
- В файле допускаются любые регистры, но при dump пишется каноника (`"GHz"` и т.п.).

### 9.2 `value_unit`

`value_unit` — **семантическая метка**, задаваемая селектором/трансформом  
(например `"db"`, `"lin"`, `"rad"`, `"deg"`, `"complex"`, `"ns"`, `"s"`, `""`).

Контракт фиксирует только то, что:

- единицы **протаскиваются и валидируются** по цепочке (см. §11),
- смыслы не должны “молчаливо” меняться селекторами.

### 9.3 Band в Selector

Если у селектора есть `band`, то:

- `band` задаётся **в единицах `selector.freq_unit`**,
- применяется как **маска по узлам сетки** (без интерполяции границ).

Если нужны границы с интерполяцией — это обязанность `BandTransform(include_edges=true)`.

### 9.4 Частотные параметры Transform/Aggregator

Если параметр физически является частотой/шириной окна, компонент **МОЖЕТ** иметь `*_unit`:

- `BandTransform.band_unit`
- `SmoothApertureTransform.fw_unit`
- `ApertureSlopeTransform.fw_unit`
- `ValueAtAgg.f0_unit`

Правило:

- если `*_unit` отсутствует → параметр интерпретируется в **текущих единицах входной freq**,
- если `*_unit` задан → параметр **приводится** к текущей `freq_unit` в unit-aware режиме (`apply/aggregate`).

---

## 10. Канонизация `type` и dump-формат

### 10.1 Канонический алиас

Если класс зарегистрирован под несколькими алиасами, то:

- **канонический алиас = первый алиас в `register_*((...))`**.

Dump **ДОЛЖЕН** писать только канонический алиас. Loader **ДОЛЖЕН** принимать все алиасы.

### 10.2 Канонический вид Transform

Dump **ДОЛЖЕН** нормализовать любые transform-формы в один вид:

- либо `transform` отсутствует/`null` (identity),
- либо `Compose` с `params.transforms=[...]`,
- либо одиночный Transform как `ComponentSpec` (допустимо, но **СЛЕДУЕТ** предпочитать `Compose` для унификации диффов).

**Рекомендуемое правило проекта:** всегда dump’ить `transform` как `Compose`, даже если там 1 элемент (кроме identity).

### 10.3 Режим dump параметров

Для стабильности и воспроизводимости:

- **Канонический режим dump = “explicit”**: писать все параметры (включая validate/политики), даже если они равны дефолтам конструктора.

Компактный режим (“diff-from-default”) **МОЖЕТ** существовать как опция, но **не является каноническим**.

---

## 11. Unit-aware исполнение и dry-валидация цепочки

### 11.1 Исполнение (runtime)

Вычислительный проход критерия **ДОЛЖЕН** использовать unit-aware API там, где он существует:

- для Transform: `apply(freq, vals, freq_unit=..., value_unit=...)`  
  если `apply` не реализован — допускается `__call__` **только если** Transform явно unit-invariant (политика проекта: лучше не допускать).
- для Aggregator: `aggregate(freq, vals, freq_unit=..., value_unit=...)`  
  если `aggregate` не реализован — допускается `__call__` как unit-invariant.

Особое требование: если в цепочке есть `Compose.apply`, то все transforms внутри должны поддерживать `apply`  
(иначе это конфигурационная ошибка).

### 11.2 Dry-валидация (на этапе загрузки, без rf.Network)

Loader **СЛЕДУЕТ** выполнять проверку совместимости цепочки до вычислений:

- старт: `selector.freq_unit` и `selector.value_unit`,
- протаскивание: `transform.out_value_unit(in_unit, freq_unit)` (если реализовано),
- затем проверка агрегатора/трансформа по `expects_value_unit` / `expects_freq_unit`, если атрибут существует.

Если ожидания не выполнены → ошибка `UnitMismatch` с путём и ожидаемыми/фактическими единицами.

---

## 12. Ошибки и диагностика (обязательный минимум)

Любая ошибка загрузки/валидации **ДОЛЖНА** включать:

- `path` (пример: `criteria[0].transform.params.transforms[2].params.basis`)
- `kind` ∈ {`UnknownType`, `UnknownParam`, `MissingParam`, `InvalidValue`, `UnitMismatch`, `DuplicateName`}
- человекочитаемое сообщение

Дополнительно **СЛЕДУЕТ** добавлять `hints` (например, ближайшие совпадения `type`).

---

## 13. Минимальный пример YAML (канонический)

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
    selector:
      type: SMagSelector
      params: { m: 1, n: 1, db: true, band: [1.0, 2.0] }
    transform:
      type: Compose
      params:
        transforms:
          - { type: SignTransform, params: { sign: -1 } }
    aggregator:
      type: MaxAgg
      params: { finite_policy: "omit", on_empty: "raise", complex_mode: "raise", validate: true }
    comparator:
      type: LEComparator
      params: { limit: 10.0, unit: "db", finite_policy: "fail", non_finite_penalty: 1.0 }

  - name: "GD ripple (ns) in 1..2 GHz"
    selector:
      type: PhaseSelector
      params: { m: 2, n: 1, unwrap: true, unit: "rad", band: [1.0, 2.0] }
    transform:
      type: Compose
      params:
        transforms:
          - { type: GroupDelayTransform, params: { out_unit: "ns" } }
          - { type: FiniteTransform, params: { mode: "drop" } }
    aggregator:
      type: RippleAgg
      params: { finite_policy: "omit", on_empty: "raise", validate: true }
    comparator:
      type: LEComparator
      params: { limit: 1.0, unit: "ns" }
```

---

## 14. Рекомендуемые практики проекта (не часть протокола)

- Всегда dump’ить `transform` как `Compose`, кроме identity.
- Использовать канонический dump-режим `explicit` (все параметры), чтобы диффы были стабильны.
- Поддерживать строгий loader (unknown params/types → ошибка), чтобы конфиги не “разъезжались” годами.
