# MWLAB Objectives Serde Contract (v1)
Контракт сериализации/десериализации (serde) для подсистемы целей/ограничений `mwlab.opt.objectives`  
Форматы: **YAML** и **JSON**  
Версия формата: **1**

> Этот документ задаёт **канонический формат** хранения и загрузки технического задания (ТЗ) / спецификаций,
> используемых в оптимизационных задачах mwlab. Формат предназначен для **воспроизводимости** вычислений,
> **строгой валидации** конфигурации и **устойчивости к ошибкам единиц измерения**.

---

## 1. Назначение и контекст

Подсистема целей/ограничений в `mwlab.opt.objectives` описывает требования к СВЧ-устройству через:

- **Specification** — набор критериев, объединённых логическим **AND**;
- **Criterion** — композиция компонентов:

```
Selector -> Transform -> Aggregator -> Comparator
```

Где:
- **Selector** извлекает частотную кривую `(freq, vals)` из `skrf.Network` и задаёт исходные единицы (`freq_unit`, `value_unit`).
- **Transform** (опционально) предварительно обрабатывает кривую `(freq, vals)` (band, сглаживание, производные, GD и т.п.).
- **Aggregator** сворачивает кривую в скаляр (float).
- **Comparator** интерпретирует скаляр как pass/fail и возвращает штраф `penalty >= 0`.

Контракт serde гарантирует:
- **воспроизводимость** цепочек вычислений,
- **прозрачность инженерных соглашений** (ничего «неявного»),
- **строгую проверку параметров** при загрузке,
- **раннюю unit-aware валидацию** цепочки *до* вычислений (без `rf.Network`).

---

## 2. Термины и нормативные слова

В документе используются слова:

- **ДОЛЖНО / MUST** — обязательное требование
- **СЛЕДУЕТ / SHOULD** — настоятельная рекомендация
- **МОЖЕТ / MAY** — допустимый вариант

---

## 3. Общие требования к данным

### 3.1. Разрешённые типы в `params`

Все параметры компонентов (`params`) **ДОЛЖНЫ** быть сериализуемыми в YAML/JSON типами:

- `null`, `bool`, `int`, `float`, `str`,
- `list`,
- `dict` (ключи — строки).

**Запрещены**:
- `callable` (функции/лямбды), классы, объекты Python/NumPy, не являющиеся примитивами,
- неявные выражения, ссылки на модули, `!!python/*` YAML-теги (непереносимые типы).

### 3.2. Запрет `NaN`/`Inf` в параметрах

Пользовательские параметры (`defaults`, `criteria[*]...params`, `weight`, `band`, `limit`, и т.п.)
**НЕ ДОЛЖНЫ** содержать `NaN` или `Inf`.

Загрузчик (loader) **ДОЛЖЕН** отклонять такие документы с понятной ошибкой и путём до поля (`path`).

> Примечание про JSON: некоторые парсеры допускают `NaN/Infinity` (не-RFC). Для этого формата JSON loader **MUST**
> запрещать `NaN/Infinity` на этапе парсинга и/или в пост-валидации.

### 3.3. Complex-числа (канонический JSON/YAML вид)

Комплексные числа **МОЖНО** кодировать только в JSON-совместимом каноническом виде:

```yaml
{ "__complex__": [<re>, <im>] }
```

где `<re>` и `<im>` — конечные (`finite`) числа (не NaN, не Inf).

**Запрещено**:
- YAML-тип `!!python/complex`,
- произвольные формы `"a+bj"` как строка (недетерминированно, сложно валидировать).

---

## 4. Заголовок документа

Каждый YAML/JSON документ спецификации **ДОЛЖЕН** содержать поля:

```yaml
format: mwlab.spec
version: 1
name: "spec"        # опционально; если отсутствует — используется "spec"
```

Правила:
- `format` фиксирован и равен `mwlab.spec`;
- `version` фиксирован и равен `1` для этого контракта;
- `name` — человекочитаемое имя спецификации (для логов/отчётов).

### 4.1. Расширения формата (forward-compat)

- Поля верхнего уровня, начинающиеся с `x-`, **МОЖНО** игнорировать loader’ом (расширения, метаданные).
- Прочие неизвестные поля верхнего уровня **СЛЕДУЕТ** считать ошибкой в строгом режиме (по умолчанию).

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

`criteria` **ДОЛЖНО** быть **непустым** списком.

Дополнительно допускаются поля:
- `defaults` — дефолтные параметры (см. ниже),
- `meta` — произвольные сериализуемые метаданные,
- `x-*` — расширения.

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

**Правила merge:**
- merge выполняется **только по отсутствующим ключам** (явные параметры в критерии всегда сильнее дефолтов);
- merge выполняется отдельно по группам: `criterion/selector/transform/aggregator/comparator`;
- `defaults` не должны содержать невалидных значений (`NaN/Inf`) и должны проходить ту же строгую проверку параметров.

---

## 6. Структура Criterion

Каждый элемент `criteria` **ДОЛЖЕН** иметь вид:

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
- `name` **ДОЛЖНО** быть уникальным в пределах `criteria` (иначе конфликт ключей в `report`);
- `weight` **ДОЛЖЕН** быть конечным числом; **СЛЕДУЕТ** требовать `weight > 0`;
- `selector/aggregator/comparator` обязательны;
- отсутствие `transform` или `transform: null` трактуется как **identity** (ничего не делать).

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

`params` **МОЖЕТ** отсутствовать. Тогда считается `params: {}`.

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

Loader **СЛЕДУЕТ** делать строгим:
- неизвестные параметры в `params` → **ошибка** (`UnknownParam`),
- отсутствующие обязательные параметры конструктора → **ошибка** (`MissingParam`),
- неверные типы/диапазоны → **ошибка** (`InvalidValue`).

Реализация **ДОЛЖНА** проверять параметры через introspection (например, `inspect.signature(__init__)`),
а также выполнять пост-валидацию (например, запрет `NaN/Inf`, корректность единиц и т.п.).

---

## 8. Спецификация Transform (TransformSpec)

Поле `transform` в Criterion **МОЖЕТ** быть задано четырьмя способами.

### 8.1. Отсутствует / null (identity)

```yaml
transform: null
```

или поле `transform` отсутствует — это **identity** (ничего не делать).

### 8.2. Одиночный Transform (ComponentSpec)

```yaml
transform:
  type: FiniteTransform
  params: { mode: drop }
```

### 8.3. Канонический Compose (рекомендуемый стандарт)

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
- `transforms` **ДОЛЖНО** быть списком `ComponentSpec`;
- `Compose` **СЛЕДУЕТ** использовать как **каноническое** представление цепочек.

### 8.4. Синтаксический сахар: список transforms

Допускается:

```yaml
transform:
  - { type: FiniteTransform, params: { mode: drop } }
  - { type: SmoothPointsTransform, params: { n_pts: 9 } }
```

Правило:
- loader преобразует такой список в канонический `Compose`.

---

## 9. Registry, `type` и алиасы

### 9.1. `type`

Каждый компонент в файле идентифицируется по строковому `type`, который:
- **ДОЛЖЕН** существовать в соответствующем registry (`selector/transform/aggregator/comparator`);
- **ДОЛЖЕН** быть **каноническим алиасом** (см. ниже);
- трактуется **case-sensitive**.

### 9.2. Канонический алиас

Один и тот же класс может регистрироваться под несколькими алиасами.

**Правило канона:**
- канонический алиас = **первый** алиас в регистрации (либо строка, если передана одна).

Это обеспечивает:
- стабильный формат файлов,
- возможность поддерживать «старые имена» как синонимы в loader.

### 9.3. Неизвестный `type`

Неизвестный `type` **ДОЛЖЕН** приводить к ошибке, содержащей:
- путь в документе (например, `criteria[2].selector.type`),
- категорию (`selector/transform/aggregator/comparator`),
- список известных типов или “ближайшие совпадения” (по Levenshtein/`difflib`).

---

## 10. Единицы и соглашения (critical)

### 10.1. Единицы частоты (`freq_unit`)

Канонические значения: `"Hz"`, `"kHz"`, `"MHz"`, `"GHz"`.

Loader **ДОЛЖЕН** нормализовать единицы через `normalize_freq_unit()`
(например, регистр/пробелы допускаются при загрузке).

### 10.2. Семантические единицы значений (`value_unit`)

`value_unit` задаётся селектором и является **семантической меткой** (не обязательно строгой SI-единицей).

Примеры: `"db"`, `"lin"`, `"rad"`, `"deg"`, `"complex"`, `"ns"`, `"s"`, `""`.

Рекомендуется нормализовать строку: `strip()`, для некоторых unit — `lower()`,
но конкретные правила остаются на стороне mwlab.

### 10.3. Band у Selector

Если у селектора есть `band`, то:
- `band` интерпретируется **в единицах `selector.freq_unit`**;
- применяется как **маска по существующим точкам сетки** (без интерполяции граничных точек).

Если нужна строгая полоса с включением границ — используйте `BandTransform(include_edges=true)`.

### 10.4. Параметры Transform/Aggregator с частотным смыслом (`*_unit`)

Для параметров с частотным смыслом используются поля `*_unit`:

- `BandTransform`: `band_unit`
- `SmoothApertureTransform` и `ApertureSlopeTransform`: `fw_unit`
- `ValueAtAgg`: `f0_unit`

Правило:
- если `*_unit` не задан, параметр трактуется в текущих единицах частоты;
- если задан, unit-aware выполнение и/или loader должны корректно приводить величину к текущему `freq_unit`.

---

## 11. LineSpec: линии limit/target

Некоторые агрегаторы используют **опорные/пороговые линии** (например, `UpIntAgg.limit`, `RippleIntAgg.target`).

Разрешённые формы `LineSpec`:
1) **Число** (float/int) — константа.
2) **Табличная линия** — список пар `[[f,y], [f,y], ...]`, минимум 2 точки, частоты строго возрастают.

Каноническая YAML-форма:

```yaml
limit:
  - [1.0, -10.0]
  - [2.0, -10.0]
```

Требования:
- повторяющиеся значения частоты в линии **запрещены** (иначе интерполяция неоднозначна),
- диапазон частоты линии **должен покрывать** частоты агрегирования (иначе ошибка).

---

## 12. Unit-aware исполнение и dry-валидация цепочки

### 12.1. Режим вычисления

В вычислительном проходе критерия **СЛЕДУЕТ** использовать unit-aware API:

- `Transform.apply(freq, vals, freq_unit=..., value_unit=...)`
- `Aggregator.aggregate(freq, vals, freq_unit=..., value_unit=...)`

Это важно для корректной физики (производные в Hz, GD, апертуры и т.п.).

### 12.2. Dry-валидация при загрузке (без `rf.Network`)

Loader **СЛЕДУЕТ** выполнять проверку совместимости цепочки **до вычислений**, используя:

- `selector.freq_unit` и `selector.value_unit` как стартовые единицы,
- `transform.out_value_unit(...)` для протаскивания `value_unit` по цепочке,
- атрибуты `expects_value_unit` / `expects_freq_unit` у компонентов, если они заданы,
- проверку наличия unit-aware методов (`Transform.apply`, `Aggregator.aggregate`) для выбранного режима работы.

Если `expects_value_unit` — список/кортеж, то допустим любой из перечисленных.

### 12.3. Примеры конфигурационных ошибок, которые надо ловить заранее

- `GroupDelayTransform` требует `value_unit ∈ {"rad","deg"}`.
- Интегральные агрегаторы с `basis="Hz"` требуют вызова `aggregate(..., freq_unit=...)` (unit-aware), а не `__call__`.
- `Compose.apply` требует, чтобы все transforms имели `apply`.

---

## 13. Политики NaN/Inf и пустых данных

Serde-контракт **сохраняет** параметры политик как часть `params`, например:
- Transform: `FiniteTransform(mode=...)`, `on_empty=...`
- Aggregator: `finite_policy`, `on_empty`, `complex_mode`
- Comparator: `finite_policy`, `non_finite_penalty`

**Важно:** семантика этих политик определяется реализацией классов; serde лишь фиксирует, как это хранится.

---

## 14. Dump (сериализация) — требования

### 14.1. Общие правила dump

При dump `Specification` в dict (и далее в YAML/JSON):

- всегда писать `format/version`,
- всегда писать канонический `type`,
- все значения приводить к python-примитивам (например, `np.float64 -> float`),
- complex — только через `{"__complex__":[re,im]}`,
- не записывать несериализуемые объекты (ошибка или пропуск по политике, но рекомендуется ошибка).

### 14.2. Режимы dump

Разрешены два режима:

1) **Полный (explicit)** — писать все параметры.
2) **Компактный (diff-from-default)** — писать только параметры, отличающиеся от дефолтов конструктора.

Контракт допускает оба режима. Для устойчивости CI/диффов **СЛЕДУЕТ** выбрать один режим как стандарт проекта
(рекомендуется `explicit` для v1).

### 14.3. Канонизация transforms при dump

Рекомендуется опция `canonical_transforms=true`, тогда:
- отсутствие transform не пишется или пишется `null` (на выбор проекта),
- одиночный transform может быть записан как `Compose` с одним элементом (для унификации),
- список transforms всегда пишется как `Compose`.

---

## 15. Диагностика ошибок (рекомендуемый стандарт)

Ошибка загрузки/валидации **СЛЕДУЕТ** включать:

- `path` (пример: `criteria[0].transform.params.transforms[2].params.basis`)
- `kind` (например: `UnknownType`, `UnknownParam`, `MissingParam`, `InvalidValue`, `UnitMismatch`, `NonFiniteParam`)
- человекочитаемое сообщение
- (опционально) `hints`: подсказки (например, близкие `type` по Levenshtein)

---

## 16. Пример полного YAML

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

## 17. Пример JSON (канонический)

```json
{
  "format": "mwlab.spec",
  "version": 1,
  "name": "demo_spec",
  "criteria": [
    {
      "name": "S11 (dB) <= -10 in 1..2 GHz",
      "weight": 1.0,
      "selector": { "type": "SMagSelector", "params": { "m": 1, "n": 1, "db": true, "band": [1.0, 2.0] } },
      "transform": { "type": "SignTransform", "params": { "sign": -1 } },
      "aggregator": { "type": "MaxAgg", "params": { "finite_policy": "omit", "on_empty": "raise", "complex_mode": "raise" } },
      "comparator": { "type": "LEComparator", "params": { "limit": 10.0, "unit": "db", "finite_policy": "fail", "non_finite_penalty": 1.0 } }
    }
  ]
}
```

---

## 18. Рекомендации по реализации loader/dumper (вне формата)

Эти пункты не являются требованиями формата, но крайне полезны для реализации:

1) Factory по registry: `make_component(kind, type, params)`  
2) Интроспекция конструктора для строгой проверки параметров (`inspect.signature(__init__)`)  
3) Нормализация единиц на входе (`normalize_freq_unit`, `strip/lower`)  
4) Миграции версий: `version -> migration -> v1-структура -> build`  
5) Канонизация при dump: `Compose` и канонические `type`  
6) Запрет `NaN/Inf` сразу после parse с выдачей `path`

---

## 19. Чек-лист совместимости компонентов с serde

Компоненты (Selector/Transform/Aggregator/Comparator), которые участвуют в serde, должны:
- не требовать `callable`/несериализуемых объектов в параметрах,
- хранить параметры в атрибутах (для dump),
- иметь зарегистрированный `type` и канонический алиас,
- (желательно) поддерживать unit-aware API (`apply` / `aggregate`).

---

## 20. История версий

- **v1** — первоначальная версия контракта.
