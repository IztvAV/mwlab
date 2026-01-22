# mwlab/opt/objectives/serde/pvf.py
"""
Конвертер PVF/WEAVE-спецификаций ( *.pvf / *.weave ) -> mwlab.spec v1 (YAML/JSON).

Зачем нужен этот модуль
-----------------------
Файлы PVF обычно генерируются внешними инструментами. Они представляют набор
объектов вида:

    obj(<id>) {
        Class = Project / Limit / Ripple / Band / ISSPlot / ...
        ...
    }

Наша цель — преобразовать эти объекты в спецификацию библиотеки mwlab
(контракт "mwlab.spec", version=1), то есть в список mwlab-критериев.

Поддерживаемые PVF-объекты (MVP)
--------------------------------
- Project: Units, Variables, Description
- Limit  : верхние/нижние лимиты (по точке или по диапазону частот)
- Ripple : рябь (peak-to-peak) на диапазоне частот
- Band   : диапазоны (используем как метаданные; не превращаем в критерии)
- ISSPlot(Parameter="Slope"): параметры окна для расчёта наклона (*1/*2)

Потоки (stream) / бинарные блоки
--------------------------------
PVF/WEAVE может включать "потоки" с бинарными данными (например, массивы X/Y).
Они не нужны для ТЗ и мешают парсингу, поэтому мы их удаляем на этапе препроцессинга.

Поддерживаем два распространённых синтаксиса потоков:

1) WEAVE v1.0:   stream(<N>){<N байт произвольных данных>}
   - Важно: содержимое может содержать любые байты, включая переводы строк.

2) WEAVE v1.1+:  <ключ>=< ... >\\n   (угловые скобки с бинарным содержимым)
   - Блок начинается с '=<'
   - Блок заканчивается на маркере '>\\n' или '>\\r\\n', после которого начинается
     "похожий на текст" PVF-фрагмент (ключ=..., obj(...), } и т.п.).
   - Чтобы избежать ложных срабатываний на '>\\n' внутри бинарных байт, мы ищем
     конец по эвристике "после маркера начинается осмысленный PVF-текст".

Результат конвертации
---------------------
- Функция pvf_to_mwlab_dict(...) возвращает dict в формате mwlab.spec v1.
- Функция convert_pvf_to_yaml(...) пишет YAML-файл.

Правила преобразования PVF -> mwlab.criteria (кратко)
----------------------------------------------------
Limit:
  - selector: по PlotType
      * MgDB, *1 -> SMagSelector(db=True)
      * GD, Phase, *2 -> PhaseSelector(unwrap=True, unit="rad") + (GroupDelayTransform для GD/*2)
  - transform:
      * GD/*2 -> GroupDelayTransform(out_unit=<Time unit проекта>)
      * *1/*2 -> ApertureSlopeTransform(fw=<SlopeWindow из ISSPlot для MgDB или GD>)
      * YShifter -> ShiftByRefInBandTransform(band=[X1,X2], ref=max|min|mean|median)
      * если Limit задан по диапазону (X1!=X2) -> BandTransform(band=[X1,X2], include_edges=True)
  - aggregator:
      * диапазон: max (Upper) / min (Lower)
      * точка (X1==X2): ValueAtAgg(f0=X1)
  - comparator:
      * Upper -> le(limit=Y1)
      * Lower -> ge(limit=Y1)
  - Ограничение MVP: поддерживаем только "плоский" лимит (Y1 == Y2).
    Для наклонной линии нужен отдельный mwlab-примитив (его здесь не предполагаем).

Ripple:
  - selector: аналогично (обычно MgDB или GD)
  - transform:
      * BandTransform(band=[X1,X2], include_edges=True)
      * + GroupDelayTransform если GD
      * + YShifter при наличии
  - aggregator: ripple
  - comparator: le(limit=<Threshold>)

Band:
  - в критерии не конвертируем; сохраняем в meta["pvf"]["bands"].

ISSPlot(Parameter="Slope"):
  - извлекаем SlopeWindow для PlotType ("MgDB"/"GD"/...) и используем для *1/*2.

Зависимости
-----------
- PyYAML (опционально) — только для записи YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

from ..registry import canonicalize_alias

# =============================================================================
# Ошибки конвертера
# =============================================================================

class PvfError(ValueError):
    """Базовая ошибка PVF-парсинга/конвертации с понятным текстом."""
    pass


# =============================================================================
# PVF AST (минимальная модель)
# =============================================================================

@dataclass(frozen=True)
class PvfObject:
    """
    Один объект PVF:

        obj(<id>) { ... }

    data — вложенный словарь, который мы собираем из блоков {...} и полей key=value.
    """
    obj_id: int
    data: Dict[str, Any]


# =============================================================================
# Низкоуровневые утилиты
# =============================================================================

_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_plausible_pvf_text_start(buf: bytes) -> bool:
    """
    Эвристика: похоже ли, что buf начинается с "осмысленного" PVF текста.
    Используется для поиска конца бинарного блока в синтаксисе =< ... >\\n.

    Мы разрешаем:
    - пробелы/табуляцию,
    - далее: '}' или '#', или 'obj(', или 'KEY = ...' (KEY: [A-Za-z_][A-Za-z0-9_]*)
    """
    i = 0
    while i < len(buf) and buf[i] in b" \t\r":
        i += 1
    if i >= len(buf):
        return False

    if buf[i:i + 1] in b"}#":
        return True
    if buf[i:i + 4] == b"obj(":
        return True

    c = buf[i]
    if (65 <= c <= 90) or (97 <= c <= 122) or c == 95:  # A-Z a-z _
        j = i + 1
        while j < len(buf):
            cj = buf[j]
            if (65 <= cj <= 90) or (97 <= cj <= 122) or (48 <= cj <= 57) or cj == 95:
                j += 1
            else:
                break
        k = j
        while k < len(buf) and buf[k] in b" \t":
            k += 1
        if k < len(buf) and buf[k:k + 1] == b"=":
            return True

    return False


def _strip_angle_streams(data: bytes) -> bytes:
    """
    Удалить бинарные блоки формата:
        <ключ>=< ... бинарные байты ... >\\n

    Реализация:
    - ищем маркер '=<'
    - заменяем значение на '=[]' (чтобы синтаксически оставаться в key=value)
    - конец ищем по первому кандидату '>\\n' / '>\\r\\n',
      после которого начинается "похожий на текст" PVF-фрагмент.
    """
    out = bytearray()
    i = 0
    n = len(data)

    while i < n:
        j = data.find(b"=<", i)
        if j == -1:
            out.extend(data[i:])
            break

        # Копируем всё до маркера (включая "key="), а значение заменяем на "[]".
        out.extend(data[i:j])
        out.extend(b"=[]")

        # Теперь пропускаем бинарный хвост начиная с "<".
        k = j + 2  # указывает на символ '<'
        scan = k + 1

        term_end: Optional[int] = None
        newline: bytes = b"\n"

        while True:
            cand_nl = data.find(b">\n", scan)
            cand_crlf = data.find(b">\r\n", scan)

            if cand_nl == -1 and cand_crlf == -1:
                break

            if cand_crlf != -1 and (cand_nl == -1 or cand_crlf < cand_nl):
                cand = cand_crlf
                term_len = 3
                nl = b"\r\n"
            else:
                cand = cand_nl
                term_len = 2
                nl = b"\n"

            after = cand + term_len
            if _is_plausible_pvf_text_start(data[after:after + 80]):
                term_end = after
                newline = nl
                break

            # Если это "ложный конец" (например, байты внутри потока),
            # продолжаем поиск дальше.
            scan = cand + 1

        if term_end is None:
            # Не нашли корректный конец — лучше не портить файл.
            # Возвращаем исходный хвост как есть (парсер потом упадёт с понятной ошибкой).
            out.extend(data[j:])
            break

        # Восстанавливаем перевод строки и продолжаем после конца потока.
        out.extend(newline)
        i = term_end

    return bytes(out)


def _strip_stream_v10(data: bytes) -> bytes:
    """
    Удалить бинарные блоки формата WEAVE v1.0:
        stream(<N>){<N байт>}

    Мы заменяем всё выражение stream(...) { ... } на литерал '[]' (пустой список),
    чтобы сохранить корректность key=value (если stream стоял справа).

    Алгоритм:
    - ищем подстроку b"stream("
    - читаем целое N до ')'
    - пропускаем пробелы до '{'
    - пропускаем ровно N байт содержимого
    - ожидаем закрывающую '}'
    """
    out = bytearray()
    i = 0
    n = len(data)

    while i < n:
        j = data.find(b"stream(", i)
        if j == -1:
            out.extend(data[i:])
            break

        # Копируем всё до stream(
        out.extend(data[i:j])

        # Пытаемся распарсить длину.
        p = j + len(b"stream(")
        q = data.find(b")", p)
        if q == -1:
            # Некорректная форма — оставляем как есть.
            out.extend(data[j:])
            break

        len_bytes = data[p:q].strip()
        if not len_bytes.isdigit():
            out.extend(data[j:])
            break
        length = int(len_bytes)

        # После ')' может быть whitespace, затем '{'
        r = q + 1
        while r < n and data[r] in b" \t\r\n":
            r += 1
        if r >= n or data[r:r + 1] != b"{":
            out.extend(data[j:])
            break

        body_start = r + 1
        body_end = body_start + length
        if body_end > n:
            out.extend(data[j:])
            break

        # После тела должен идти '}' (возможны пробелы перед ним — но по описанию чаще сразу)
        s = body_end
        if s < n and data[s:s + 1] == b"}":
            # Ок, закрывающая скобка прямо здесь.
            i = s + 1
            out.extend(b"[]")
            continue

        # Иногда после N байт могут быть пробелы, затем '}' — допускаем.
        t = s
        while t < n and data[t] in b" \t\r\n":
            t += 1
        if t < n and data[t:t + 1] == b"}":
            i = t + 1
            out.extend(b"[]")
            continue

        # Если не нашли закрывающую '}' — не трогаем.
        out.extend(data[j:])
        break

    return bytes(out)


def strip_pvf_streams(data: bytes) -> bytes:
    """
    Полный препроцессинг: убираем бинарные потоки обоих типов.

    Порядок важен:
    - сначала stream(v1.0) — он текстово размечен и может встречаться в "текстовой" части,
    - затем угловые =<...> — там обычно реально сырые байты.
    """
    data2 = _strip_stream_v10(data)
    data3 = _strip_angle_streams(data2)
    return data3


def detect_pvf_encoding(text_head: str) -> str:
    """
    Определить кодировку по заголовку.

    В PVF/WEAVE часто встречается строка:
        #ENCODING utf-8

    Если не нашли — возвращаем 'utf-8'.
    """
    for line in text_head.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            # обычно заголовки идут подряд в начале файла
            continue
        if line.upper().startswith("#ENCODING"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].strip()
    return "utf-8"


# =============================================================================
# Парсер текстовой части PVF
# =============================================================================

def _parse_list(raw: str) -> List[str]:
    """
    PVF-списки обычно выглядят как:
        [ "a" "b" "c" ]
        [ "s2:1" ]
        [ *Something ]   (звёздочка — ссылка/алиас в некоторых диалектах)

    Мы:
    - убираем внешние [ ... ]
    - заменяем запятые на пробелы (на случай "[a, b]")
    - разбиваем shlex.split (корректно понимает кавычки)
    """
    content = raw.strip()[1:-1].strip()
    if not content:
        return []
    content = content.replace(",", " ")
    items = shlex.split(content)

    # Убираем ведущую "*" у ссылок вида "*Foo"
    out: List[str] = []
    for item in items:
        if item.startswith("*"):
            out.append(item[1:])
        else:
            out.append(item)
    return out


def _parse_value(raw: str) -> Any:
    """
    Разбор PVF значения на примитивы:
    - [ ... ] -> list[str]
    - "..." / '...' -> str
    - true/false -> bool
    - число -> int/float
    - иначе -> raw str (без изменений)
    """
    val = raw.strip()
    if val.startswith("[") and val.endswith("]"):
        return _parse_list(val)

    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]

    lower = val.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    if _NUM_RE.match(val):
        # Отличаем int от float по наличию "." или экспоненты
        if "." in val or "e" in lower:
            return float(val)
        return int(val)

    return val


def parse_pvf_text(text: str) -> List[PvfObject]:
    """
    Спарсить PVF/WEAVE текст (после вырезания stream-блоков) в список PvfObject.

    Поддерживаем:
    - obj(<id>) { ... }
    - вложенные блоки вида:
        Key = {
            ...
        }
      или "Key={"
    - строки key=value

    Замечание:
    PVF-грамматика богаче, но для ТЗ обычно хватает именно этой подмножины.
    """
    objects: List[PvfObject] = []
    stack: List[Dict[str, Any]] = []

    for ln, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        # Пустые строки и комментарии
        if not line or line.startswith("#"):
            continue

        # Начало объекта: obj(123){
        if line.startswith("obj(") and line.endswith("{"):
            try:
                obj_id = int(line[4: line.index(")")])
            except Exception as e:
                raise PvfError(f"PVF parse error at line {ln}: invalid obj(id): {line!r} ({e})") from e

            obj_dict: Dict[str, Any] = {}
            objects.append(PvfObject(obj_id=obj_id, data=obj_dict))
            stack.append(obj_dict)
            continue

        # Закрытие блока/объекта
        if line == "}":
            if not stack:
                raise PvfError(f"PVF parse error at line {ln}: unexpected '}}'")
            stack.pop()
            continue

        # Открытие вложенного блока: Key={  или Key = {
        if line.endswith("{"):
            if not stack:
                raise PvfError(f"PVF parse error at line {ln}: orphaned block start: {line!r}")

            head = line[:-1].strip()
            # допускаем формы "Key=" и "Key ="
            if head.endswith("="):
                head = head[:-1].strip()

            if not head:
                raise PvfError(f"PVF parse error at line {ln}: empty block key in: {line!r}")

            new_dict: Dict[str, Any] = {}
            stack[-1][head] = new_dict
            stack.append(new_dict)
            continue

        # Присваивание key=value
        if "=" in line:
            if not stack:
                raise PvfError(f"PVF parse error at line {ln}: key=value outside of object: {line!r}")

            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                raise PvfError(f"PVF parse error at line {ln}: empty key in: {line!r}")

            stack[-1][key] = _parse_value(val)
            continue

        # Если дошли сюда — строка неизвестного вида.
        # Для MVP лучше не падать жёстко, но и молча игнорировать опасно.
        # Поэтому включаем "мягкую строгость": считаем это ошибкой.
        raise PvfError(f"PVF parse error at line {ln}: unsupported syntax: {line!r}")

    if stack:
        # Если стек не пуст, значит где-то не закрыли блок.
        raise PvfError("PVF parse error: unclosed '{' blocks (file ended too early)")

    return objects


def read_pvf_objects(path: Union[str, Path]) -> List[PvfObject]:
    """
    Прочитать PVF/WEAVE файл, удалить stream-блоки, декодировать и распарсить в объекты.
    """
    p = Path(path)
    data = p.read_bytes()

    # Кодировку берём из заголовка (до первого бинарного мусора)
    head = data[:4096].decode("ascii", errors="ignore")
    encoding = detect_pvf_encoding(head)

    # Удаляем потоки (бинарные вставки)
    cleaned = strip_pvf_streams(data)

    try:
        text = cleaned.decode(encoding, errors="strict")
    except Exception:
        # Если кодировка/данные "грязные", используем replace, чтобы сохранить структуру.
        text = cleaned.decode(encoding, errors="replace")

    return parse_pvf_text(text)


# =============================================================================
# Безопасная обработка переменных (Variables) и выражений
# =============================================================================

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _safe_eval_expr(expr: str, variables: Mapping[str, float]) -> float:
    """
    Безопасно вычислить арифметическое выражение PVF.

    Разрешаем:
    - числа (int/float),
    - имена переменных (Name),
    - унарные +/-,
    - бинарные + - * / **,
    - скобки (через AST).

    Запрещаем:
    - вызовы функций, атрибуты, индексацию, любые другие узлы AST.

    Если встречается неизвестное имя — KeyError(name).
    Если встречается неподдерживаемый синтаксис — ValueError.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expression")

    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Python 3.8+: числа обычно в Constant
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError("PVF expression: non-numeric constant")

        # На всякий случай для старых версий AST
        if isinstance(node, ast.Num):  # pragma: no cover
            return float(node.n)

        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise KeyError(node.id)
            return float(variables[node.id])

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
            v = _eval(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v

        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            a = _eval(node.left)
            b = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b

        raise ValueError(f"PVF expression: unsupported syntax node {type(node).__name__}")

    return float(_eval(tree))


def resolve_pvf_variables(variables: Mapping[str, Any]) -> Dict[str, float]:
    """
    Разрешить Variables из Project.Variables в числа.

    PVF допускает, что переменная задаётся:
    - числом,
    - строкой-числом,
    - выражением, использующим другие переменные.

    Разрешаем зависимости в "несколько проходов" до стабилизации.
    Если за проход не смогли продвинуться — значит цикл или неизвестные имена.
    """
    resolved: Dict[str, float] = {}
    pending: Dict[str, Any] = dict(variables)

    while pending:
        progressed = False

        for key in list(pending.keys()):
            raw = pending[key]

            # Число
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                resolved[key] = float(raw)
                pending.pop(key)
                progressed = True
                continue

            # Строка: число или выражение
            if isinstance(raw, str):
                s = raw.strip()
                if _NUM_RE.match(s):
                    resolved[key] = float(s)
                    pending.pop(key)
                    progressed = True
                    continue
                try:
                    resolved[key] = _safe_eval_expr(s, resolved)
                except KeyError:
                    # зависит от ещё не вычисленных переменных
                    continue
                pending.pop(key)
                progressed = True
                continue

            raise PvfError(f"PVF Variables: unsupported value for {key!r}: {raw!r}")

        if not progressed:
            # Не можем вычислить оставшиеся: цикл/пропущенные имена.
            missing = ", ".join(sorted(pending.keys()))
            raise PvfError(f"PVF Variables: cannot resolve expressions for: {missing}")

    return resolved


def eval_numeric(value: Any, variables: Mapping[str, float], *, what: str) -> float:
    """
    Привести PVF-значение к float, с учётом переменных/выражений.

    - int/float -> float
    - str:
        * "123" -> float
        * "F1 + 10" -> вычисляем выражение
    """
    if value is None:
        raise PvfError(f"PVF: missing numeric value for {what}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    if isinstance(value, str):
        s = value.strip()
        if _NUM_RE.match(s):
            return float(s)
        return float(_safe_eval_expr(s, variables))

    raise PvfError(f"PVF: unsupported numeric value for {what}: {value!r}")


# =============================================================================
# Нормализация единиц (Units)
# =============================================================================

def map_freq_unit(unit: Any) -> str:
    """
    PVF хранит частотные единицы часто как 'kilo/mega/giga' или 'Hz/kHz/MHz/GHz'.

    Возвращаем строку для mwlab: 'Hz', 'kHz', 'MHz', 'GHz'.
    По умолчанию — 'MHz' (самая частая в RF задачах).
    """
    u = str(unit or "").strip().lower()
    mapping = {
        "hz": "Hz",
        "kilo": "kHz",
        "khz": "kHz",
        "mega": "MHz",
        "mhz": "MHz",
        "giga": "GHz",
        "ghz": "GHz",
    }
    return mapping.get(u, "MHz")


def resolve_time_unit(unit: Any) -> Tuple[str, float]:
    """
    PVF хранит единицы времени как 'nano/micro/milli' или 'ns/us/ms'.

    В mwlab GroupDelayTransform поддерживает только 's' и 'ns', поэтому:
    - выбираем допустимую единицу,
    - возвращаем коэффициент пересчёта PVF->выбранная_единица.

    Возвращает (out_unit, scale), где:
      out_unit: "s" | "ns"
      scale   : множитель для значений (PVF_value * scale -> out_unit)
    """
    u = str(unit or "").strip().lower()
    mapping: Dict[str, Tuple[str, float]] = {
        "s": ("s", 1.0),
        "sec": ("s", 1.0),
        "second": ("s", 1.0),
        "nano": ("ns", 1.0),
        "ns": ("ns", 1.0),
        "micro": ("ns", 1.0e3),
        "us": ("ns", 1.0e3),
        "milli": ("ns", 1.0e6),
        "ms": ("ns", 1.0e6),
    }
    return mapping.get(u, ("ns", 1.0))


# =============================================================================
# Преобразование PVF объектов в mwlab criteria
# =============================================================================

def _parse_attachment_sparam(attachments: Any) -> Tuple[int, int]:
    """
    Извлечь (m, n) из Attachments.

    В PVF это обычно список строк вида: ["s2:1"].
    """
    if attachments is None:
        return 1, 1

    items: List[str]
    if isinstance(attachments, str):
        items = [attachments]
    elif isinstance(attachments, list):
        items = [str(x) for x in attachments]
    else:
        items = [str(attachments)]

    for att in items:
        s = att.strip()
        if not s:
            continue
        # допускаем 's2:1' или 'S2:1'
        if (s[0] in "sS") and ":" in s[1:]:
            a, b = s[1:].split(":", 1)
            if a.isdigit() and b.isdigit():
                return int(a), int(b)

    return 1, 1


def selector_spec_for_plot(plot_type: str, attachments: Any, freq_unit: str) -> Dict[str, Any]:
    """
    Построить selector ComponentSpec по PlotType.

    Маппинг (MVP):
    - MgDB, *1            -> SMagSelector(db=True)
    - GD, Phase, *2       -> PhaseSelector(unwrap=True, unit="rad")
    """
    pt = (plot_type or "").strip()

    m, n = _parse_attachment_sparam(attachments)

    if pt in ("GD", "Phase", "*2"):
        return {
            "type": canonicalize_alias("selector", "PhaseSelector"),
            "params": {
                "m": m,
                "n": n,
                "unwrap": True,
                "unit": "rad",
                "freq_unit": freq_unit,
            },
        }

    # По умолчанию считаем, что это модуль S-параметра в dB
    return {
        "type": canonicalize_alias("selector", "SMagSelector"),
        "params": {
            "m": m,
            "n": n,
            "db": True,
            "freq_unit": freq_unit,
        },
    }


def base_transforms_for_plot(
    plot_type: str,
    slope_fw: Optional[float],
    time_unit: str,
) -> List[Dict[str, Any]]:
    """
    Базовые transforms, зависящие от PlotType.

    - GD / *2 : сначала GroupDelayTransform (из фазы)
    - *1 / *2 : затем ApertureSlopeTransform (окно fw берём из ISSPlot.SlopeWindow)
    """
    pt = (plot_type or "").strip()
    transforms: List[Dict[str, Any]] = []

    if pt in ("GD", "*2"):
        transforms.append(
            {
                "type": canonicalize_alias("transform", "GroupDelayTransform"),
                "params": {
                    "out_unit": time_unit,
                },
            }
        )

    if pt in ("*1", "*2"):
        if slope_fw is None:
            raise PvfError(
                f"PVF: PlotType={pt} requires slope window (ISSPlot Parameter=Slope), but it was not found."
            )
        transforms.append(
            {
                "type": canonicalize_alias("transform", "ApertureSlopeTransform"),
                "params": {
                    "fw": float(slope_fw),
                },
            }
        )

    return transforms


def yshifter_transform(
    yshifter: Any,
    variables: Mapping[str, float],
    freq_unit: str,
) -> Optional[Dict[str, Any]]:
    """
    Конвертация PVF.YShifter в mwlab ShiftByRefInBandTransform.

    PVF пример:
        YShifter={
            Type=Maximum
            X1=...
            X2=...
        }

    В mwlab:
        ShiftByRefInBandTransform(band=[x1,x2], ref="max", band_unit=<freq_unit>)
    """
    if not isinstance(yshifter, Mapping):
        return None

    ref_type = str(yshifter.get("Type", "")).strip().upper()
    ref_map = {
        "MAXIMUM": "max",
        "MINIMUM": "min",
        "MEAN": "mean",
        "MEDIAN": "median",
    }
    ref = ref_map.get(ref_type)
    if not ref:
        # Неизвестный тип — просто игнорируем (лучше, чем падать).
        return None

    x1 = eval_numeric(yshifter.get("X1"), variables, what="YShifter.X1")
    x2 = eval_numeric(yshifter.get("X2"), variables, what="YShifter.X2")

    return {
        "type": canonicalize_alias("transform", "ShiftByRefInBandTransform"),
        "params": {
            "band": [x1, x2],
            "ref": ref,
            "band_unit": freq_unit,
        },
    }


def _unique_name(desired: str, used: Dict[str, int]) -> str:
    """
    Сделать имя уникальным (на случай дубликатов Name в PVF).

    used[name] хранит счётчик.
    """
    base = desired.strip() or "criterion"
    if base not in used:
        used[base] = 0
        return base
    used[base] += 1
    return f"{base}__{used[base]}"


def _scale_time_value(value: float, plot_type: str, time_scale: float) -> float:
    """
    Применить масштаб времени для GD/*2, если Units.Time не в ns/s.
    """
    pt = (plot_type or "").strip()
    if pt in ("GD", "*2"):
        return value * time_scale
    return value


def pvf_to_mwlab_dict(
    objects: Sequence[PvfObject],
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Преобразовать список PvfObject в mwlab.spec v1 (dict).

    Важно:
    - Возвращаем именно dict (не Specification-объект).
    - Этот dict можно записать в YAML и затем загрузить через mwlab.opt.objectives.serde.load_spec_dict().
    """
    objs = list(objects)

    # --- Project: единицы/переменные/описание ---
    project = next((o.data for o in objs if o.data.get("Class") == "Project"), {})
    units = cast(Mapping[str, Any], project.get("Units", {}) or {})
    variables_raw = cast(Mapping[str, Any], project.get("Variables", {}) or {})
    description = cast(Mapping[str, Any], project.get("Description", {}) or {})

    resolved_vars = resolve_pvf_variables(variables_raw) if variables_raw else {}

    freq_unit = map_freq_unit(units.get("Frequency", "mega"))
    time_unit, time_scale = resolve_time_unit(units.get("Time", "nano"))

    # --- ISSPlot(Slope): извлекаем slope окна по PlotType ---
    slope_windows: Dict[str, float] = {}
    for o in objs:
        d = o.data
        if d.get("Class") != "ISSPlot":
            continue
        if str(d.get("Parameter", "")).strip() != "Slope":
            continue

        pt = str(d.get("PlotType", "")).strip()
        if not pt:
            continue

        sw = d.get("SlopeWindow", None)
        if sw is None:
            continue

        slope_windows[pt] = eval_numeric(sw, resolved_vars, what=f"ISSPlot(Slope).SlopeWindow[{pt}]")

    # --- Bands как метаданные ---
    bands_meta: List[Dict[str, Any]] = []
    for o in objs:
        d = o.data
        if d.get("Class") != "Band":
            continue
        bands_meta.append(
            {
                "obj_id": o.obj_id,
                "name": d.get("Name", ""),
                "type": d.get("Type", ""),
                "level": d.get("Level"),
                "absolute": d.get("Absolute"),
                "plot_type": d.get("PlotType"),
                "attachments": d.get("Attachments", []),
            }
        )

    # --- Limit/Ripple -> criteria ---
    criteria: List[Dict[str, Any]] = []
    used_names: Dict[str, int] = {}

    for o in objs:
        d = o.data
        cls = d.get("Class")
        if cls not in ("Limit", "Ripple"):
            continue

        plot_type = str(d.get("PlotType", "")).strip()
        attachments = d.get("Attachments", [])

        # selector
        selector = selector_spec_for_plot(plot_type, attachments, freq_unit=freq_unit)

        # transforms (по PlotType)
        slope_fw: Optional[float] = None
        # По договорённости: *1 использует slope окно "MgDB", *2 — окно "GD".
        if plot_type == "*1":
            slope_fw = slope_windows.get("MgDB")
        elif plot_type == "*2":
            slope_fw = slope_windows.get("GD")

        transforms: List[Dict[str, Any]] = base_transforms_for_plot(plot_type, slope_fw, time_unit)

        # YShifter
        ysh = yshifter_transform(d.get("YShifter"), resolved_vars, freq_unit=freq_unit)
        if ysh is not None:
            transforms.append(ysh)

        # Имя критерия (делаем уникальным)
        desired_name = str(d.get("Name") or f"{cls}_{o.obj_id}")
        crit_name = _unique_name(desired_name, used_names)

        # Конвертация Limit / Ripple
        if cls == "Limit":
            # Обязательные координаты
            x1 = eval_numeric(d.get("X1"), resolved_vars, what=f"{crit_name}.X1")
            x2 = eval_numeric(d.get("X2"), resolved_vars, what=f"{crit_name}.X2")
            y1 = eval_numeric(d.get("Y1"), resolved_vars, what=f"{crit_name}.Y1")
            y2 = eval_numeric(d.get("Y2"), resolved_vars, what=f"{crit_name}.Y2")
            y1 = _scale_time_value(y1, plot_type, time_scale)
            y2 = _scale_time_value(y2, plot_type, time_scale)

            # MVP: поддерживаем только плоскую линию
            if y1 != y2:
                raise PvfError(
                    f"PVF Limit '{crit_name}' uses non-flat line (Y1 != Y2). "
                    f"MVP converter supports only flat limits."
                )

            lim_type = str(d.get("Type", "")).strip().lower()  # 'upper' / 'lower'
            is_upper = (lim_type == "upper")
            is_lower = (lim_type == "lower")
            if not (is_upper or is_lower):
                # Если поле не задано, считаем upper по умолчанию (так чаще в TЗ),
                # но это место легко заменить на strict-ошибку при необходимости.
                is_upper = True

            # Если это диапазон (X1 != X2), ограничиваем band transform'ом и берём max/min
            if x1 != x2:
                transforms.append(
                    {
                        "type": canonicalize_alias("transform", "BandTransform"),
                        "params": {
                            "band": [x1, x2],
                            "band_unit": freq_unit,
                            "include_edges": True,
                        },
                    }
                )
                aggregator = {
                    "type": canonicalize_alias("aggregator", "max" if is_upper else "min"),
                    "params": {},
                }
            else:
                # Точка: берём значение в f0
                aggregator = {
                    "type": canonicalize_alias("aggregator", "ValueAtAgg"),
                    "params": {
                        "f0": x1,
                        "f0_unit": freq_unit,
                    },
                }

            comparator = {
                "type": canonicalize_alias("comparator", "le" if is_upper else "ge"),
                "params": {"limit": y1},
            }

        else:  # Ripple
            x1 = eval_numeric(d.get("X1"), resolved_vars, what=f"{crit_name}.X1")
            x2 = eval_numeric(d.get("X2"), resolved_vars, what=f"{crit_name}.X2")
            thr = eval_numeric(d.get("Threshold"), resolved_vars, what=f"{crit_name}.Threshold")
            thr = _scale_time_value(thr, plot_type, time_scale)

            transforms.append(
                {
                    "type": canonicalize_alias("transform", "BandTransform"),
                    "params": {
                        "band": [x1, x2],
                        "band_unit": freq_unit,
                        "include_edges": True,
                    },
                }
            )

            aggregator = {"type": canonicalize_alias("aggregator", "ripple"), "params": {}}
            comparator = {"type": canonicalize_alias("comparator", "le"), "params": {"limit": thr}}

        # --- ensure ApertureSlopeTransform is always the last transform ---
        AP_SLOPE = canonicalize_alias("transform", "ApertureSlopeTransform")
        if transforms:
            slope = [t for t in transforms if t.get("type") == AP_SLOPE]
            if slope:
                transforms[:] = [t for t in transforms if t.get("type") != AP_SLOPE] + slope

        # transform поле в mwlab может быть:
        # - отсутствовать (None)
        # - dict ComponentSpec
        # - list[ComponentSpec] (синтаксический сахар, serde.core это понимает)
        crit: Dict[str, Any] = {
            "name": crit_name,
            "weight": 1.0,              # PVF обычно не задаёт веса — ставим 1.0
            "assume_prepared": False,   # безопасное значение по умолчанию
            "selector": selector,
        }

        if transforms:
            crit["transform"] = transforms if len(transforms) > 1 else transforms[0]

        crit.update(
            {
                "aggregator": aggregator,
                "comparator": comparator,
                "meta": {
                    "pvf": {
                        "obj_id": o.obj_id,
                        "class": cls,
                        "plot_type": plot_type,
                    }
                },
            }
        )

        criteria.append(crit)

    spec_name = (
        name
        or str(description.get("ProjectName") or "").strip()
        or "spec"
    )

    # Итоговый mwlab.spec документ (v1)
    return {
        "format": "mwlab.spec",
        "version": 1,
        "name": spec_name,
        "criteria": criteria,
        "meta": {
            "pvf": {
                "units": dict(units),
                "variables": dict(variables_raw),
                "resolved_variables": dict(resolved_vars),
                "description": dict(description),
                "bands": bands_meta,
            }
        },
    }


# =============================================================================
# Запись YAML
# =============================================================================

def dumps_yaml_dict(doc: Mapping[str, Any], *, sort_keys: bool = False) -> str:
    """
    Записать dict в YAML (без python-тегов), как "чистый" JSON-поднабор YAML.

    Здесь мы намеренно НЕ используем serde.core.dumps_yaml(), потому что
    serde.core.dumps_yaml() ожидает Specification-объект, а не dict.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyYAML не установлен: YAML вывод недоступен. Установите пакет 'PyYAML'.") from e

    # allow_unicode=True — чтобы русские комментарии/имена не экранировались
    return yaml.safe_dump(dict(doc), sort_keys=sort_keys, allow_unicode=True)


def convert_pvf_to_yaml(pvf_path: Union[str, Path], yaml_path: Union[str, Path]) -> None:
    """
    Прочитать PVF/WEAVE файл и записать YAML в формате mwlab.spec v1.
    """
    pvf_path = Path(pvf_path)
    yaml_path = Path(yaml_path)

    objects = read_pvf_objects(pvf_path)
    spec_dict = pvf_to_mwlab_dict(objects, name=pvf_path.stem)

    yaml_text = dumps_yaml_dict(spec_dict, sort_keys=False)
    yaml_path.write_text(yaml_text, encoding="utf-8")


__all__ = [
    "PvfError",
    "PvfObject",
    "strip_pvf_streams",
    "parse_pvf_text",
    "read_pvf_objects",
    "resolve_pvf_variables",
    "pvf_to_mwlab_dict",
    "dumps_yaml_dict",
    "convert_pvf_to_yaml",
]