#mwlab/filters/cm_io.py
"""
cm_io.py — ввод/вывод расширенных матриц связи (Coupling Matrix I/O)
====================================================================

Назначение
----------
Экспорт и импорт **реальных расширенных матриц связи** (coupling-matrix)
в автономные файлы. Поддерживаются два формата:

* ``fmt="ascii"`` — текстовый файл: строки/столбцы через разделитель
  (по умолчанию таб). Первая строка — комментарий вида
  ``# mwlab-cm layout=SL order=4 ports=2``.
* ``fmt="json"`` — JSON-словарь с полями ``layout``, ``order``,
  ``ports``, ``M`` (список списков).

Главные функции
----------------
* :func:`write_matrix` — сохранить матрицу из :class:`CouplingMatrix`.
* :func:`read_matrix`  — прочитать файл и получить :class:`CouplingMatrix`.

Реализация использует Torch для внутренних представлений, но для ASCII
записи/чтения данные переводятся в NumPy для удобства ``savetxt/loadtxt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Sequence
import json
import re

import numpy as np
import torch

from .cm import CouplingMatrix, MatrixLayout, make_perm
from .cm_core import DT_R, DEFAULT_DEVICE

# тип для путей
PathLike = Union[str, Path]

# -----------------------------------------------------------------------------
#                                   WRITE
# -----------------------------------------------------------------------------
def write_matrix(
    cm: CouplingMatrix,
    path: PathLike,
    *,
    layout: MatrixLayout = MatrixLayout.SL,
    fmt: str = "ascii",
    precision: int = 15,
    delimiter: str = "\t",
    include_qu: bool = True,
    include_phase: bool = True,
    permutation: Sequence[int] | None = None,
) -> None:
    """
    Сохраняет расширенную матрицу связи на диск.

    Параметры
    ----------
    cm : CouplingMatrix
        Объект с матрицей связи (во внутреннем порядке TAIL).
    path : str | Path
        Куда писать файл.
    layout : MatrixLayout
        Макет портов в выходной матрице.
    fmt : {"ascii", "json"}
        Формат вывода.
    precision : int
        Кол-во значащих цифр для ASCII (`%.{precision}g`).
    delimiter : str
        Разделитель столбцов для ASCII.
    include_qu : bool
        Включать ли информацию о qu (скаляр или вектор).
    include_phase : bool
        Включать ли phase_a/phase_b (скаляр или вектор).

    Замечания
    ---------
    * Для ASCII всё пишется в одну комментарную строку вида:
      ``# mwlab-cm layout=SL order=4 ports=2 qu=700 phase_a1=0.1 phase_b2=0.0 ...``
    * Для JSON используются поля:
      - "qu" (скаляр) или "qu_vec" (вектор),
      - "phase_a" / "phase_b" (скаляры)
      - "phase_a_vec" / "phase_b_vec" (вектора).
    """
    path = Path(path)

    # --- матрица в нужном layout ---
    M = cm.to_matrix(layout=layout, permutation=permutation)
    if hasattr(M, "detach"):  # torch.Tensor
        M_np = M.detach().cpu().numpy()
    else:
        M_np = np.asarray(M, dtype=float)

    # --- подготовка meta-инфо (ASCII header) ---
    def _pack_phases(prefix: str, ph):
        """Возвращает словарь ключей phase_* для header/json."""
        if ph is None:
            return {}
        if np.isscalar(ph):
            return {prefix: float(ph)}
        arr = np.asarray(ph, dtype=float).ravel()
        return {f"{prefix}{i}": float(v) for i, v in enumerate(arr, 1)}

    if fmt == "ascii":
        meta = {
            "layout": layout.name,
            "order": cm.topo.order,
            "ports": cm.topo.ports,
        }

        if include_qu and cm.qu is not None:
            q = cm.qu
            if np.isscalar(q):
                meta["qu"] = float(q)
            else:
                q_arr = np.asarray(q, dtype=float).ravel()
                for i, v in enumerate(q_arr, 1):
                    meta[f"qu_{i}"] = float(v)

        if include_phase:
            meta.update(_pack_phases("phase_a", cm.phase_a))
            meta.update(_pack_phases("phase_b", cm.phase_b))

        header = "mwlab-cm " + " ".join(f"{k}={v}" for k, v in meta.items())

        np.savetxt(
            path,
            M_np,
            fmt=f"%.{precision}g",
            delimiter=delimiter,
            header=header,
            comments="# ",
        )
        return

    if fmt == "json":
        blob: dict[str, object] = {
            "layout": layout.name,
            "order": cm.topo.order,
            "ports": cm.topo.ports,
            "M": M_np.tolist(),
        }
        if include_qu and cm.qu is not None:
            q = cm.qu
            if np.isscalar(q):
                blob["qu"] = float(q)
            else:
                blob["qu_vec"] = list(map(float, np.asarray(q, dtype=float).ravel()))

        if include_phase:
            # phase_a
            if cm.phase_a is not None:
                if np.isscalar(cm.phase_a):
                    blob["phase_a"] = float(cm.phase_a)
                else:
                    blob["phase_a_vec"] = list(map(float, np.asarray(cm.phase_a, dtype=float).ravel()))
            # phase_b
            if cm.phase_b is not None:
                if np.isscalar(cm.phase_b):
                    blob["phase_b"] = float(cm.phase_b)
                else:
                    blob["phase_b_vec"] = list(map(float, np.asarray(cm.phase_b, dtype=float).ravel()))

        path.write_text(json.dumps(blob, indent=2))
        return

    raise ValueError("fmt должен быть 'ascii' или 'json'")

# -----------------------------------------------------------------------------
#                                    READ
# -----------------------------------------------------------------------------
def read_matrix(
    path: PathLike,
    *,
    topo=None,
    layout: MatrixLayout | str = "auto",
    delimiter: str = "\t",
    permutation: Sequence[int] | None = None,
):
    """
    Читает файл с **расширенной реальной матрицей связи** и возвращает `CouplingMatrix`.

    Поддерживаемые форматы файла:
      • ASCII (табличный) — первая(-ые) строка(-и) начинаются с '#', там же могут быть
        зашиты метаданные: `layout`, `order`, `ports`, `qu`, `qu_i`, `phase_a[_i]`, `phase_b[_i]`.
      • JSON — объект с полями: "layout", "order", "ports", "M" (список списков),
        а также опционально "qu"/"qu_vec", "phase_a"/"phase_a_vec", "phase_b"/"phase_b_vec".

    Параметры
    ----------
    path : str | Path
        Путь к файлу.
    topo : Topology | None
        Если **указана топология**, то чтение будет **строго ограничено** этой топологией:
        из матрицы берутся **только** элементы
          1) главной диагонали **резонаторной** подматрицы (M1_1..M_order_order),
          2) пары (i,j) из `topo.links` (верхний треугольник, 1-based),
        причём **считываются все значения, включая нулевые**.
        Порядок портов во входном файле учитывается через параметр `layout`.
        Если `topo` не задана — поведение остаётся прежним (реконструкция по всем
        ненулевым элементам верхнего треугольника через `CouplingMatrix.from_matrix`).
    layout : MatrixLayout | "auto" | str
        Макет портов во входном файле:
          • "auto" — попытаться определить (SL/TAIL) эвристикой или из метаданных;
          • явное значение `MatrixLayout` или имя Enum (строкой).
    delimiter : str
        Разделитель столбцов для ASCII.
    permutation : Sequence[int] | None
        Пользовательская перестановка для `layout=CUSTOM` (0-based, длина K, является
        перестановкой 0..K-1). Семантика та же, что и в `make_perm()`:
        `perm[i] = j` означает, что **i-я строка/столбец внешней матрицы** берётся
        из **j-й строки/столбца канонической (TAIL)**.

    Возвращает
    ----------
    CouplingMatrix
        Объект с матрицей связи и (при наличии) параметрами qu/phase_a/phase_b.

    Особенности поведения при явно переданной `topo`
    -----------------------------------------------
    1) Матрица из файла сначала приводится к каноническому порядку **TAIL**
       по указанному/детектированному `layout` (операция `M_can = P^T @ M_ext @ P`).
    2) Формируется словарь `mvals` **только** по:
         • резонаторной диагонали: (1,1)..(order,order);
         • звеньям из `topo.links` (i<j, 1-based; портовые связи также допустимы).
       В `mvals` записываются **все** значения (включая нули).
    3) Возвращается `CouplingMatrix(topo, mvals, qu, phase_a, phase_b)`
       **без** промежуточного вызова `CouplingMatrix.from_matrix()`, чтобы
       не отбрасывать нулевые значения и не добавлять «лишние» связи.

    Если `topo` не задана, используется прежняя логика:
    — чтение полной матрицы → `CouplingMatrix.from_matrix(...)` с возможной
      авто-догадкой топологии/макета.
    """
    path = Path(path)
    text = path.read_text()

    qu = None
    phase_a = None
    phase_b = None
    detected = None  # тип макета из файла (если присутствует явно)

    # ---------------- JSON ----------------
    if text.lstrip().startswith("{"):
        blob = json.loads(text)

        # сама матрица (внешний макет)
        M = np.asarray(blob["M"], dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("JSON: поле 'M' должно быть квадратной матрицей K×K")

        # макет из файла (если указан)
        if "layout" in blob:
            detected = MatrixLayout[str(blob["layout"]).upper()]

        # qu
        if "qu" in blob:
            qu = float(blob["qu"])
        elif "qu_vec" in blob:
            qu = np.asarray(blob["qu_vec"], dtype=float)

        # phase_a
        if "phase_a" in blob:
            phase_a = float(blob["phase_a"])
        elif "phase_a_vec" in blob:
            phase_a = np.asarray(blob["phase_a_vec"], dtype=float)

        # phase_b
        if "phase_b" in blob:
            phase_b = float(blob["phase_b"])
        elif "phase_b_vec" in blob:
            phase_b = np.asarray(blob["phase_b_vec"], dtype=float)

    # ---------------- ASCII ----------------
    else:
        # Собираем все строки-комментарии (начинаются с '#') в единый заголовок
        header_lines = [ln for ln in text.splitlines() if ln.startswith("#")]
        header = " ".join(h[1:].strip() for h in header_lines)

        # layout=...
        m = re.search(r"\blayout\s*=\s*(\w+)", header)
        if m:
            detected = MatrixLayout[m.group(1)]

        # qu (скаляр)
        m_qu = re.search(r"\bqu\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", header)
        if m_qu:
            qu = float(m_qu.group(1))
        else:
            # qu_i из заголовка
            qu_matches = re.findall(
                r"\bqu_(\d+)\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", header
            )
            if qu_matches:
                max_i = max(int(idx) for idx, _ in qu_matches)
                arr = np.zeros(max_i, dtype=float)
                for idx, val in qu_matches:
                    arr[int(idx) - 1] = float(val)
                qu = arr

        # phase_a
        m_pa = re.search(r"\bphase_a\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", header)
        if m_pa:
            phase_a = float(m_pa.group(1))
        else:
            pa_matches = re.findall(
                r"\bphase_a(\d+)\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
                header,
            )
            if pa_matches:
                max_i = max(int(idx) for idx, _ in pa_matches)
                arr = np.zeros(max_i, dtype=float)
                for idx, val in pa_matches:
                    arr[int(idx) - 1] = float(val)
                phase_a = arr

        # phase_b
        m_pb = re.search(r"\bphase_b\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", header)
        if m_pb:
            phase_b = float(m_pb.group(1))
        else:
            pb_matches = re.findall(
                r"\bphase_b(\d+)\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
                header,
            )
            if pb_matches:
                max_i = max(int(idx) for idx, _ in pb_matches)
                arr = np.zeros(max_i, dtype=float)
                for idx, val in pb_matches:
                    arr[int(idx) - 1] = float(val)
                phase_b = arr

        # численные данные матрицы (внешний макет)
        M = np.loadtxt(path, delimiter=delimiter)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("ASCII: считанная матрица должна быть квадратной K×K")
        M = np.asarray(M, dtype=float)

    # --------------- определяем макет (если не задан явно) ---------------
    if layout == "auto":
        if detected is not None:
            lay = detected
        else:
            lay = _guess_layout(M) or MatrixLayout.TAIL
    else:
        lay = MatrixLayout[layout] if isinstance(layout, str) else layout

    # --------------- ОСНОВНАЯ ВЕТКА: topo задан → «считываем строго по топологии» ---------------
    if topo is not None:
        K = M.shape[0]
        if topo.size != K:
            raise ValueError(f"Размер матрицы ({K}) не совпадает с topo.size ({topo.size})")

        # Приводим внешнюю матрицу к каноническому порядку TAIL.
        # Семантика перестановки соответствует make_perm():
        #   M_ext = P @ M_can @ P.T  →  M_can = P.T @ M_ext @ P
        if lay is MatrixLayout.TAIL:
            perm = list(range(K))
        elif lay is MatrixLayout.SL:
            if topo.ports != 2:
                raise ValueError("layout='SL' применим только к 2-портовым устройствам (ports == 2)")
            perm = make_perm(topo.order, topo.ports, MatrixLayout.SL)
        elif lay is MatrixLayout.CUSTOM:
            if permutation is None:
                raise ValueError("CUSTOM layout требует параметр 'permutation'")
            perm = list(permutation)
            if len(perm) != K or sorted(perm) != list(range(K)):
                raise ValueError("permutation должна быть перестановкой 0…K-1")
        else:
            # Защита от будущих значений Enum
            raise ValueError(f"Неизвестный layout: {lay}")

        P = np.eye(K)[perm]
        M_can = P.T @ M @ P  # теперь M_can в порядке [R1..Rn, P1..Pp]

        # Формируем mvals ТОЛЬКО по диагонали резонаторов и topo.links.
        # ВАЖНО: записываем даже нулевые значения (не используем порог отсечки).
        mvals: dict[str, float] = {}

        # (1) Диагональ резонаторов: индексы 1..order (1-based).
        for i in range(1, topo.order + 1):
            mvals[f"M{i}_{i}"] = float(M_can[i - 1, i - 1])

        # (2) Связи из topo.links (верхний треугольник, 1-based; портовые тоже возможны).
        # Гарантируем, что ключи идут как (min,max).
        for (i, j) in topo.links:
            if j < i:
                i, j = j, i
            mvals[f"M{i}_{j}"] = float(M_can[i - 1, j - 1])

        # Собираем итоговый объект строго под заданную топологию.
        return CouplingMatrix(
            topo,
            mvals,
            qu=qu,
            phase_a=phase_a,
            phase_b=phase_b,
        )

    # --------------- ИНАЧЕ: прежнее поведение (полная реконструкция) ---------------
    # Без заданной topo передаём всё в CouplingMatrix.from_matrix:
    #   • он при необходимости усреднит матрицу с её транспонированной (force_sym),
    #   • отфильтрует малые значения по atol,
    #   • восстановит или примет переданную топологию и т.д.
    return CouplingMatrix.from_matrix(
        M,
        topo=topo,
        layout=lay,
        qu=qu,
        phase_a=phase_a,
        phase_b=phase_b,
        permutation=permutation,
    )



# -----------------------------------------------------------------------------
#                     Простая эвристика определения макета
# -----------------------------------------------------------------------------
def _guess_layout(M, *, rtol: float = 1e-6):
    """Пытается отличить **TAIL** от **SL** (только p=2) по структуре матрицы.

    Предположения:
    - диагональ портов ≈ 0;
    - связь порт-порт ≈ 0;
    - есть хотя бы одна порт-резонаторная связь.
    Если определённо сказать нельзя — возвращаем None.
    """
    # --- привести к единым операциям ---
    is_torch = torch.is_tensor(M)
    if not is_torch:
        M_np = np.asarray(M, dtype=float)
        K = M_np.shape[0]
        absM = np.abs(M_np)
        max_mag = absM.max() if M_np.size else 0.0
        def get(i, j): return M_np[i, j]
    else:
        K = M.shape[0]
        absM = torch.abs(M)
        max_mag = absM.max().item() if M.numel() else 0.0
        def get(i, j): return M[i, j].item()

    if K < 3 or max_mag == 0.0:
        return None

    tol = max_mag * rtol if max_mag > 0 else 1e-12

    # ---------- признак TAIL ----------
    # нижние строки (конец матрицы) – «портовые»: диагональ≈0 и правее диагонали≈0
    tail_ports = 0
    for i in range(K - 1, -1, -1):
        if abs(get(i, i)) <= tol and all(abs(get(i, j)) <= tol for j in range(i, K)):
            tail_ports += 1
        else:
            break
    if tail_ports >= 2:
        return MatrixLayout.TAIL

    # ---------- признак SL (2-порт) ----------
    if K >= 4 and abs(get(0, 0)) <= tol and abs(get(K - 1, K - 1)) <= tol:
        if abs(get(0, K - 1)) <= tol:  # нет порт-порт связи
            # связи порт-резонатор
            s_links = any(abs(get(0, j)) > tol for j in range(1, K - 1))
            l_links = any(abs(get(K - 1, j)) > tol for j in range(1, K - 1))
            if s_links and l_links:
                return MatrixLayout.SL

    return None


__all__ = [
    "write_matrix",
    "read_matrix",
    "MatrixLayout",
]

