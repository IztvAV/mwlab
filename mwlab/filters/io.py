#mwlab/filters/io.py
"""
MWLab · filters · io
====================
Модуль обеспечивает **экспорт и импорт расширенных матриц связи** (coupling‑matrix)
в автономные файлы.  Это позволяет обмениваться матрицами с внешними
пакетами — такими как **CST Filter Designer**, **SynMatrix**, MATLAB‑скрипты, —
без участия Touchstone‑файлов.

Основная идея
-------------
1. **Внутреннее представление mwlab** остаётся неизменным: порты идут
   в конце («в хвосте-tail») после всех резонаторов ``(R₁ … Rₙ, P₁ … Pᴘ)``.
2. При записи/чтении мы умеем **переставлять строки/столбцы** матрицы
   в другой порядок (enum :class:`MatrixLayout`).  Самые частые случаи:
   * ``TAIL`` – «хвост‑порты» (canonical, mwlab);
   * ``SL``   – классическая для фильтров форма
     ``(S, R₁ … Rₙ, L)`` (Source/Load в начале и в конце).
3. Для нестандартных порядков пользователь может передать **явную
   перестановку** (`permutation`).

Форматы файлов
--------------
* ``fmt="ascii"`` – столбцы/строки через `delimiter` (\t по умолчанию).
  Первая строка — комментарий «# mwlab-cm layout=SL order=4 ports=2».
* ``fmt="json"``  – словарь с полями ``layout``, ``order``, ``ports``, ``M``.

Дополнительная функция `_guess_layout` — простая эвристика, которая
определяет SL/TAIL для двухпортовых матриц автоматом.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union
import warnings

import numpy as np

from mwlab.filters.cm import CouplingMatrix, MatrixLayout

# -----------------------------------------------------------------------------
#                               helpers: I/O
# -----------------------------------------------------------------------------

PathLike = Union[str, Path]

# ──────────────────────────────────────────────────────────────────────────────
#                               WRITE / EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def write_matrix(
    cm: CouplingMatrix,
    path: PathLike,
    *,
    layout: MatrixLayout = MatrixLayout.SL,
    fmt: str = "ascii",
    precision: int = 15,               # максимум для float32
    delimiter: str = "\t",
) -> None:
    """Сохраняет матрицу связи на диск.

    Parameters
    ----------
    cm : CouplingMatrix
        Объект, содержащий внутреннюю (tail‑ports) матрицу.
    path : str | Path
        Куда писать файл.
    layout : MatrixLayout, default SL
        В каком порядке портов формировать вывод.
    fmt : {"ascii", "json"}
        Формат файла.
    precision : int, default 15 (15 значащих цифр → float32 воспроизводится побитно-точно)
        Кол-во значащих цифр для ASCII‑вывода.
    delimiter : str, default "\t"
        Разделитель столбцов для ASCII.
    """

    path = Path(path)
    M = cm.to_matrix(layout)

    if fmt == "ascii":
        header = (
            f"# mwlab-cm layout={layout.name} "
            f"order={cm.topo.order} ports={cm.topo.ports}"
        )
        np.savetxt(
            path,
            M,
            fmt=f"%.{precision}g",
            delimiter=delimiter,
            header=header,
        )
        return

    if fmt == "json":
        blob = {
            "layout": layout.name,
            "order": cm.topo.order,
            "ports": cm.topo.ports,
            "M": M.tolist(),
        }
        path.write_text(json.dumps(blob, indent=2))
        return

    raise ValueError("fmt должен быть 'ascii' или 'json'")


# ──────────────────────────────────────────────────────────────────────────────
#                               READ / IMPORT
# ──────────────────────────────────────────────────────────────────────────────

def read_matrix(
    path: PathLike,
    *,
    topo=None,
    layout: MatrixLayout | str = "auto",
    delimiter: str = "\t",
):
    """Читает файл с матрицей связи и возвращает :class:`CouplingMatrix`.

    *Если* ``layout='auto'`` — «пытается определить SL/TAIL для любого p;
    если неопределимо, берётся TAIL».
    """

    path = Path(path)
    text = path.read_text()

    # ----------- JSON -------------------------------------------------------
    if text.lstrip().startswith("{"):
        blob = json.loads(text)
        M = np.asarray(blob["M"], dtype=float)
        detected = MatrixLayout[blob.get("layout", "TAIL")]

    # ----------- ASCII ------------------------------------------------------
    else:
        import re

        header_lines = [ln for ln in text.splitlines() if ln.startswith("#")]
        header = " ".join(header_lines)
        m = re.search(r"layout\s*=\s*(\w+)", header)
        detected = MatrixLayout[m.group(1)] if m else None
        M = np.loadtxt(path, delimiter=delimiter)

    # ---------------------------------------------------------------- layout
    if layout == "auto":
        lay = detected or _guess_layout(M) or MatrixLayout.TAIL
    else:
        lay = MatrixLayout[layout] if isinstance(layout, str) else layout

    return CouplingMatrix.from_matrix(M, topo=topo, layout=lay)


# ──────────────────────────────────────────────────────────────────────────────
#                        layout heuristics (2‑port SL vs TAIL)
# ──────────────────────────────────────────────────────────────────────────────
def _guess_layout(M: np.ndarray) -> MatrixLayout | None:
    """
    Определяет раскладку матрицы связи:
      - TAIL для многопортовых устройств (последние P индексы — порты);
      - SL для классического 2-портового Source–Load;
      - None если не удалось определить однозначно.
    """
    K = M.shape[0]
    if K < 3:
        return None

    if np.allclose(M, 0.0, atol=1e-12):
        return None  # вся матрица нулевая → неопределимо

    # 1) TAIL: считаем количество подряд идущих строк «в хвосте»,
    #    у которых все элементы от диагонали вправо нули.
    P = 0
    for i in range(K - 1, -1, -1):
        if np.allclose(M[i, i:], 0.0, atol=1e-12):
            P += 1
        else:
            break
    if P >= 2:
        return MatrixLayout.TAIL

    # 2) SL (2-портовый): [S, R1…Rn, L]
    #    проверяем наличие S–R1 и Rn–L связей без прямой S–L.
    #    Индексы: S=0, R1=1, Rn=K-2, L=K-1.
    if (not np.isclose(M[0, 1],   0.0, atol=1e-12)  # S–R1
        and not np.isclose(M[K-2, K-1], 0.0, atol=1e-12)  # Rn–L
        and  np.isclose(M[0, K-1], 0.0, atol=1e-12)  # нет S–L
    ):
        return MatrixLayout.SL

    # не удалось однозначно распознать
    return None