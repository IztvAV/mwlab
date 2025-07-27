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
from typing import Union, Optional
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
    precision: int = 9,
    delimiter: str = "\t",
    device: str | torch.device = DEFAULT_DEVICE,
) -> None:
    """Сохраняет полную реальную матрицу связи на диск.

    Parameters
    ----------
    cm : CouplingMatrix
        Объект, содержащий топологию и значения M.
    path : str | Path
        Куда писать файл.
    layout : MatrixLayout, default SL
        В каком порядке располагать порты во внешней матрице.
    fmt : {"ascii", "json"}
        Формат файла.
    precision : int, default 9
        Кол-во значащих цифр для ASCII-вывода (9 достаточно для float32).
    delimiter : str, default "\t"
        Разделитель столбцов для ASCII.
    device : torch.device | str
        Устройство, на которое будет собрана матрица (по факту влияет только
        на промежуточный тензор). Для записи всё равно переводится в NumPy.
    """
    path = Path(path)

    # Строим внешнюю матрицу
    M = cm.to_matrix(layout=layout, device=device)
    M_np = M.cpu().numpy()

    if fmt == "ascii":
        header = (
            f"# mwlab-cm layout={layout.name} order={cm.topo.order} ports={cm.topo.ports}"
        )
        np.savetxt(
            path,
            M_np,
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
            "M": M_np.tolist(),
        }
        path.write_text(json.dumps(blob, indent=2))
        return

    raise ValueError("fmt должен быть 'ascii' или 'json'")


# -----------------------------------------------------------------------------
#                                    READ
# -----------------------------------------------------------------------------

def read_matrix(
    path: PathLike,
    *,
    topo: Optional[object] = None,
    layout: MatrixLayout | str = "auto",
    delimiter: str = "\t",
    device: str | torch.device = DEFAULT_DEVICE,
) -> CouplingMatrix:
    """Читает файл с матрицей связи и возвращает :class:`CouplingMatrix`.

    Если ``layout='auto'`` — сначала ищем подсказку в заголовке/JSON, затем
    применяем эвристику :func:`_guess_layout` (только для p=2). Если определить
    невозможно — принимаем **TAIL**.
    """
    path = Path(path)
    text = path.read_text()

    # ----------------------------- JSON -----------------------------------
    if text.lstrip().startswith("{"):
        blob = json.loads(text)
        M = torch.as_tensor(blob["M"], dtype=DT_R, device=device)
        detected = blob.get("layout", "TAIL")
        try:
            detected_layout = MatrixLayout[detected]
        except KeyError as exc:
            raise ValueError(f"Неизвестный layout в JSON: {detected!r}") from exc
        order = int(blob.get("order", 0))
        ports = int(blob.get("ports", 0))
    # ----------------------------- ASCII ----------------------------------
    else:
        # отделяем заголовок(и) c '#'
        header_lines = [ln for ln in text.splitlines() if ln.strip().startswith("#")]
        header = " ".join(header_lines)
        m = re.search(r"layout\s*=\s*(\w+)", header)
        detected_layout = MatrixLayout[m.group(1)] if m else None
        mo = re.search(r"order\s*=\s*(\d+)", header)
        mp = re.search(r"ports\s*=\s*(\d+)", header)
        order = int(mo.group(1)) if mo else 0
        ports = int(mp.group(1)) if mp else 0

        M_np = np.loadtxt(path, delimiter=delimiter)
        M = torch.as_tensor(M_np, dtype=DT_R, device=device)

    # ------------------------- layout resolve ------------------------------
    if layout == "auto":
        if detected_layout is not None:
            lay = detected_layout
        else:
            lay = _guess_layout(M, order_hint=order, ports_hint=ports) or MatrixLayout.TAIL
    else:
        lay = MatrixLayout[layout] if isinstance(layout, str) else layout

    return CouplingMatrix.from_matrix(M, topo=topo, layout=lay, device=device)


# -----------------------------------------------------------------------------
#                     Простая эвристика определения макета
# -----------------------------------------------------------------------------

def _guess_layout(
    M: torch.Tensor,
    *,
    order_hint: int = 0,
    ports_hint: int = 0,
    rtol: float = 1e-6,
) -> Optional[MatrixLayout]:
    """Пытается отличить **TAIL** от **SL** (только p=2) по структуре матрицы.

    Предположения:
    - диагональ портов ≈ 0;
    - связь порт-порт ≈ 0;
    - есть хотя бы одна порт-резонаторная связь.
    Если определённо сказать нельзя — возвращаем None.
    """
    K = M.shape[0]
    if K < 3:
        return None

    # адаптивный порог
    max_mag = torch.max(torch.abs(M)).item() if M.numel() else 0.0
    tol = max_mag * rtol if max_mag > 0 else 1e-12

    # --- попытка TAIL: подряд идущие портовые строки снизу
    tail_ports = 0
    for i in range(K - 1, -1, -1):
        diag_ok = abs(M[i, i].item()) <= tol
        right_zero = torch.all(torch.abs(M[i, i:]) <= tol).item()
        if diag_ok and right_zero:
            tail_ports += 1
        else:
            break
    if tail_ports >= 2:
        return MatrixLayout.TAIL

    # --- попытка SL (только 2 порта)
    if K >= 4:
        d0 = abs(M[0, 0].item()) <= tol
        dN = abs(M[K - 1, K - 1].item()) <= tol
        if d0 and dN:
            if abs(M[0, K - 1].item()) <= tol:  # порт-порт связь ≈ 0
                s_links = torch.abs(M[0, 1:K - 1]) > tol
                l_links = torch.abs(M[K - 1, 1:K - 1]) > tol
                if s_links.any() and l_links.any():
                    return MatrixLayout.SL

    return None


__all__ = [
    "write_matrix",
    "read_matrix",
    "MatrixLayout",
]

