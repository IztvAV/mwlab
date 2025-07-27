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
        return {f"{prefix}_{i}": float(v) for i, v in enumerate(arr, 1)}

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
    Читает файл с расширенной матрицей связи и возвращает CouplingMatrix.

    Параметры
    ----------
    path : str | Path
        Путь к ASCII/JSON файлу.
    topo : Topology | None
        Готовая топология, если известна. Иначе будет восстановлена в from_matrix().
    layout : MatrixLayout | \"auto\" | str
        Макет во входном файле:
          * \"auto\" — попытаться определить (SL/TAIL) эвристикой;
          * явное значение Enum или строковое имя.
    delimiter : str
        Разделитель для ASCII.

    Возвращает
    ----------
    CouplingMatrix
        Объект с матрицей связи и (при наличии) параметрами qu/phase_a/phase_b.
    """
    path = Path(path)
    text = path.read_text()

    qu = None
    phase_a = None
    phase_b = None

    # ---------------- JSON ----------------
    if text.lstrip().startswith("{"):
        blob = json.loads(text)
        M = np.asarray(blob["M"], dtype=float)
        detected = MatrixLayout[blob.get("layout", "TAIL")]

        # qu
        if "qu" in blob:
            qu = float(blob["qu"])
        elif "qu_vec" in blob:
            qu = np.asarray(blob["qu_vec"], dtype=float)

        # phases
        if "phase_a" in blob:
            phase_a = float(blob["phase_a"])
        elif "phase_a_vec" in blob:
            phase_a = np.asarray(blob["phase_a_vec"], dtype=float)

        if "phase_b" in blob:
            phase_b = float(blob["phase_b"])
        elif "phase_b_vec" in blob:
            phase_b = np.asarray(blob["phase_b_vec"], dtype=float)

    # ---------------- ASCII ----------------
    else:
        import re

        header_lines = [ln for ln in text.splitlines() if ln.startswith("#")]
        header = " ".join(h[1:].strip() for h in header_lines)

        # layout
        m = re.search(r"layout\s*=\s*(\w+)", header)
        detected = MatrixLayout[m.group(1)] if m else None

        # qu scalar
        m_qu = re.search(r"\bqu\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header)
        if m_qu:
            qu = float(m_qu.group(1))
        else:
            # qu_i
            qu_matches = re.findall(
                r"qu_(\d+)\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header
            )
            if qu_matches:
                max_i = max(int(idx) for idx, _ in qu_matches)
                arr = np.zeros(max_i, dtype=float)
                for idx, val in qu_matches:
                    arr[int(idx) - 1] = float(val)
                qu = arr

        # phase_a
        m_pa = re.search(r"\bphase_a\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header)
        if m_pa:
            phase_a = float(m_pa.group(1))
        else:
            pa_matches = re.findall(
                r"phase_a_(\d+)\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
                header,
            )
            if pa_matches:
                max_i = max(int(idx) for idx, _ in pa_matches)
                arr = np.zeros(max_i, dtype=float)
                for idx, val in pa_matches:
                    arr[int(idx) - 1] = float(val)
                phase_a = arr

        # phase_b
        m_pb = re.search(r"\bphase_b\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", header)
        if m_pb:
            phase_b = float(m_pb.group(1))
        else:
            pb_matches = re.findall(
                r"phase_b_(\d+)\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
                header,
            )
            if pb_matches:
                max_i = max(int(idx) for idx, _ in pb_matches)
                arr = np.zeros(max_i, dtype=float)
                for idx, val in pb_matches:
                    arr[int(idx) - 1] = float(val)
                phase_b = arr

        # сама матрица
        M = np.loadtxt(path, delimiter=delimiter)

    # --------------- определяем макет ---------------
    if layout == "auto":
        if detected is not None:
            lay = detected
        else:
            lay = _guess_layout(M) or MatrixLayout.TAIL
    else:
        lay = MatrixLayout[layout] if isinstance(layout, str) else layout

    # --------------- финальный объект ---------------
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

