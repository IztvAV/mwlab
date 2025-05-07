# mwlab/io/touchstone.py
"""
mwlab.io.touchstone
-------------------
Базовый контейнер TouchstoneData + (де)сериализация в NumPy‑словарь.

Основные возможности:

* TouchstoneData.load(path)      – чтение *.sNp
* TouchstoneData(network, ...)   – создание «с нуля»
* .save(path)                    – запись в Touchstone-файл
* .to_numpy() / .from_numpy()    – сериализация S-матрицы в NumPy-словарь
  для хранения в бинарных backend-ах (HDF5, LMDB, Zarr и т.п.).
"""
from __future__ import annotations

import pathlib
import re
from typing import Dict, Optional, Union

import numpy as np
import skrf as rf

# ------------------------------ шаблон "Parameters = {k=v; ...}"
_PARAM_RE = re.compile(r"Parameters\s*=\s*\{([^}]*)\}", re.IGNORECASE)


class TouchstoneData:
    """
    Контейнер:
        • self.network : skrf.Network      – S‑матрица + частоты
        • self.params  : dict[str, float|str]  – пользовательские метаданные
    """

    # ---------------------------------------------------------- init
    def __init__(
        self,
        network: rf.Network,
        params: Optional[Dict[str, float | str]] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
    ):
        # scikit‑rf иногда отдает comments=None
        network.comments = list(network.comments or [])

        # --- собираем параметры из network.comments
        parsed = self._params_from_comments(network.comments)
        if params:
            parsed.update(params)

        self.network = network
        self.params: Dict[str, float | str] = parsed
        self.path = pathlib.Path(path) if path else None

        # Ограничения
        if self.network.number_of_ports > 9:
            raise ValueError("> 9 портов не поддерживается.")

    # ------------------------------------------------------- Загрузка *.sNp
    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "TouchstoneData":
        """
        Универсальный конструктор из Touchstone-файла.

        • Читаем файл scikit-rf'ом.
        • Пытаемся вытащить строку «! Parameters = {...}».
        """
        path = pathlib.Path(path)
        net = rf.Network(str(path))
        obj = cls(net, path=path)

        # запасной парсинг, если comments пусты
        if not obj.params:
            obj.params = obj._params_from_file(path)
        return obj

    # ------------------------------------------------------- Сохранение
    def save(self, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        """
        Записывает данные в Touchstone-файл.

        • Если *path* не указан – используем self.path.
        • Записываем S-матрицу штатным методом scikit-rf.
        • Строку «! Parameters = {...}» вставляем сами, чтобы
          сохранить словарь self.params.
        """
        target = pathlib.Path(path) if path else self.path
        if target is None:
            raise ValueError("Не указан путь сохранения.")

        target.parent.mkdir(parents=True, exist_ok=True)

        # 1) штатная запись S-параметров
        n_ports = int(self.network.number_of_ports)
        self.network.write_touchstone(filename=target.name, dir=str(target.parent))

        stem = target.with_suffix("").name
        real_file = target.parent / f"{stem}.s{n_ports}p"

        # 2) формируем строку Parameters
        p_line = (
            "! Parameters = {"
            + "; ".join(f"{k}={v}" for k, v in self.params.items())
            + "}\n"
        )

        # 3) читаем файл → убираем старые Parameters → вставляем новую
        with real_file.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        lines = [
            l for l in lines if not (l.lstrip().startswith("!") and "Parameters" in l)
        ]
        insert_at = next(
            (i for i, l in enumerate(lines) if l.lstrip().startswith("#")), len(lines)
        )
        lines.insert(insert_at, p_line)

        with real_file.open("w", encoding="utf-8") as fh:
            fh.writelines(lines)

    # --------------------------------------------- СЕРИАЛИЗАЦИЯ → NumPy
    def to_numpy(self) -> dict[str, np.ndarray]:
        """
        Возвращает словарь ndarray-ов (готов к записи в HDF5/LMDB).

        Ключи:
            's'                 – (F,P,P) complex64
            'f'                 – (F,)    float64   (в Гц)
            'meta/unit'         – ()      S ascii
            'meta/s_def'        – ()      S ascii
            'meta/z0'           – (P,)    complex64
            'meta/comments'     – (N,)    uint8  (если есть)
            'param/...'         – пользовательские ключи
        """
        out: dict[str, np.ndarray] = {}
        out["s"] = self.network.s.astype(np.complex64)
        # ---- частотная сетка -------------------------------------------
        unit = self.network.frequency.unit or "Hz"
        # f_scaled – массив уже в единице `unit` (если unit='GHz', числа в ГГц)
        out["f"] = self.network.frequency.f_scaled.astype(np.float64)

        # ---- системные метаданные ---------------------------------------
        out["meta/unit"] = np.bytes_(unit)
        out["meta/s_def"] = np.bytes_(self.network.s_def)
        out["meta/z0"] = self.network.z0.astype(np.complex64)
        if self.network.comments:
            clean = [c.rstrip("\n") for c in self.network.comments]
            out["meta/comments"] = np.frombuffer(
                "\n".join(clean).encode(), dtype="uint8"
            )

        # ---- пользовательские параметры ---------------------------------
        for k, v in self.params.items():
            if isinstance(v, int):
                out[f"param/{k}"] = np.array(v, dtype=np.int64)
            elif isinstance(v, float):
                out[f"param/{k}"] = np.array(v, dtype=np.float32)
            else:
                out[f"param/{k}"] = np.frombuffer(str(v).encode(), dtype="uint8")
        return out

    # --------------------------------------------- NumPy → TouchstoneData
    @classmethod
    def from_numpy(cls, dct: dict[str, np.ndarray]) -> "TouchstoneData":
        """
        Обратная операция: словарь ndarray-ов → TouchstoneData.
        """
        # --- обязательные поля ------------------------------------------
        s = dct["s"].astype(np.complex64)
        f = dct["f"].astype(np.float64)

        unit = bytes(dct["meta/unit"]).decode()
        freq = rf.Frequency.from_f(f, unit=unit)

        net = rf.Network(frequency=freq, s=s)
        net.s_def = bytes(dct["meta/s_def"]).decode()
        net.z0[:] = dct["meta/z0"].astype(np.complex64)

        if "meta/comments" in dct:
            net.comments = bytes(dct["meta/comments"]).decode().split("\n")

        # --- параметры пользователя -------------------------------------
        params: Dict[str, float | str] = {}
        for k, v in dct.items():
            if not k.startswith("param/"):
                continue
            name = k.split("/", 1)[1]
            params[name] = (
                bytes(v).decode() if v.dtype == "uint8" else v.item()
            )
        return cls(net, params)

    # --------------------------------------------------- helpers: парсинг
    @staticmethod
    def _split_params(raw: str) -> Dict[str, float | str]:
        out: Dict[str, float | str] = {}
        for item in raw.split(";"):
            if "=" not in item:
                continue
            k, v = (s.strip() for s in item.split("=", 1))
            if not k:
                continue
            try:
                out[k] = float(v.replace(",", "."))
            except ValueError:
                out[k] = v
        return out

    @classmethod
    def _params_from_comments(cls, comments) -> Dict[str, float | str]:
        if not comments:
            return {}
        m = _PARAM_RE.search(" ".join(c.strip() for c in comments))
        return cls._split_params(m.group(1)) if m else {}

    @staticmethod
    def _params_from_file(path: pathlib.Path) -> Dict[str, float | str]:
        try:
            with path.open("r", errors="ignore") as fh:
                for _ in range(100):
                    line = fh.readline()
                    if not line or line.lstrip().startswith("#"):
                        break
                    if "Parameters" in line:
                        m = _PARAM_RE.search(line)
                        if m:
                            return TouchstoneData._split_params(m.group(1))
        except Exception:
            pass
        return {}

    def __repr__(self) -> str:
        n_ports = self.network.number_of_ports
        n_freq  = len(self.network.frequency)
        keys    = ", ".join(self.params) or "—"
        return (
            f"<TouchstoneData {n_ports}‑port · {n_freq}pts · "
            f"params=[{keys}] · unit={self.network.frequency.unit}>"
        )