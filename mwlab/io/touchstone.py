# mwlab/io/touchstone.py
"""
mwlab.io.touchstone
-------------------
Базовый контейнер TouchstoneData.

Основные возможности:

* TouchstoneData.load(path)      – чтение *.sNp
* TouchstoneData(network, ...)   – создание «с нуля»
* .save(path)                    – запись в Touchstone-файл

НОВОЕ в версии 0.1.1:
* .to_numpy() / .from_numpy()    – сериализация S-матрицы в NumPy-словарь
  для хранения в бинарных backend-ах (HDF5, LMDB, Zarr и т.п.).
"""

from __future__ import annotations

import pathlib
import re
from typing import Dict, Optional, Union

import numpy as np
import skrf as rf

# ------------------------------ вспомогательный шаблон "Parameters = {k=v; ...}"
_PARAM_RE = re.compile(r"Parameters\s*=\s*\{([^}]*)\}", re.IGNORECASE)


class TouchstoneData:
    """
    Контейнер:
        • self.network : skrf.Network      – S-параметры + частотная сетка
        • self.params  : dict[str, float]  – произвольные скалярные параметры
    """

    # ---------------------------------------------------------------- init
    def __init__(
        self,
        network: rf.Network,
        params: Optional[Dict[str, float]] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
    ):
        # scikit-rf иногда отдает comments=None – приводим к списку
        if network.comments is None:
            network.comments = []

        # --- собираем параметры из комментария + явные **params
        parsed = self._params_from_comments(network.comments)
        if params:
            parsed.update(params)

        self.network = network
        self.params: Dict[str, float] = parsed
        self.path = pathlib.Path(path) if path else None

        # Ограничение: scikit-rf не поддерживает n_ports > 9 при записи
        if self.network.number_of_ports > 9:
            raise ValueError("> 9 портов не поддерживается в данной реализации.")

    # -------------------------------------------------------- Загрузка *.sNp
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

    # ------------------------------------------------------------- Сохранение
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
        stem = target.with_suffix("").name
        n_ports = int(self.network.number_of_ports)
        self.network.write_touchstone(filename=stem, dir=str(target.parent))

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

    # -------------------------------------------------- СЕРИАЛИЗАЦИЯ БИНАРНАЯ
    def to_numpy(self) -> dict[str, np.ndarray]:
        """
        Возвращает словарь ndarray-ов (готов к записи в HDF5/LMDB).

        Формат:
            {
              "s":  complex64  (freq, n_ports, n_ports)
              "f":  float64    (freq,)     – частоты в Гц
              "params_k": float32|str      – по одному ключу на атрибут
            }
        """
        out: dict[str, np.ndarray] = {}
        out["s"] = self.network.s.astype(np.complex64)
        out["f"] = self.network.f.astype(np.float64)

        # Параметры: отделяем числовые и строковые, т.к. HDF5 плохо пишет mixed-dtypes.
        for k, v in self.params.items():
            if isinstance(v, (int, float)):
                out[f"param/{k}"] = np.array(v, dtype=np.float32)
            else:
                out[f"param/{k}"] = np.frombuffer(
                    str(v).encode("utf-8"), dtype="uint8"
                )

        return out

    @classmethod
    def from_numpy(cls, dct: dict[str, np.ndarray]) -> "TouchstoneData":
        """
        Обратная операция: словарь ndarray-ов → TouchstoneData.
        """
        s = dct["s"].astype(np.complex64)
        f = dct["f"].astype(np.float64)
        freq = rf.Frequency.from_f(f, unit="Hz")
        net = rf.Network(frequency=freq, s=s)

        params: Dict[str, float] = {}
        for k, v in dct.items():
            if not k.startswith("param/"):
                continue
            name = k.split("/", 1)[1]
            if v.dtype == "uint8":  # значит строка
                params[name] = bytes(v).decode("utf-8")
            else:
                params[name] = float(v)

        return cls(net, params=params)

    # ------------------------------------------------------ helpers: парсинг
    @staticmethod
    def _split_params(raw: str) -> Dict[str, float]:
        out = {}
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
    def _params_from_comments(cls, comments) -> Dict[str, float]:
        if not comments:
            return {}
        joined = " ".join(c.strip() for c in comments)
        m = _PARAM_RE.search(joined)
        return cls._split_params(m.group(1)) if m else {}

    @staticmethod
    def _params_from_file(path: pathlib.Path) -> Dict[str, float]:
        try:
            with path.open("r", errors="ignore") as fh:
                for _ in range(100):  # читаем первые 100 строк
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

