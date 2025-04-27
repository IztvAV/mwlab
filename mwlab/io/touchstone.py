"""
mwlab.io.touchstone
-------------------
Универсальный контейнер TouchstoneData (play‑ground‑версия).

* TouchstoneData.load(path)             – чтение *.sNp
* TouchstoneData(network, params, ...)  – создание «с нуля»
* .save(path)                           – запись с корректным заголовком
"""

import pathlib
import re
from typing import Dict, Optional, Union

import skrf as rf

_PARAM_RE = re.compile(r"Parameters\s*=\s*\{([^}]*)\}", re.IGNORECASE)


class TouchstoneData:
    """Контейнер S‑матрицы (`skrf.Network`) + словаря параметров."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        network: rf.Network,
        params: Optional[Dict[str, float]] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
    ):
        # scikit‑rf иногда даёт comments=None
        if network.comments is None:
            network.comments = []

        parsed = self._params_from_comments(network.comments)
        if params:
            parsed.update(params)

        self.network = network
        self.params: Dict[str, float] = parsed
        self.path = pathlib.Path(path) if path else None

        if self.network.number_of_ports > 9:
            raise ValueError("> 9 портов не поддерживается.")

    # ------------------------------------------------------------- Uploading file
    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "TouchstoneData":
        path = pathlib.Path(path)
        net = rf.Network(str(path))
        obj = cls(net, path=path)

        # запасной парсинг, если comments пуст
        if not obj.params:
            obj.params = obj._params_from_file(path)

        return obj

    # ------------------------------------------------------------------ save
    def save(self, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        """
        Записывает данные в Touchstone‑файл *path*.
        Если *path* не указан, используется self.path.
        """
        target = pathlib.Path(path) if path else self.path
        if target is None:
            raise ValueError("Не указан путь сохранения.")

        target.parent.mkdir(parents=True, exist_ok=True)

        # 1) штатная запись S‑параметров
        stem = target.with_suffix("").name
        n_ports = int(self.network.number_of_ports)
        self.network.write_touchstone(filename=stem, dir=str(target.parent))

        real_file = target.parent / f"{stem}.s{n_ports}p"

        # 2) формируем единственную строку Parameters
        p_line = (
            "! Parameters = {"
            + "; ".join(f"{k}={v}" for k, v in self.params.items())
            + "}\n"
        )

        with real_file.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        # 3) удаляем все старые строки Parameters и вставляем новую перед '#'
        lines = [
            l for l in lines if not (l.lstrip().startswith("!") and "Parameters" in l)
        ]
        insert_at = next(
            (i for i, l in enumerate(lines) if l.lstrip().startswith("#")), len(lines)
        )
        lines.insert(insert_at, p_line)

        with real_file.open("w", encoding="utf-8") as fh:
            fh.writelines(lines)

    # ------------------------------------------------------ helper:  parsing
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
