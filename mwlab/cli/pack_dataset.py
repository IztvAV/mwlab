#mwlab/cli/pack_dataset.py
"""
CLI-скрипт: упаковка каталога с .sNp → HDF5.

Пример запуска:
    python -m mwlab.cli.pack_dataset \\
        --src raw_data/ --dst train.h5 --pattern "*.s2p"
"""

from __future__ import annotations

import argparse
import pathlib
from tqdm import tqdm

from mwlab.io.backends import FileBackend, HDF5Backend
from mwlab.datasets.touchstone_dataset import TouchstoneDataset


def parse_args():
    p = argparse.ArgumentParser(description="Упаковка Touchstone-файлов в HDF5")
    p.add_argument("--src", required=True, help="Каталог с *.sNp")
    p.add_argument("--dst", required=True, help="Файл назначения .h5")
    p.add_argument("--pattern", default="*.s?p", help="Глоб-шаблон (rglob)")
    return p.parse_args()


def main():
    args = parse_args()
    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)

    file_backend = FileBackend(src, args.pattern)
    ds = TouchstoneDataset(file_backend)  # без трансформов – сырые данные

    with HDF5Backend(dst, mode="w") as h5b:
        for ts in tqdm((ds.backend.read(i) for i in range(len(ds))), total=len(ds)):
            h5b.append(ts)


if __name__ == "__main__":
    main()
