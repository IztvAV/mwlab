#tests/test_cli_pack_dataset.py
"""
Тест CLI‑скрипта `mwlab.cli.pack_dataset`.

Проверяем, что:
    • вызывается без ошибки (через подмену sys.argv)
    • создаёт .h5‑файл
    • количество записей в HDF5 совпадает с числом *.sNp в исходной директории
"""

from __future__ import annotations

import sys
import pathlib

import pytest

from mwlab.io.backends.hdf5_backend import HDF5Backend


def test_pack_dataset_cli(tmp_dir, sample_dir, monkeypatch):
    """Полный end‑to‑end без tqdm‑спиннера."""
    dst = pathlib.Path(tmp_dir) / "packed.h5"

    # ---- подменяем sys.argv для CLI -----------------------------------
    argv = [
        "pack_dataset",
        "--src",
        str(sample_dir),
        "--dst",
        str(dst),
        "--pattern",
        "*.s2p",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # ---- отключаем прогресс‑бар, чтобы не засорял вывод ---------------
    import mwlab.cli.pack_dataset as cli

    monkeypatch.setattr(cli, "tqdm", lambda x, total=None: x)

    # ---- запускаем ----------------------------------------------------
    cli.main()

    # ---- проверки -----------------------------------------------------
    assert dst.exists()

    with HDF5Backend(dst, mode="r") as reader:
        assert len(reader) == len(list(sample_dir.rglob("*.s2p")))
