# io/file_utils.py
import os
import pathlib
import zipfile
import warnings

def backup_to_drive(env_path: str, drive_zip_fname: str):
    """Архивирует папку `env_path` в ZIP‑файл `drive_zip_fname`."""
    env_path = pathlib.Path(env_path)
    if not env_path.exists():
        warnings.warn(f"Папка {env_path} не найдена.\nАрхивация не выполнена.", UserWarning)
        return
    with zipfile.ZipFile(drive_zip_fname, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(env_path):
            for f in files:
                fp = pathlib.Path(root) / f
                zf.write(fp, arcname=fp.relative_to(env_path))

def restore_from_drive(drive_zip_fname: str, restore_path: str):
    """Распаковывает ZIP‑архив в указанную директорию."""
    restore_path = pathlib.Path(restore_path)
    if not pathlib.Path(drive_zip_fname).exists():
        warnings.warn(f"Файл {drive_zip_fname} не найден.\nВосстановление не выполнено.", UserWarning)
        return
    with zipfile.ZipFile(drive_zip_fname, 'r') as zf:
        zf.extractall(restore_path)

