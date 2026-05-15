from __future__ import annotations

import shutil
from pathlib import Path


def ensure_dir(dir_path: Path) -> Path:
    """
    Создаёт директорию, если её ещё нет.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def reset_dir(dir_path: Path) -> None:
    """
    Полностью пересоздаёт директорию:
    """
    dir_path = Path(dir_path)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def remove_dir(dir_path: Path) -> None:
    """
    Удаляет директорию, если она существует
    """
    dir_path = Path(dir_path)

    if dir_path.exists():
        shutil.rmtree(dir_path)


def copy_file(src: Path, dst: Path) -> None:
    """
    Копирует один файл с сохранением метаданных.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    """
    Копирует директорию и её содержимое.
    """
    src = Path(src)
    dst = Path(dst)

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)