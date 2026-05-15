from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str = "virtual_clone",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Создаёт и настраивает logger с выводом в консоль.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Проверяем, есть ли уже console handler, который мы добавили
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_own_console", False)
        for h in logger.handlers
    )

    if not has_console_handler:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Отмечаем этот handler, чтобы не добавлять его повторно
        console_handler._own_console = True  # type: ignore[attr-defined]

        logger.addHandler(console_handler)

    return logger


def add_file_handler(
    logger: logging.Logger,
    log_file: Path = Path("logs/virtual_clone.log"),
    log_level: int = logging.DEBUG,
) -> None:
    """
    Подключает к logger запись в файл.
    """
    log_file = Path(log_file).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == log_file:
                    return
            except Exception:
                pass

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)