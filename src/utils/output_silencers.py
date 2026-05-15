from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any


class NoOpTqdm:
    """
    No-op замена tqdm.
    """

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else ()

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *args, **kwargs):
        return None

    def close(self):
        return None

    def set_description(self, *args, **kwargs):
        return None

    def set_description_str(self, *args, **kwargs):
        return None

    def set_postfix(self, *args, **kwargs):
        return None

    def set_postfix_str(self, *args, **kwargs):
        return None

    def refresh(self, *args, **kwargs):
        return None


def no_op_print(*args: Any, **kwargs: Any) -> None:
    """
    No-op замена print.
    """
    return None


def patch_tqdm_in_modules(module_names: Iterable[str]) -> None:
    """
    Отключает tqdm в указанных уже импортируемых/импортируемых модулях.
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        if hasattr(module, "tqdm"):
            module.tqdm = NoOpTqdm


def patch_print_in_modules(module_names: Iterable[str]) -> None:
    """
    Отключает print внутри конкретных third-party модулей.
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        module.print = no_op_print
        
def configure_hf_quiet_env() -> None:
    """
    Настройки окружения для HuggingFace/Transformers.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def quiet_transformers_logging() -> None:
    """
    Глушит warning-и transformers/HuggingFace, которые могут влезать
    прямо в stdout рядом с LLM streaming.
    """
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.pipelines").setLevel(logging.ERROR)
    logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass
    
@contextlib.contextmanager
def suppress_stdout_stderr() -> Iterator[None]:
    """
    Временно глушит stdout/stderr на уровне file descriptors.
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        sys.stdout.flush()
        sys.stderr.flush()

        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)

            yield

    finally:
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)

        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        