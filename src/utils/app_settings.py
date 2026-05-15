from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_ROOT / ".env"


@dataclass
class AppSettings:
    """
    Глобальные настройки всего проекта.
    """
    hf_token: str | None


def load_app_settings() -> AppSettings:
    """
    Загружает настройки приложения.
    """
    load_dotenv(ENV_PATH)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is not None:
        hf_token = hf_token.strip()

    return AppSettings(hf_token=hf_token)


def require_hf_token() -> str:
    """
    Возвращает Hugging Face token.
    """
    settings = load_app_settings()

    if not settings.hf_token:
        raise RuntimeError(
            "Не найден Hugging Face token.\n"
            "Запусти:\n"
            "  python scripts/setup.py --hf-token hf_xxx\n"
        )

    return settings.hf_token