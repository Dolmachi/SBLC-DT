from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from unidecode import unidecode

from src.embeddings.registry import resolve_embedding_model_id
from src.utils.fs import copy_file, copy_tree, ensure_dir, reset_dir
from src.utils.logger import add_file_handler
from src.utils.profile_config import (
    ProfileConfig,
    ProfilePaths,
    build_profile_paths,
    save_profile_config,
)


BASE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROFILES_ROOT = BASE_ROOT / "profiles"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp"}

AVATAR_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ValidatedSourceBundle:
    """
    Проверенные исходные данные.

    Ожидаем структуру:
    data/
    ├── dialogs/
    ├── profile.txt
    └── avatar.jpg | avatar.png | avatar.webp | avatar.bmp
    """

    dialogs_dir: Path
    profile_txt_path: Path
    avatar_path: Path


def sanitize_slug(text: str, *, max_len: int = 64) -> str:
    """
    Преобразует имя человека в безопасный slug для имени папки профиля.
    """
    text = (text or "").strip()
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"[_-]{2,}", "_", text)
    text = text.strip("_-")

    if len(text) > max_len:
        text = text[:max_len].rstrip("_-")

    return text or "profile"


def has_media_files(folder: Path, exts: Iterable[str]) -> bool:
    exts_set = {ext.lower() for ext in exts}

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts_set:
            return True

    return False


def find_avatar_image(data_path: Path) -> Path:
    candidates: list[Path] = []

    wrong_avatar_files: list[Path] = []

    for path in data_path.iterdir():
        if not path.is_file():
            continue

        if path.stem.lower() != "avatar":
            continue

        ext = path.suffix.lower()

        if ext in AVATAR_IMAGE_EXTS:
            candidates.append(path)
        else:
            wrong_avatar_files.append(path)

    if wrong_avatar_files and not candidates:
        found = ", ".join(path.name for path in wrong_avatar_files)
        allowed = ", ".join(sorted(f"avatar{ext}" for ext in AVATAR_IMAGE_EXTS))
        raise RuntimeError(
            "Найден avatar-файл неподдерживаемого формата.\n"
            f"Найдено: {found}\n"
            "Для FlashHead avatar должен быть изображением.\n"
            f"Допустимые варианты: {allowed}"
        )

    if not candidates:
        allowed = ", ".join(sorted(f"avatar{ext}" for ext in AVATAR_IMAGE_EXTS))
        raise FileNotFoundError(
            "Не найден avatar-файл.\n"
            "Для FlashHead положи во входную папку один файл с именем avatar.\n"
            f"Допустимые варианты: {allowed}"
        )

    if len(candidates) > 1:
        found = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Найдено несколько avatar-изображений: {found}\n"
            "Оставь только один avatar-файл."
        )

    if wrong_avatar_files:
        found_wrong = ", ".join(path.name for path in wrong_avatar_files)
        found_ok = candidates[0].name
        raise RuntimeError(
            "Во входной папке одновременно есть корректный avatar image "
            "и лишние avatar-файлы неподдерживаемого формата.\n"
            f"Корректный файл: {found_ok}\n"
            f"Лишние файлы: {found_wrong}\n"
            "Оставь только один avatar image."
        )

    return candidates[0]


def validate_source_bundle(data_path: Path) -> ValidatedSourceBundle:
    """
    Проверяет, что пользователь передал корректную входную папку.

    Ожидаем структуру:
    data/
    ├── dialogs/
    ├── profile.txt
    └── avatar.jpg | avatar.png | avatar.webp | avatar.bmp
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Входная папка не найдена: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Ожидалась папка, но получен файл: {data_path}")

    dialogs_dir = data_path / "dialogs"
    if not dialogs_dir.exists() or not dialogs_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка dialogs: {dialogs_dir}")

    if not has_media_files(dialogs_dir, AUDIO_EXTS | VIDEO_EXTS):
        raise RuntimeError(f"В папке dialogs нет аудио или видео: {dialogs_dir}")

    profile_txt_path = data_path / "profile.txt"
    if not profile_txt_path.exists() or not profile_txt_path.is_file():
        raise FileNotFoundError(f"Не найден файл profile.txt: {profile_txt_path}")

    profile_text = profile_txt_path.read_text(encoding="utf-8").strip()
    if not profile_text:
        raise RuntimeError(f"Файл profile.txt пуст: {profile_txt_path}")

    avatar_path = find_avatar_image(data_path)

    return ValidatedSourceBundle(
        dialogs_dir=dialogs_dir,
        profile_txt_path=profile_txt_path,
        avatar_path=avatar_path,
    )


def create_profile_dirs(paths: ProfilePaths) -> None:
    """
    Создаёт базовую структуру директорий профиля.
    """
    ensure_dir(paths.source_dir)
    ensure_dir(paths.train_data_dir)
    ensure_dir(paths.interim_dir)
    ensure_dir(paths.processed_dir)

    ensure_dir(paths.artifacts_dir)
    ensure_dir(paths.artifacts_rag_dir)
    ensure_dir(paths.artifacts_tts_dir)
    ensure_dir(paths.artifacts_avatar_dir)

    ensure_dir(paths.memory_dir)
    ensure_dir(paths.logs_dir)


def run(name: str, lang: str, data_path: Path, logger: logging.Logger) -> Path:
    """
    Stage 00: инициализация профиля.

    Что делает:
    1. Проверяет входную папку.
    2. Создаёт profiles/<slug>/.
    3. Копирует туда source-данные.
    4. Создаёт базовые папки train/artifacts/logs.
    5. Сохраняет profile config.
    """
    data_path = Path(data_path).expanduser().resolve()
    logger.info("Проверяю входную папку: %s", data_path)

    bundle = validate_source_bundle(data_path)

    slug = sanitize_slug(name)
    profile_dir = PROFILES_ROOT / slug

    logger.info("Создаю профиль: %s", profile_dir)

    # Rebuild профиля с нуля.
    reset_dir(profile_dir)

    embedding_model_id = resolve_embedding_model_id(lang)
    logger.debug("Embedding model: %s", embedding_model_id)

    cfg = ProfileConfig(
        name=name.strip(),
        slug=slug,
        lang=lang.strip(),
        avatar_file_name=bundle.avatar_path.name,
        embedding_model_id=embedding_model_id,
    )

    paths = build_profile_paths(profile_dir, cfg)
    create_profile_dirs(paths)

    add_file_handler(logger, paths.logs_dir / "train.log")

    logger.debug("Копирую dialogs...")
    copy_tree(bundle.dialogs_dir, paths.source_dialogs_dir)

    logger.debug("Копирую profile.txt...")
    copy_file(bundle.profile_txt_path, paths.source_profile_txt)

    logger.debug("Копирую avatar image: %s", bundle.avatar_path.name)
    copy_file(bundle.avatar_path, paths.source_avatar_path)

    config_path = save_profile_config(cfg, profile_dir)
    logger.debug("Сохранён profile config: %s", config_path)
    logger.info("Инициализация профиля завершена")

    return profile_dir