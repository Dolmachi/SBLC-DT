from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.training.context import TrainingContext
from src.utils.fs import ensure_dir, reset_dir


# VoxCPM2 docs: practical reference range 5–30 sec.
MAX_PROMPT_DURATION_SEC = 30.0


def load_tts_dataset_metadata(tts_dataset_dir: Path) -> list[dict[str, Any]]:
    metadata_path = tts_dataset_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Не найден metadata TTS dataset: {metadata_path}")

    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    return rows


def choose_best_prompt_segment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Выбирает один фрагмент для VoxCPM2 Ultimate Cloning.

    Логика:
    - tts_dataset уже содержит качественные сегменты;
    - значит здесь выбираем самый долгий фрагмент;
    - если есть фрагменты <= 30 секунд, выбираем самый долгий среди них.
    """
    within_limit = [
        row for row in rows
        if float(row["duration"]) <= MAX_PROMPT_DURATION_SEC
    ]

    candidates = within_limit if within_limit else rows

    return max(
        candidates,
        key=lambda row: float(row["duration"]),
    )


def save_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 08: подготовка TTS artifacts для VoxCPM2 Ultimate Cloning.

    Вход:
    - train_data/processed/tts_dataset/*.wav
    - train_data/processed/tts_dataset/metadata.json

    Выход:
    - artifacts/tts/reference.wav
    - artifacts/tts/reference.txt
    """
    tts_dataset_dir = ctx.paths.tts_dataset_dir
    artifacts_tts_dir = ctx.paths.artifacts_tts_dir

    logger.info("Начинаю tts_prepare")
    logger.debug("TTS dataset dir: %s", tts_dataset_dir)

    metadata = load_tts_dataset_metadata(tts_dataset_dir)
    if not metadata:
        raise RuntimeError("TTS dataset metadata пуст.")

    selected = choose_best_prompt_segment(metadata)

    selected_file_name = str(selected["file_name"])
    selected_text = str(selected["text"]).strip()

    if not selected_text:
        raise RuntimeError(
            f"У выбранного TTS prompt-сегмента нет текста: {selected_file_name}"
        )

    source_wav = tts_dataset_dir / selected_file_name
    if not source_wav.exists():
        raise FileNotFoundError(f"Не найден выбранный TTS wav: {source_wav}")

    reset_dir(artifacts_tts_dir)
    ensure_dir(artifacts_tts_dir)

    shutil.copy2(source_wav, ctx.paths.artifacts_tts_reference_wav_path)
    save_text(ctx.paths.artifacts_tts_reference_text_path, selected_text)

    logger.info("tts_prepare завершён")
    logger.debug("VoxCPM2 artifacts: %s", ctx.paths.artifacts_tts_reference_wav_path)