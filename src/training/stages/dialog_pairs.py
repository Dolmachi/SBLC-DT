from __future__ import annotations

import json
import logging
from itertools import groupby
from pathlib import Path
from typing import Any

from src.training.context import TrainingContext
from src.utils.fs import reset_dir


SegmentData = dict[str, Any]

TARGET_SPEAKER_ID = "SPEAKER_TARGET"


def normalize_text(text: str) -> str:
    """
    Минимальная нормализация текста реплики:
    убираем края и схлопываем лишние пробелы.
    """
    return " ".join(text.strip().split())


def load_segments(json_path: Path) -> list[SegmentData]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data["segments"]


def merge_consecutive_segments(segments: list[SegmentData]) -> list[SegmentData]:
    """
    Объединяет подряд идущие сегменты одного speaker.
    """
    merged: list[SegmentData] = []

    for speaker, group in groupby(segments, key=lambda seg: seg["speaker"]):
        text = " ".join(normalize_text(seg["text"]) for seg in group)

        if text:
            merged.append({
                "speaker": speaker,
                "text": text,
            })

    return merged


def build_dialog_pairs(segments: list[SegmentData]) -> list[dict[str, str]]:
    """
    Формирует пары:
    - подряд идущие не-target реплики собираются как user;
    - следующая target-реплика становится assistant.
    """
    pairs: list[dict[str, str]] = []
    user_buffer: list[str] = []

    for seg in segments:
        speaker = seg["speaker"]
        text = normalize_text(seg["text"])

        if speaker == TARGET_SPEAKER_ID:
            if user_buffer:
                pairs.append({
                    "user": " ".join(user_buffer),
                    "assistant": text
                })
                user_buffer.clear()
        else:
            user_buffer.append(text)

    return pairs


def save_jsonl(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 04: построение dialog pairs.
    """
    asr_dir = ctx.paths.asr_postprocess_dir
    out_dir = ctx.paths.dialog_pairs_dir
    out_path = ctx.paths.dialog_pairs_path

    logger.info("Начинаю dialog_pairs")
    reset_dir(out_dir)

    json_files = sorted(asr_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Не найдены ASR postprocess JSON-файлы: {asr_dir}")

    all_pairs: list[dict[str, str]] = []

    for json_path in json_files:
        logger.debug("Формирую пары из файла: %s", json_path.name)

        segments = load_segments(json_path)
        merged_segments = merge_consecutive_segments(segments)
        pairs = build_dialog_pairs(merged_segments)

        all_pairs.extend(pairs)

        logger.debug(
            "Из файла %s сформировано пар: %d",
            json_path.name,
            len(pairs),
        )

    if not all_pairs:
        raise RuntimeError(
            "Не удалось сформировать ни одной dialog pair. "
        )

    save_jsonl(all_pairs, out_path)

    logger.info("dialog_pairs завершён")
    logger.info("Всего dialog pairs сохранено: %d", len(all_pairs))
    logger.debug("Файл: %s", out_path)