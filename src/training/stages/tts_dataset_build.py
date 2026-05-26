from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.training.context import TrainingContext
from src.utils.fs import reset_dir


def load_metadata(target_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_path = target_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Не найден metadata target-сегментов: {metadata_path}")

    rows = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        str(row["file_name"]): dict(row)
        for row in rows
    }


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 07: построение TTS dataset.

    Dataset = все чистые target-сегменты.
    Отбор одного prompt-сегмента для текущего zero-shot TTS остается в tts_prepare.
    """

    in_dir = ctx.paths.target_audio_segments_dir
    out_dir = ctx.paths.tts_dataset_dir

    logger.info("Начинаю tts_dataset_build")

    wav_files = sorted(in_dir.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"Не найдены target-сегменты: {in_dir}")

    metadata = load_metadata(in_dir)

    reset_dir(out_dir)

    rows: list[dict[str, Any]] = []
    total_duration = 0.0

    for wav_path in wav_files:
        if wav_path.name not in metadata:
            raise RuntimeError(f"Нет metadata для target-сегмента: {wav_path.name}")

        shutil.copy2(wav_path, out_dir / wav_path.name)

        row = dict(metadata[wav_path.name])
        rows.append(row)
        total_duration += float(row.get("duration", 0.0))

    (out_dir / "metadata.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("tts_dataset_build завершён")
    logger.info("Сегментов в TTS dataset: %d", len(rows))
    logger.info("Суммарная длительность TTS dataset: %.1f с", total_duration)