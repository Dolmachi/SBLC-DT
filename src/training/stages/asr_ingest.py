from __future__ import annotations

import json
import logging
from pathlib import Path

from src.asr.base import TrainASR
from src.training.context import TrainingContext
from src.utils.fs import reset_dir


def save_asr_json(data: dict, out_path: Path) -> None:
    """
    Сохраняет ASR JSON в файл.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(
    ctx: TrainingContext,
    asr_engine: TrainASR,
    logger: logging.Logger,
) -> None:
    """
    Stage 02: ASR ingest.

    Что делает:
    1. Берёт wav-файлы из train_data/interim/audio_asr.
    2. Прогоняет каждый файл через train-ASR.
    3. Сохраняет JSON в train_data/interim/asr_raw.
    """
    audio_dir = ctx.paths.audio_asr_dir
    out_dir = ctx.paths.asr_raw_dir

    logger.info("Начинаю asr_ingest")

    reset_dir(out_dir)

    wav_files = sorted(audio_dir.glob("*.wav"))

    logger.info("Найдено wav-файлов для ASR: %d", len(wav_files))

    saved_count = 0
    skipped_count = 0

    for wav_path in wav_files:
        logger.debug("ASR обработка: %s", wav_path.name)

        try:
            asr_result = asr_engine.transcribe_file(wav_path)

            if asr_result is None:
                skipped_count += 1
                logger.warning(
                    "Файл %s пропущен: диаризация не прошла фильтр качества\n"
                    "Возможные причины:\n"
                    "- Слишком много говорящих людей на записи\n"
                    "- Один говорящий человек на записи\n"
                    "- Похожие голоса у интервьюера и копируемого человека\n"
                    "- Низкое качество записи\n",
                    wav_path.name,
                )
                continue

            out_json = out_dir / f"{wav_path.stem}.json"
            save_asr_json(asr_result, out_json)

            saved_count += 1
            logger.debug("ASR JSON сохранён: %s", out_json.name)

        except Exception as error:
            skipped_count += 1

            logger.warning(
                "Ошибка ASR на файле %s: %s",
                wav_path.name,
                error,
                exc_info=True,
            )

    if saved_count == 0:
        raise RuntimeError(
            "ASR модуль не смог транскрибировать ни один файл"
        )

    logger.info("asr_ingest завершён")
    logger.info("ASR JSON сохранено: %d", saved_count)
    logger.info("ASR файлов пропущено: %d", skipped_count)