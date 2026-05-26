from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torchaudio
from transformers import Wav2Vec2Processor
from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_HF,
    DEFAULT_ALIGN_MODELS_TORCH,
    LANGUAGES_WITHOUT_SPACES,
)

from src.training.context import TrainingContext
from src.utils.fs import reset_dir


SegmentData = dict[str, Any]

TARGET_SPEAKER_ID = "SPEAKER_TARGET"

# Фильтры качества сегментов для voice cloning.
MIN_SEGMENT_SEC = 3.0
MAX_SEGMENT_SEC = 30.0
MIN_WORDS = 3

# Фильтры по уверенности ASR.
MIN_AVG_SCORE = 0.80
BAD_SCORE_THRESHOLD = 0.50
MAX_BAD_FRACTION = 0.30

# Фильтры по темпу речи, слов/сек.
MIN_SPEECH_RATE = 1.0
MAX_SPEECH_RATE = 4.0

SENTENCE_PUNCTUATION_CHARS = ".?,!…;"

IGNORED_ALIGN_CHARS = set(SENTENCE_PUNCTUATION_CHARS) | {
    "«", "»", "\"", "'", "“", "”", "‘", "’",
    "(", ")", "[", "]", "{", "}",
    "-", "—", "–", ":",
}


@dataclass(frozen=True)
class TargetSegment:
    """
    Кандидат на аудиофрагмент для клонирования голоса.
    """
    audio_path: Path
    stem: str
    segment_idx: int
    start: float
    end: float
    text: str


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """
    Запускает внешнюю команду.
    """
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def load_alignment_dictionary(language_code: str) -> dict[str, int]:
    """
    Загружает dictionary alignment-модели для проверки,
    что текст сегмента может быть корректно выровнен.
    """
    if language_code in DEFAULT_ALIGN_MODELS_TORCH:
        model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        bundle = torchaudio.pipelines.__dict__[model_name]
        labels = bundle.get_labels()
        return {char.lower(): index for index, char in enumerate(labels)}

    if language_code in DEFAULT_ALIGN_MODELS_HF:
        model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        vocab = processor.tokenizer.get_vocab()
        return {char.lower(): code for char, code in vocab.items()}

    raise RuntimeError(f"Нет default alignment model для языка: {language_code}")


def load_segments(json_path: Path) -> list[SegmentData]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data["segments"]


def score_stats(seg: SegmentData) -> tuple[float, float]:
    """
    Считает:
    - средний ASR score слов;
    - долю слов с плохим score.
    """
    scores = [float(word["score"]) for word in seg["words"]]

    avg_score = sum(scores) / len(scores)
    bad_count = sum(score < BAD_SCORE_THRESHOLD for score in scores)
    bad_fraction = bad_count / len(scores)

    return avg_score, bad_fraction


def is_clean_target_segment(seg: SegmentData) -> bool:
    """
    Проверяет, что сегмент полностью принадлежит target speaker.
    """
    if seg["speaker"] != TARGET_SPEAKER_ID:
        return False

    return all(word["speaker"] == TARGET_SPEAKER_ID for word in seg["words"])


def has_sentence_punctuation(seg: SegmentData) -> bool:
    """
    Проверяет, что сегмент содержит хотя бы один знак пунктуации.

    Беспунктуационные сегменты не используем для voice cloning,
    потому что они чаще являются результатом плохой сегментации WhisperX
    и дают плохой prompt_text для VoxCPM2.
    """
    text = str(seg.get("text") or "").strip()

    return any(char in text for char in SENTENCE_PUNCTUATION_CHARS)


def is_fully_alignable_segment(
    seg: SegmentData,
    model_dictionary: dict[str, int],
    language: str,
) -> bool:
    """
    Проверяет, что все значимые символы текста покрываются
    alignment dictionary.

    Это повторяет ключевую логику WhisperX alignment preprocessing:
    - lower();
    - пробелы -> "|" для языков с пробелами;
    - крайние пробелы игнорируются.
    """
    text = seg["text"]

    num_leading = len(text) - len(text.lstrip())
    num_trailing = len(text) - len(text.rstrip())

    for char_index, char in enumerate(text):
        if char in IGNORED_ALIGN_CHARS:
            continue
        
        normalized_char = char.lower()

        if language not in LANGUAGES_WITHOUT_SPACES:
            normalized_char = normalized_char.replace(" ", "|")

        # WhisperX игнорирует крайние пробелы при alignment.
        if char_index < num_leading:
            continue

        if char_index > len(text) - num_trailing - 1:
            continue

        if normalized_char not in model_dictionary:
            return False

    return True


def passes_quality_filters(
    seg: SegmentData,
    model_dictionary: dict[str, int],
    language: str,
) -> bool:
    """
    Проверяет сегмент по всем фильтрам качества.
    """
    if not is_clean_target_segment(seg):
        return False
    
    # Фильтр по пунктуации
    if not has_sentence_punctuation(seg):
        return False

    start = float(seg["start"])
    end = float(seg["end"])
    duration = end - start

    # Фильтр по длительности
    if duration < MIN_SEGMENT_SEC or duration > MAX_SEGMENT_SEC:
        return False

    words = seg["words"]

    # Фильтр по количеству слов
    if len(words) < MIN_WORDS:
        return False

    # Фильтр по темпу речи
    speech_rate = len(words) / duration
    if speech_rate < MIN_SPEECH_RATE or speech_rate > MAX_SPEECH_RATE:
        return False

    # Фильтры по качеству ASR
    avg_score, bad_fraction = score_stats(seg)
    if avg_score < MIN_AVG_SCORE or bad_fraction > MAX_BAD_FRACTION:
        return False

    # Фильтр по alignability
    # (WhisperX может плохо выровнять сегмент, если в нём есть символы, отсутствующие в dictionary)
    if not is_fully_alignable_segment(seg, model_dictionary, language):
        return False

    return True


def find_available_stems(ctx: TrainingContext) -> list[str]:
    """
    Возвращает stems, для которых есть и postprocessed ASR JSON,
    и соответствующее TTS-аудио.
    """
    json_stems = {path.stem for path in ctx.paths.asr_postprocess_dir.glob("*.json")}
    audio_stems = {path.stem for path in ctx.paths.audio_tts_dir.glob("*.wav")}

    return sorted(json_stems & audio_stems)


def collect_target_segments(
    ctx: TrainingContext,
    stems: list[str],
    model_dictionary: dict[str, int],
    logger: logging.Logger,
) -> list[TargetSegment]:
    """
    Собирает чистые target-сегменты из выбранных аудио.
    """
    candidates: list[TargetSegment] = []

    for stem in stems:
        json_path = ctx.paths.asr_postprocess_dir / f"{stem}.json"
        audio_path = ctx.paths.audio_tts_dir / f"{stem}.wav"

        logger.debug("Ищу target-сегменты в файле: %s", stem)

        segments = load_segments(json_path)

        file_candidates = 0

        for segment_idx, seg in enumerate(segments):
            if not passes_quality_filters(
                seg=seg,
                model_dictionary=model_dictionary,
                language=ctx.cfg.lang,
            ):
                continue

            candidates.append(
                TargetSegment(
                    audio_path=audio_path,
                    stem=stem,
                    segment_idx=segment_idx,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=str(seg["text"]).strip(),
                )
            )

            file_candidates += 1

        logger.debug(
            "В файле %s найдено пригодных target-сегментов: %d",
            stem,
            file_candidates,
        )

    return candidates


def cut_audio_segment(segment: TargetSegment, out_path: Path) -> None:
    """
    Вырезает аудиофрагмент через ffmpeg.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(segment.audio_path),
            "-ss", str(segment.start),
            "-to", str(segment.end),
            "-acodec", "copy",
            str(out_path),
        ]
    )


def save_target_segments(
    segments: list[TargetSegment],
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Нарезает и сохраняет найденные target-сегменты c metadata
    """
    reset_dir(out_dir)

    total_duration = 0.0
    metadata: list[dict[str, object]] = []

    for segment in segments:
        out_name = f"{segment.stem}_seg_{segment.segment_idx}.wav"
        out_path = out_dir / out_name

        cut_audio_segment(segment, out_path)
        
        duration = segment.end - segment.start
        total_duration += duration
        
        metadata.append(
            {
                "file_name": out_name,
                "source_stem": segment.stem,
                "segment_idx": segment.segment_idx,
                "start": segment.start,
                "end": segment.end,
                "duration": duration,
                "text": segment.text,
            }
        )
        
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Сохранено target-сегментов: %d", len(segments))
    logger.info("Суммарная длительность target-сегментов: %.1f с", total_duration)
    logger.debug("Target segments dir: %s", out_dir)


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 06: выбор и нарезка target-сегментов для voice cloning / TTS training.

    Используем все postprocessed dialog-аудио.
    Root-level reference.* сюда не попадает: он нужен только для speaker-id.
    """
    logger.info("Начинаю target_segments")

    stems = find_available_stems(ctx)
    if not stems:
        raise RuntimeError(
            "Не найдены пары ASR JSON + TTS audio для target_segments."
        )

    model_dictionary = load_alignment_dictionary(ctx.cfg.lang)

    target_segments = collect_target_segments(
        ctx=ctx,
        stems=stems,
        model_dictionary=model_dictionary,
        logger=logger,
    )

    if not target_segments:
        raise RuntimeError(
            "Не удалось получить ни одного чистого target-сегмента для клонирования голоса.\n"
            "Возможные причины: мало речи target speaker, плохая диаризация, шум, "
            "слишком короткие/длинные реплики"
        )

    save_target_segments(
        segments=target_segments,
        out_dir=ctx.paths.target_audio_segments_dir,
        logger=logger,
    )

    logger.info("target_segments завершён")