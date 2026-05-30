from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf
import shutil

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
from src.utils.fs import ensure_dir, reset_dir


SegmentData = dict[str, Any]

TARGET_SPEAKER_ID = "SPEAKER_TARGET"

MIN_SEGMENT_SEC = 3.0
MAX_SEGMENT_SEC = 15.0
MIN_WORDS = 3

MIN_AVG_SCORE = 0.80
BAD_SCORE_THRESHOLD = 0.50
MAX_BAD_FRACTION = 0.30

MIN_SPEECH_RATE = 1.0
MAX_SPEECH_RATE = 4.0

MIN_SEGMENTS_FOR_VAL = 50
VAL_EVERY_N = 10

DATASET_SAMPLE_RATE = 16000

# Waveform tail trim
RMS_FRAME_SEC = 0.020
RMS_HOP_SEC = 0.010

TAIL_GAP_SEARCH_SEC = 1.25
MIN_QUIET_GAP_SEC = 0.18
MAX_AFTER_GAP_SEC = 0.90

KEEP_AFTER_SPEECH_SEC = 0.06
MIN_TRIM_SEC = 0.08

ABS_SILENCE_DB = -48.0
REL_SILENCE_FROM_PEAK_DB = 35.0
NOISE_MARGIN_DB = 8.0

SENTENCE_PUNCTUATION_CHARS = ".?,!…;"

IGNORED_ALIGN_CHARS = set(SENTENCE_PUNCTUATION_CHARS) | {
    "«", "»", "\"", "'", "“", "”", "‘", "’",
    "(", ")", "[", "]", "{", "}",
    "-", "—", "–", ":",
}


@dataclass(frozen=True)
class TTSSegment:
    audio_path: Path
    stem: str
    segment_idx: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CutResult:
    duration: float
    tail_trim_sec: float
    tail_trim_reason: str


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
    Считает средний ASR score и долю слов с плохим score.
    """
    scores = [float(word["score"]) for word in seg["words"]]

    avg_score = sum(scores) / len(scores)
    bad_count = sum(score < BAD_SCORE_THRESHOLD for score in scores)
    bad_fraction = bad_count / len(scores)

    return avg_score, bad_fraction


def is_clean_target_segment(seg: SegmentData) -> bool:
    """
    Проверяет, что сегмент полностью принадлежит SPEAKER_TARGET.
    """
    if seg["speaker"] != TARGET_SPEAKER_ID:
        return False

    return all(word["speaker"] == TARGET_SPEAKER_ID for word in seg["words"])


def has_sentence_punctuation(seg: SegmentData) -> bool:
    """
    Проверяет наличие знаков пунктуации.
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

        # WhisperX игнорирует крайние пробелы при alignment
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
    Проверяет сегмент по фильтрам качества
    """
    # Проверка, что сегмент целиком принадлежит SPEAKER_TARGET
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
    Находит файлы, для которых есть postprocessed ASR JSON и TTS-source wav.
    """
    json_stems = {path.stem for path in ctx.paths.asr_postprocess_dir.glob("*.json")}
    audio_stems = {path.stem for path in ctx.paths.audio_tts_dir.glob("*.wav")}

    return sorted(json_stems & audio_stems)


def collect_tts_segments(
    ctx: TrainingContext,
    stems: list[str],
    model_dictionary: dict[str, int],
    logger: logging.Logger,
) -> list[TTSSegment]:
    """
    Собирает чистые SPEAKER_TARGET сегменты.
    """
    segments_out: list[TTSSegment] = []

    for stem in stems:
        json_path = ctx.paths.asr_postprocess_dir / f"{stem}.json"
        audio_path = ctx.paths.audio_tts_dir / f"{stem}.wav"

        logger.debug("Ищу TTS-сегменты в файле: %s", stem)

        file_count = 0

        for segment_idx, seg in enumerate(load_segments(json_path)):
            if not passes_quality_filters(
                seg=seg,
                model_dictionary=model_dictionary,
                language=ctx.cfg.lang,
            ):
                continue

            segments_out.append(
                TTSSegment(
                    audio_path=audio_path,
                    stem=stem,
                    segment_idx=segment_idx,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=str(seg["text"]).strip(),
                )
            )

            file_count += 1

        logger.debug("В файле %s пригодных TTS-сегментов: %d", stem, file_count)

    return segments_out


def rms_db_frames(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    """
    Возвращает dB RMS по коротким фреймам.

    Используется только для анализа хвоста уже вырезанного TTS-клипа.
    """
    frame_length = max(1, int(RMS_FRAME_SEC * sample_rate))
    hop_length = max(1, int(RMS_HOP_SEC * sample_rate))

    if audio.size < frame_length:
        rms = np.asarray(
            [np.sqrt(np.mean(audio**2) + 1e-12)],
            dtype=np.float32,
        )
    else:
        frames = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-12)

    db = 20.0 * np.log10(np.maximum(rms, 1e-8))

    return db.astype(np.float32), frame_length, hop_length


def speech_threshold_db(db: np.ndarray) -> float:
    """
    Робастный порог speech/non-speech для конкретного клипа.

    Не используем только абсолютный -48 dB, потому что записи могут отличаться
    по уровню шума и громкости.
    """
    peak_db = float(np.max(db))
    noise_floor_db = float(np.percentile(db, 20))

    return max(
        ABS_SILENCE_DB,
        peak_db - REL_SILENCE_FROM_PEAK_DB,
        noise_floor_db + NOISE_MARGIN_DB,
    )


def find_tail_cut_sample(audio: np.ndarray, sample_rate: int) -> tuple[int, str]:
    """
    Ищет безопасную точку отрезания хвоста.

    Случай 1:
        target speech -> silence
        Тогда режем trailing low-energy tail.

    Случай 2:
        target speech -> silence gap -> чужой кусочек / шум / начало следующего segment
        Тогда режем на начале тихого gap в последней части клипа.
    """
    min_samples = int(MIN_SEGMENT_SEC * sample_rate)

    if audio.size <= min_samples:
        return audio.size, "none"

    db, frame_length, hop_length = rms_db_frames(audio, sample_rate)
    threshold = speech_threshold_db(db)

    quiet = db < threshold

    duration_sec = audio.size / sample_rate
    tail_start_sec = max(0.0, duration_sec - TAIL_GAP_SEARCH_SEC)
    tail_start_frame = int(tail_start_sec * sample_rate / hop_length)

    min_gap_frames = max(1, int(MIN_QUIET_GAP_SEC * sample_rate / hop_length))

    frame_count = len(quiet)
    index = max(0, tail_start_frame)

    while index < frame_count:
        if not quiet[index]:
            index += 1
            continue

        run_start = index

        while index < frame_count and quiet[index]:
            index += 1

        run_end = index
        run_len = run_end - run_start

        if run_len < min_gap_frames:
            continue

        gap_start_sec = run_start * hop_length / sample_rate
        after_gap_sec = duration_sec - gap_start_sec

        # Режем только gap близко к концу, чтобы не убить нормальную паузу
        # внутри фразы.
        if after_gap_sec > MAX_AFTER_GAP_SEC:
            continue

        cut_sec = gap_start_sec + KEEP_AFTER_SPEECH_SEC
        cut_sample = min(audio.size, int(cut_sec * sample_rate))

        if audio.size - cut_sample < int(MIN_TRIM_SEC * sample_rate):
            continue

        if cut_sample < min_samples:
            continue

        return cut_sample, "tail_gap"

    speech_frames = np.where(~quiet)[0]

    if speech_frames.size == 0:
        return audio.size, "none"

    last_speech_frame = int(speech_frames[-1])
    last_speech_end_sample = last_speech_frame * hop_length + frame_length

    cut_sample = min(
        audio.size,
        last_speech_end_sample + int(KEEP_AFTER_SPEECH_SEC * sample_rate),
    )

    if audio.size - cut_sample < int(MIN_TRIM_SEC * sample_rate):
        return audio.size, "none"

    if cut_sample < min_samples:
        return audio.size, "none"

    return cut_sample, "trailing_low_energy"


def trim_audio_tail(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float, str]:
    """
    Убирает хвост клипа по аудиосигналу.
    """
    cut_sample, reason = find_tail_cut_sample(audio, sample_rate)

    if cut_sample >= audio.size:
        return audio, 0.0, reason

    trimmed = audio[:cut_sample]
    trim_sec = (audio.size - cut_sample) / sample_rate

    return trimmed, trim_sec, reason


def cut_audio_segment(segment: TTSSegment, out_path: Path) -> CutResult | None:
    """
    Вырезает TTS-сегмент и чистит хвост по waveform.

    Входные start/end уже пришли из asr_postprocess.
    Здесь НЕ строим новые bounds по словам.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_duration = max(0.0, segment.end - segment.start)
    tmp_path = out_path.with_name(f"{out_path.stem}.raw.wav")

    try:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-nostats",
                "-i",
                str(segment.audio_path),
                "-ss",
                f"{segment.start:.3f}",
                "-t",
                f"{raw_duration:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(DATASET_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(tmp_path),
            ]
        )

        audio, _sr = librosa.load(
            tmp_path,
            sr=DATASET_SAMPLE_RATE,
            mono=True,
        )
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if audio.size == 0:
            return None

        trimmed, tail_trim_sec, trim_reason = trim_audio_tail(
            audio=audio,
            sample_rate=DATASET_SAMPLE_RATE,
        )

        duration = trimmed.size / DATASET_SAMPLE_RATE

        if duration < MIN_SEGMENT_SEC:
            return None

        sf.write(
            out_path,
            trimmed,
            DATASET_SAMPLE_RATE,
            subtype="PCM_16",
        )

        return CutResult(
            duration=duration,
            tail_trim_sec=tail_trim_sec,
            tail_trim_reason=trim_reason,
        )

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def split_train_val(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Делает детерминированный train/val split.

    Если сегментов мало, validation не откусываем:
    все данные идут в train.
    """
    if len(rows) < MIN_SEGMENTS_FOR_VAL:
        return rows, []

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if index % VAL_EVERY_N == 0:
            val_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows:
        return rows, []

    return train_rows, val_rows


def to_manifest_row(row: dict[str, Any], reference_audio_path: Path) -> dict[str, Any]:
    return {
        "audio": str(row["audio"]),
        "text": str(row["text"]),
        "duration": float(row["duration"]),
        "dataset_id": 0,
        "ref_audio": str(reference_audio_path.resolve()),
    }
    

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_tts_dataset(
    segments: list[TTSSegment],
    out_dir: Path,
    ctx: TrainingContext,
    logger: logging.Logger,
) -> None:
    """
    Сохраняет VoxCPM2 dataset:
    - wavs/*.wav
    - reference.wav
    - metadata.json
    - train.jsonl
    - val.jsonl

    reference.wav берется из пользовательского reference.*,
    подготовленного на preprocess_source.
    """
    reset_dir(out_dir)
    ensure_dir(ctx.paths.tts_wavs_dir)

    if not ctx.paths.reference_wav_path.exists():
        raise FileNotFoundError(f"Не найден prepared reference.wav: {ctx.paths.reference_wav_path}")

    dataset_reference_path = out_dir / "reference.wav"
    shutil.copy2(ctx.paths.reference_wav_path, dataset_reference_path)

    metadata: list[dict[str, Any]] = []
    total_duration = 0.0
    skipped_after_trim = 0

    for segment in segments:
        out_name = f"{segment.stem}_seg_{segment.segment_idx}.wav"
        out_path = ctx.paths.tts_wavs_dir / out_name

        cut_result = cut_audio_segment(segment, out_path)

        if cut_result is None:
            skipped_after_trim += 1

            if out_path.exists():
                out_path.unlink()

            continue

        total_duration += cut_result.duration
        effective_end = segment.start + cut_result.duration

        metadata.append(
            {
                "file_name": out_name,
                "audio": str(out_path.resolve()),
                "source_stem": segment.stem,
                "segment_idx": segment.segment_idx,
                "start": segment.start,
                "end": effective_end,
                "duration": cut_result.duration,
                "source_end": segment.end,
                "tail_trim_sec": cut_result.tail_trim_sec,
                "tail_trim_reason": cut_result.tail_trim_reason,
                "text": segment.text,
            }
        )

    if not metadata:
        raise RuntimeError("После TTS tail-trim не осталось пригодных сегментов.")

    train_meta, val_meta = split_train_val(metadata)

    train_manifest = [
        to_manifest_row(row, dataset_reference_path)
        for row in train_meta
    ]

    val_manifest = [
        to_manifest_row(row, dataset_reference_path)
        for row in val_meta
    ]

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_jsonl(ctx.paths.tts_train_manifest_path, train_manifest)
    write_jsonl(ctx.paths.tts_val_manifest_path, val_manifest)

    logger.info("TTS dataset сохранён: %d сегментов", len(metadata))
    logger.info("TTS segments skipped after trim: %d", skipped_after_trim)
    logger.info("Train samples: %d", len(train_manifest))
    logger.info("Val samples: %d", len(val_manifest))
    logger.info("TTS reference from user input: %s", dataset_reference_path)
    logger.info("Суммарная длительность TTS dataset: %.1f с", total_duration)
    logger.debug("TTS dataset dir: %s", out_dir)


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 06: построение VoxCPM2 TTS dataset.

    Выход:
    - train_data/processed/tts_dataset/wavs/*.wav
    - train_data/processed/tts_dataset/metadata.json
    - train_data/processed/tts_dataset/train.jsonl
    - train_data/processed/tts_dataset/val.jsonl
    """
    logger.info("Начинаю tts_dataset_build")

    stems = find_available_stems(ctx)

    if not stems:
        raise RuntimeError(
            "Не найдены пары ASR JSON + TTS audio для tts_dataset_build."
        )

    model_dictionary = load_alignment_dictionary(ctx.cfg.lang)

    segments = collect_tts_segments(
        ctx=ctx,
        stems=stems,
        model_dictionary=model_dictionary,
        logger=logger,
    )

    if not segments:
        raise RuntimeError(
            "Не удалось получить ни одного чистого TTS-сегмента.\n"
            "Возможные причины: мало речи SPEAKER_TARGET, плохая диаризация, "
            "низкое качество ASR, шум или слишком короткие/длинные реплики."
        )

    save_tts_dataset(
        segments=segments,
        out_dir=ctx.paths.tts_dataset_dir,
        ctx=ctx,
        logger=logger,
    )

    logger.info("tts_dataset_build завершён")