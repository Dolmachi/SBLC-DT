from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from unidecode import unidecode

from src.training.context import TrainingContext
from src.utils.fs import reset_dir


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp"}

ASR_SR = 16000
TTS_SR = 16000

REFERENCE_MIN_SEC = 3.0
REFERENCE_MAX_SEC = 30.0


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """
    Запускает внешнюю команду
    """
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def sanitize_media_stem(text: str, *, max_len: int = 80) -> str:
    """
    Преобразует имя медиафайла в безопасный stem для выходного wav.
    """
    text = unidecode((text or "").strip()).lower()

    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"[_-]{2,}", "_", text)
    text = text.strip("_-")

    if len(text) > max_len:
        text = text[:max_len].rstrip("_-")

    return text or "media"


def make_unique_stem(stem: str, used: set[str]) -> str:
    """
    Делает stem уникальным внутри текущего запуска.

    Нужно на случай, если у пользователя есть файлы:
    interview.mp4 и interview.wav.
    """
    if stem not in used:
        used.add(stem)
        return stem

    index = 2
    while f"{stem}_{index}" in used:
        index += 1

    unique = f"{stem}_{index}"
    used.add(unique)
    
    return unique


def discover_media(input_dir: Path) -> list[Path]:
    """
    Находит все аудио/видео файлы внутри source/dialogs.
    """
    media_exts = AUDIO_EXTS | VIDEO_EXTS
    paths: list[Path] = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in media_exts:
            paths.append(path)

    return sorted(paths)


def measure_peak_gain_db(
    input_path: Path,
    target_dbfs: float = -1.0,
    max_abs_gain_db: float = 8.0,
) -> float:
    """
    Определяет, на сколько dB надо поднять/опустить громкость,
    чтобы peak был около target_dbfs.

    Ограничиваем усиление/ослабление max_abs_gain_db,
    чтобы не раздувать шум и не делать агрессивную нормализацию.
    """
    try:
        result = run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(input_path),
                "-vn",
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ]
        )

        match = re.search(
            r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB",
            result.stderr,
            flags=re.IGNORECASE,
        )

        if not match:
            return 0.0

        max_volume = float(match.group(1))
        gain_db = target_dbfs - max_volume

        if abs(gain_db) < 0.1:
            return 0.0

        return max(-max_abs_gain_db, min(max_abs_gain_db, gain_db))

    except Exception:
        return 0.0


def convert_to_wav(
    input_media_path: Path,
    output_wav_path: Path,
    sample_rate: int,
    gain_db: float,
) -> None:
    """
    Конвертирует аудио/видео в mono WAV PCM s16le с нужной частотой дискретизации.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-i",
            str(input_media_path),
            "-vn",
            "-af",
            f"volume={gain_db}dB",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_wav_path),
        ]
    )


def delete_audio_pair(
    wav_name: str,
    asr_dir: Path,
    tts_dir: Path,
) -> None:
    """
    Удаляет ASR/TTS версии одного и того же аудио.
    """
    for folder in (asr_dir, tts_dir):
        path = folder / wav_name
        if path.exists():
            os.remove(path)


def prepare_reference_audio(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Готовит root-level reference.* для speaker embedding.

    reference.* не попадает в dialogs corpus.
    Он используется только для определения SPEAKER_TARGET.
    """

    source_path = ctx.paths.source_reference_path
    output_path = ctx.paths.reference_wav_path

    if not source_path.exists():
        raise FileNotFoundError(f"Не найден reference audio: {source_path}")

    logger.debug(
        "Конвертирую reference audio: %s -> %s",
        source_path.name,
        output_path,
    )

    gain_db = measure_peak_gain_db(source_path)

    convert_to_wav(
        input_media_path=source_path,
        output_wav_path=output_path,
        sample_rate=ASR_SR,
        gain_db=gain_db,
    )
    
    duration_sec = float(librosa.get_duration(path=str(output_path)))

    if duration_sec < REFERENCE_MIN_SEC or duration_sec > REFERENCE_MAX_SEC:
        raise RuntimeError(
            "Некорректная длительность reference audio.\n"
            f"Файл: {source_path}\n"
            f"Длительность после конвертации: {duration_sec:.2f} с\n"
            f"Ожидается: {REFERENCE_MIN_SEC:.0f}–{REFERENCE_MAX_SEC:.0f} с.\n"
            "Положи короткий чистый фрагмент голоса target-человека без собеседника, "
            "музыки, сильного шума и длинной тишины."
        )


def cleanup_audio_pairs(
    asr_dir: Path,
    tts_dir: Path,
    origin_map: dict[str, str],
    logger: logging.Logger,
) -> None:
    """
    Удаляет плохие и дублирующиеся аудио.
    """
    hashes: dict[str, list[str]] = defaultdict(list)

    for tts_wav in sorted(tts_dir.glob("*.wav")):
        wav_name = tts_wav.name
        original_name = origin_map.get(wav_name, wav_name)

        try:
            y, _ = librosa.load(tts_wav, sr=8000, mono=True)

            if len(y) == 0:
                delete_audio_pair(wav_name, asr_dir, tts_dir)
                logger.warning("Аудио файла %s пустое. Файл исключён.", original_name)
                continue

            duration_sec = len(y) / 8000
            if duration_sec < 10:
                delete_audio_pair(wav_name, asr_dir, tts_dir)
                logger.warning("Аудио файла %s слишком короткое (<10 с). Файл исключён.", original_name)
                continue

            y = librosa.util.normalize(y)
            spectrum = np.mean(np.abs(librosa.stft(y)), axis=1)
            spectrum_max = np.max(spectrum)

            rms = float(np.sqrt(np.mean(y ** 2)))

            if rms < 0.001 or np.isnan(spectrum_max) or spectrum_max <= 0:
                delete_audio_pair(wav_name, asr_dir, tts_dir)
                logger.warning("Аудио файла %s слишком тихое/битое. Файл исключён.", original_name)
                continue

            audio_hash = hashlib.md5(
                (spectrum / spectrum_max).astype(np.float32).tobytes()
            ).hexdigest()

            hashes[audio_hash].append(wav_name)

        except Exception:
            delete_audio_pair(wav_name, asr_dir, tts_dir)
            logger.warning("Аудио файла %s повреждено. Файл исключён.", original_name)

    for duplicate_names in hashes.values():
        # Первый файл оставляем, остальные считаем дублями.
        for duplicate_wav_name in duplicate_names[1:]:
            original_name = origin_map.get(duplicate_wav_name, duplicate_wav_name)
            delete_audio_pair(duplicate_wav_name, asr_dir, tts_dir)
            logger.warning("Аудио файла %s является дублем. Файл исключён.", original_name)


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 01: подготовка source-данных.

    Что делает:
    1. Конвертирует root-level reference.* в train_data/interim/reference.wav.
    2. Ищет аудио/видео в source/dialogs.
    3. Конвертирует каждый dialog-файл в:
    - audio_asr: 16 kHz mono wav
    - audio_tts: 16 kHz mono wav
    4. Удаляет пустые, короткие, тихие, битые и дублирующиеся dialog-аудио.
    """
    source_dialogs_dir = ctx.paths.source_dialogs_dir
    audio_asr_dir = ctx.paths.audio_asr_dir
    audio_tts_dir = ctx.paths.audio_tts_dir

    logger.info("Начинаю preprocess_source")
    logger.debug("Источник dialogs: %s", source_dialogs_dir)

    reset_dir(audio_asr_dir)
    reset_dir(audio_tts_dir)
    
    prepare_reference_audio(ctx, logger)

    media_files = discover_media(source_dialogs_dir)
    logger.info("Найдено медиафайлов: %d", len(media_files))

    origin_map: dict[str, str] = {}
    used_stems: set[str] = set()

    for media_path in media_files:
        base_stem = sanitize_media_stem(media_path.stem)
        out_stem = make_unique_stem(base_stem, used_stems)
        wav_name = f"{out_stem}.wav"

        asr_out = audio_asr_dir / wav_name
        tts_out = audio_tts_dir / wav_name

        logger.debug("Конвертирую: %s -> %s", media_path.name, wav_name)

        try:
            gain_db = measure_peak_gain_db(media_path)

            convert_to_wav(
                input_media_path=media_path,
                output_wav_path=asr_out,
                sample_rate=ASR_SR,
                gain_db=gain_db,
            )

            convert_to_wav(
                input_media_path=media_path,
                output_wav_path=tts_out,
                sample_rate=TTS_SR,
                gain_db=gain_db,
            )

            origin_map[wav_name] = media_path.name

        except subprocess.CalledProcessError as error:
            logger.warning(
                "FFmpeg не смог обработать файл %s. stderr:\n%s",
                media_path.name,
                error.stderr,
            )

            delete_audio_pair(wav_name, audio_asr_dir, audio_tts_dir)

        except Exception as error:
            logger.warning(
                "Ошибка при обработке файла %s: %s",
                media_path.name,
                error,
            )

            delete_audio_pair(wav_name, audio_asr_dir, audio_tts_dir)

    cleanup_audio_pairs(
        asr_dir=audio_asr_dir,
        tts_dir=audio_tts_dir,
        origin_map=origin_map,
        logger=logger,
    )

    final_asr_files = sorted(audio_asr_dir.glob("*.wav"))
    final_tts_files = sorted(audio_tts_dir.glob("*.wav"))

    if not final_asr_files or not final_tts_files:
        raise RuntimeError(
            "После preprocess_source не осталось пригодных аудиофайлов.\n"
            "Возможные причины: слишком короткие записи, битые файлы, отсутствие аудиодорожки, слишком тихий звук."
        )

    logger.info("preprocess_source завершён")