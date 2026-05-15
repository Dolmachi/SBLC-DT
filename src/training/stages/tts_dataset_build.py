from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.training.context import TrainingContext
from src.utils.fs import reset_dir


# Целевая суммарная длительность reference-аудио для voice cloning.
MAX_REFERENCE_DURATION_SEC = 90.0

# Веса факторов качества. Сумма = 1.0.
W_SNR = 0.45       # отношение речи к шуму
W_BAND = 0.25      # спектральная "полнота" записи
W_CLIP = 0.15      # отсутствие клиппинга
W_PITCH = 0.05     # близость F0 к медиане корпуса
W_LOUD = 0.10      # нормальная локальная громкость


@dataclass
class ReferenceSegment:
    """
    Аудиосегмент-кандидат для TTS dataset.
    """
    path: Path
    duration: float
    snr_db: float
    clip_ratio: float
    f0_median: float | None
    centroid_hz: float
    rms: float
    score: float = 0.0
    metadata: dict[str, object] | None = None


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    return audio.astype(np.float32), int(sample_rate)


def load_metadata(target_dir: Path) -> dict[str, dict[str, object]]:
    metadata_path = target_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Не найден metadata target-сегментов: {metadata_path}")

    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {str(row["file_name"]): row for row in rows}


def compute_features(path: Path) -> ReferenceSegment:
    """
    Считает признаки качества для одного voice reference сегмента.
    """
    audio, sample_rate = load_mono_audio(path)
    duration = len(audio) / sample_rate

    # Клиппинг: доля сэмплов почти у 0 dBFS.
    clip_ratio = float(np.mean(np.abs(audio) > 0.99))

    # Оценка SNR через RMS коротких фреймов:
    # 20-й перцентиль считаем "шумом", 80-й — "речью".
    frame_len = int(0.02 * sample_rate)
    hop_len = max(1, frame_len // 2)

    frames = librosa.util.frame(
        audio,
        frame_length=frame_len,
        hop_length=hop_len,
    )

    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-12)
    noise_rms = float(np.percentile(frame_rms, 20))
    speech_rms = float(np.percentile(frame_rms, 80))

    snr_db = 20.0 * np.log10((speech_rms + 1e-9) / (noise_rms + 1e-9))

    # локальная громкость сегмента
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))

    # F0 (медиана)
    f0_median = estimate_f0_median(audio, sample_rate)
    # спектральный центроид
    centroid_hz = estimate_spectral_centroid(audio, sample_rate)

    return ReferenceSegment(
        path=path,
        duration=duration,
        snr_db=snr_db,
        clip_ratio=clip_ratio,
        f0_median=f0_median,
        centroid_hz=centroid_hz,
        rms=rms,
    )


def estimate_f0_median(audio: np.ndarray, sample_rate: int) -> float | None:
    """
    Оценивает медианную F0.

    Если F0 не удалось извлечь, возвращаем None.
    Дальше это будет трактоваться как нейтральный фактор, а не как провал.
    """
    try:
        f0, _, _ = librosa.pyin(
            audio,
            fmin=60.0,
            fmax=400.0,
            sr=sample_rate,
        )

        values = f0[~np.isnan(f0)]
        if len(values) == 0:
            return None

        return float(np.median(values))

    except Exception:
        return None


def estimate_spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    """
    Оценивает средний спектральный центроид.

    Очень низкий центроид часто указывает на узкополосную/глухую запись.
    """
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
    )

    return float(np.mean(centroid))


def score_snr(snr_db: float) -> float:
    """
    <= 5 dB -> 0
    >= 25 dB -> 1
    между ними — линейно.
    """
    if snr_db <= 5.0:
        return 0.0

    if snr_db >= 25.0:
        return 1.0

    return (snr_db - 5.0) / 20.0


def score_clip(clip_ratio: float) -> float:
    """
    Чем меньше клиппинга, тем лучше.
    """
    if clip_ratio >= 0.01:
        return 0.0

    if clip_ratio <= 0.001:
        return 1.0

    return (0.01 - clip_ratio) / (0.01 - 0.001)


def score_bandwidth(centroid_hz: float) -> float:
    """
    Простая оценка полноты спектра.
    """
    if centroid_hz <= 1200.0:
        return 0.0

    if centroid_hz >= 3500.0:
        return 1.0

    return (centroid_hz - 1200.0) / (3500.0 - 1200.0)


def score_pitch(
    f0_median: float | None,
    reference_f0: float | None,
) -> float:
    """
    Оценивает, насколько F0 сегмента близка к медианной F0 корпуса.

    Если F0 не извлеклась — не штрафуем сегмент жёстко.
    """
    if f0_median is None or reference_f0 is None or reference_f0 <= 0:
        return 1.0

    ratio = f0_median / reference_f0

    if 0.8 <= ratio <= 1.2:
        return 1.0

    if ratio <= 0.6 or ratio >= 1.4:
        return 0.0

    if ratio < 0.8:
        return (ratio - 0.6) / (0.8 - 0.6)

    return (1.4 - ratio) / (1.4 - 1.2)


def score_loudness(
    rms: float,
    reference_rms: float,
) -> float:
    """
    Оценивает локальную громкость относительно медианы корпуса.
    """
    if reference_rms <= 0:
        return 1.0

    ratio = rms / reference_rms

    if ratio <= 0.4:
        return 0.0

    if ratio >= 0.8:
        return 1.0

    return (ratio - 0.4) / (0.8 - 0.4)


def assign_scores(segments: list[ReferenceSegment]) -> None:
    """
    Назначает итоговый score каждому сегменту.
    """
    f0_values = [
        segment.f0_median
        for segment in segments
        if segment.f0_median is not None
    ]

    reference_f0 = float(np.median(f0_values)) if f0_values else None
    reference_rms = float(np.median([segment.rms for segment in segments]))

    for segment in segments:
        segment.score = (
            W_SNR * score_snr(segment.snr_db)
            + W_BAND * score_bandwidth(segment.centroid_hz)
            + W_CLIP * score_clip(segment.clip_ratio)
            + W_PITCH * score_pitch(segment.f0_median, reference_f0)
            + W_LOUD * score_loudness(segment.rms, reference_rms)
        )


def select_best_segments(
    segments: list[ReferenceSegment],
    max_duration_sec: float,
) -> list[ReferenceSegment]:
    """
    Сортирует сегменты по score и набирает лучшие до max_duration_sec.
    """
    ranked = sorted(
        segments,
        key=lambda segment: segment.score,
        reverse=True,
    )

    selected: list[ReferenceSegment] = []
    total_duration = 0.0

    for segment in ranked:
        if total_duration >= max_duration_sec:
            break

        selected.append(segment)
        total_duration += segment.duration

    return selected


def copy_selected_segments(
    selected: list[ReferenceSegment],
    out_dir: Path,
) -> None:
    """
    Копирует выбранные сегменты в tts_dataset.
    """
    reset_dir(out_dir)
    
    metadata: list[dict[str, object]] = []

    for segment in selected:
        dst = out_dir / segment.path.name
        shutil.copy2(segment.path, dst)
        
        row = dict(segment.metadata or {})
        
        row["score"] = segment.score
        row["snr_db"] = segment.snr_db
        row["clip_ratio"] = segment.clip_ratio
        row["f0_median"] = segment.f0_median
        row["centroid_hz"] = segment.centroid_hz
        row["rms"] = segment.rms
        
        metadata.append(row)
    
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 07: построение TTS reference dataset.

    Вход:
    - train_data/interim/target_audio_segments/*.wav

    Выход:
    - train_data/processed/tts_dataset/*.wav
    """
    in_dir = ctx.paths.target_audio_segments_dir
    out_dir = ctx.paths.tts_dataset_dir

    logger.info("Начинаю tts_dataset_build")
    logger.debug("Target segments dir: %s", in_dir)

    wav_files = sorted(in_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Не найдены target-сегменты: {in_dir}")

    logger.debug("Найдено target-сегментов: %d", len(wav_files))
    
    metadata = load_metadata(in_dir)

    segments: list[ReferenceSegment] = []

    for wav_path in wav_files:
        try:
            segment = compute_features(wav_path)
            segment.metadata = metadata[wav_path.name]
            segments.append(segment)
        except Exception as error:
            logger.warning(
                "Не удалось посчитать признаки для %s: %s",
                wav_path.name,
                error,
                exc_info=True,
            )
    

    if not segments:
        raise RuntimeError(
            "Не удалось посчитать признаки ни для одного target-сегмента."
        )

    assign_scores(segments)

    selected = select_best_segments(
        segments=segments,
        max_duration_sec=MAX_REFERENCE_DURATION_SEC,
    )

    if not selected:
        raise RuntimeError("Не удалось выбрать сегменты для TTS dataset.")

    copy_selected_segments(
        selected=selected,
        out_dir=out_dir,
    )

    total_duration = sum(segment.duration for segment in selected)

    logger.info("tts_dataset_build завершён")
    logger.info("Выбрано dataset-сегментов: %d", len(selected))
    logger.info("Суммарная длительность dataset: %.1f с", total_duration)
    logger.debug("TTS dataset dir: %s", out_dir)