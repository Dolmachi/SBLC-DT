from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from src.asr.faster_whisper_runtime_asr import FasterWhisperASR
from src.training.context import TrainingContext
from src.utils.fs import reset_dir


ASRData = dict[str, Any]
SegmentData = dict[str, Any]
WordData = dict[str, Any]

TARGET_SPEAKER_ID = "SPEAKER_TARGET"

# Если первые сегменты плохие, файл лучше пропустить
N_START_SEGMENTS = 5

# Подрезка конца сегмента.
PAD_END = 0.00

REPUNCTUATION_MARGINS_SEC = (0.08, 0.20, 0.40)

WORD_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def has_sentence_punctuation(text: str) -> bool:
    """
    Проверяет наличие реальной пунктуации.
    """
    return any(char in text for char in ".?,!…;")


def normalize_tokens(text: str) -> list[str]:
    """
    Нормализует текст до последовательности слов.
    """
    return [
        match.group(0).lower().replace("ё", "е")
        for match in WORD_TOKEN_RE.finditer(text)
    ]


def build_text_from_words(words: list[WordData]) -> str:
    """
    Собирает текст сегмента из word-level данных WhisperX.
    """
    return " ".join(
        str(word.get("word", "")).strip()
        for word in words
        if str(word.get("word", "")).strip()
    ).strip()


def load_audio_slice(
    audio_path: Path,
    start: float,
    end: float,
    margin_sec: float,
) -> np.ndarray:
    offset = max(0.0, start - margin_sec)
    duration = max(0.0, end - start + 2 * margin_sec)

    audio, _sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
        offset=offset,
        duration=duration,
    )

    return np.asarray(audio, dtype=np.float32).flatten()


def apply_repunctuated_text(
    seg: SegmentData,
    new_text: str,
) -> SegmentData:
    updated = dict(seg)
    updated["text"] = new_text

    surface_words = new_text.split()
    old_words = list(seg.get("words", []))

    if len(surface_words) == len(old_words):
        updated_words: list[WordData] = []

        for old_word, new_word in zip(old_words, surface_words):
            item = dict(old_word)
            item["word"] = new_word
            updated_words.append(item)

        updated["words"] = updated_words

    return updated


def repunctuate_segment(
    seg: SegmentData,
    audio_path: Path,
    asr: FasterWhisperASR | None,
) -> SegmentData:
    if asr is None:
        return seg

    if not audio_path.exists():
        return seg

    old_text = str(seg.get("text", "")).strip()
    old_tokens = normalize_tokens(old_text)

    if not old_tokens:
        return seg

    start = float(seg["start"])
    end = float(seg["end"])

    if end <= start:
        return seg

    for margin_sec in REPUNCTUATION_MARGINS_SEC:
        audio = load_audio_slice(
            audio_path=audio_path,
            start=start,
            end=end,
            margin_sec=margin_sec,
        )

        if audio.size < int(16000 * 0.25):
            continue

        new_text = asr.transcribe(
            audio=audio,
            sample_rate=16000,
        ).strip()

        if not new_text:
            continue

        if not has_sentence_punctuation(new_text):
            continue

        if normalize_tokens(new_text) != old_tokens:
            continue

        return apply_repunctuated_text(
            seg=seg,
            new_text=new_text,
        )

    return seg


def define_target_speaker(segments: list[SegmentData]) -> tuple[bool, str | None]:
    """
    Определяет target speaker.

    Текущая эвристика:
    - берём порядок появления спикеров в сегментах;
    - считаем, что target speaker — второй говорящий.
    """
    speakers: list[str] = []

    for seg in segments:
        speaker = seg.get("speaker")

        if speaker and speaker not in speakers:
            speakers.append(speaker)

    if len(speakers) < 2:
        return (
            False,
            (
                "После постпроцессинга остался только один говорящий. "
                "Возможные причины:\n"
                "- низкое качество записи,\n"
                "- много синхронной речи"
            ),
        )

    source_target = speakers[1]

    for seg in segments:
        if seg.get("speaker") == source_target:
            seg["speaker"] = TARGET_SPEAKER_ID

        for word in seg.get("words", []):
            if word.get("speaker") == source_target:
                word["speaker"] = TARGET_SPEAKER_ID

    return True, None


def split_no_punctuation_run(
    run_segments: list[SegmentData],
    audio_path: Path,
    asr: FasterWhisperASR | None,
) -> list[SegmentData]:
    """
    Обрабатывает подряд идущие беспунктуационные сегменты.

    Логика:
    1. Склеиваем слова из нескольких WhisperX-сегментов.
    2. Режем общий поток слов по смене word-level speaker.
    3. Получившиеся speaker-clean куски пытаемся до-пунктуировать
       через faster-whisper.
    """
    merged_words: list[WordData] = []

    for seg in run_segments:
        merged_words.extend(seg["words"])

    if not merged_words:
        return []

    groups: list[tuple[str, list[WordData]]] = []

    current_speaker = merged_words[0]["speaker"]
    current_group = [merged_words[0]]

    for word in merged_words[1:]:
        if word["speaker"] == current_speaker:
            current_group.append(word)
        else:
            groups.append((current_speaker, current_group))
            current_speaker = word["speaker"]
            current_group = [word]

    groups.append((current_speaker, current_group))

    new_segments: list[SegmentData] = []

    for group_speaker, group_words in groups:
        start = float(group_words[0]["start"])
        end = float(group_words[-1]["end"]) - PAD_END

        if end <= start:
            continue

        words = [dict(word) for word in group_words]

        seg: SegmentData = {
            "start": start,
            "end": end,
            "text": build_text_from_words(words),
            "words": words,
            "speaker": group_speaker,
        }

        seg = repunctuate_segment(
            seg=seg,
            audio_path=audio_path,
            asr=asr,
        )

        new_segments.append(seg)

    return new_segments


def split_punctuation_segment(seg: SegmentData) -> list[SegmentData]:
    """
    Обрабатывает сегмент с пунктуацией.

    Основная идея:
    - если внутри сегмента встречается полноценное чужое предложение,
      переносим его в отдельный сегмент;
    - если чужие слова выглядят как отдельные ошибочные word-level метки,
      оставляем сегмент как есть.
    """
    words: list[WordData] = seg["words"]
    segment_speaker = seg["speaker"]

    if not words:
        return []

    def starts_with_upper(word: WordData) -> bool:
        text = (word.get("word") or "").strip().lstrip("\"'«»()[]{}—- ")
        return bool(text) and text[0].isalpha() and text[0].isupper()

    def ends_with_sentence_punct(word: WordData) -> bool:
        text = (word.get("word") or "").strip().rstrip("\"'«»()[]{} ")
        return bool(text) and text[-1] in ".?!…"

    result: list[SegmentData] = []
    i = 0
    chunk_start = 0

    while i < len(words):
        if words[i]["speaker"] == segment_speaker:
            i += 1
            continue

        foreign_speaker = words[i]["speaker"]
        foreign_start = i

        while i < len(words) and words[i]["speaker"] == foreign_speaker:
            i += 1

        foreign_end = i
        foreign_words = words[foreign_start:foreign_end]

        is_foreign_sentence = (
            len(foreign_words) >= 2
            and starts_with_upper(foreign_words[0])
            and ends_with_sentence_punct(foreign_words[-1])
        )

        if not is_foreign_sentence:
            continue

        # Кусок до чужого предложения
        if chunk_start < foreign_start:
            group_words = words[chunk_start:foreign_start]
            start = float(group_words[0]["start"])
            end = float(group_words[-1]["end"]) - PAD_END

            if end > start:
                result.append(
                    {
                        "start": start,
                        "end": end,
                        "text": " ".join(word["word"] for word in group_words).strip(),
                        "words": group_words,
                        "speaker": segment_speaker,
                    }
                )

        # чужое предложение
        start = float(foreign_words[0]["start"])
        end = float(foreign_words[-1]["end"]) - PAD_END

        if end > start:
            result.append(
                {
                    "start": start,
                    "end": end,
                    "text": " ".join(word["word"] for word in foreign_words).strip(),
                    "words": foreign_words,
                    "speaker": foreign_speaker,
                }
            )

        chunk_start = foreign_end

    # Хвост
    if chunk_start < len(words):
        group_words = words[chunk_start:]
        start = float(group_words[0]["start"])
        end = float(group_words[-1]["end"]) - PAD_END

        if end > start:
            result.append(
                {
                    "start": start,
                    "end": end,
                    "text": " ".join(word["word"] for word in group_words).strip(),
                    "words": group_words,
                    "speaker": segment_speaker,
                }
            )

    if result:
        return result

    # Если полноценного чужого предложения не нашли,
    # возвращаем сегмент в нормализованном виде.
    start = float(words[0]["start"])
    end = float(words[-1]["end"]) - PAD_END

    if end <= start:
        return []

    return [
        {
            "start": start,
            "end": end,
            "text": seg["text"],
            "words": words,
            "speaker": segment_speaker,
        }
    ]


def prepare_segments(data: ASRData) -> tuple[list[SegmentData] | None, str | None]:
    """
    Нормализует исходные WhisperX-сегменты.

    Что делаем:
    1. Убираем пустые сегменты.
    2. Если speaker сегмента отсутствует, восстанавливаем по словам.
    3. Если speaker у слова отсутствует, присваиваем speaker сегмента.

    Если проблема встречается в первых N_START_SEGMENTS, файл пропускаем,
    потому что старт важен для определения target speaker.
    """
    raw_segments = data.get("segments", [])
    prepared_segments: list[SegmentData] = []

    for index, seg in enumerate(raw_segments):
        words = seg.get("words")
        text = (seg.get("text") or "").strip()

        if not words or not text:
            # важен старт, поэтому отбрасываем плохие первые сегменты
            if index < N_START_SEGMENTS: 
                return (
                    None,
                    (
                        "ASR не смогла хорошо транскрибировать начало аудио. Причины:"
                        "- Шум/артефакты в начале аудио"
                        "- Невнятная речь в начале аудио"
                    ),
                )
            continue

        speaker = seg.get("speaker")

        # Заполняем пустые speaker
        if not speaker:
            word_speakers = [
                word.get("speaker")
                for word in words
                if word.get("speaker")
            ]

            if word_speakers:
                speaker = Counter(word_speakers).most_common(1)[0][0]
            else:
                if index < N_START_SEGMENTS:
                    return (
                        None,
                        (
                            "ASR плохо обработала начало аудио: "
                            "не удалось восстановить speaker в первых сегментах."
                        ),
                    )
                continue

        # Заполняем пустые speaker для слов
        for word in words:
            if not word.get("speaker"):
                word["speaker"] = speaker

        prepared_segments.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": text,
                "words": words,
                "speaker": speaker,
            }
        )

    return prepared_segments, None


def split_segments_by_speaker(
    prepared_segments: list[SegmentData],
    audio_path: Path,
    asr: FasterWhisperASR | None,
) -> list[SegmentData]:
    """
    Разделяет подготовленные сегменты на speaker-clean сегменты.

    Сегменты с пунктуацией обрабатываются мягко.
    Беспунктуационные run'ы режутся по word-level speaker и затем
    до-пунктуируются через faster-whisper.
    """
    new_segments: list[SegmentData] = []

    i = 0

    while i < len(prepared_segments):
        seg = prepared_segments[i]

        if has_sentence_punctuation(seg["text"]):
            new_segments.extend(split_punctuation_segment(seg))
            i += 1
            continue

        run_segments = [seg]
        i += 1

        while (
            i < len(prepared_segments)
            and not has_sentence_punctuation(prepared_segments[i]["text"])
        ):
            run_segments.append(prepared_segments[i])
            i += 1

        new_segments.extend(
            split_no_punctuation_run(
                run_segments=run_segments,
                audio_path=audio_path,
                asr=asr,
            )
        )

    return new_segments


def postprocess(
    data: ASRData,
    audio_path: Path,
    asr: FasterWhisperASR | None,
) -> tuple[ASRData | None, str | None]:
    """
    Полный postprocess одного ASR JSON.

    Возвращает:
    - (processed_data, None), если файл успешно обработан;
    - (None, reason), если файл нужно пропустить.
    """
    prepared_segments, reason = prepare_segments(data)
    if prepared_segments is None:
        return None, reason

    new_segments = split_segments_by_speaker(
        prepared_segments=prepared_segments,
        audio_path=audio_path,
        asr=asr,
    )

    if not new_segments:
        return None, "После разделения не осталось пригодных сегментов."

    target_ok, reason = define_target_speaker(new_segments)
    if not target_ok:
        return None, reason

    data["segments"] = new_segments
    return data, None


def save_json(data: ASRData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 03: ASR postprocess.

    Что делает:
    1. Читает JSON из asr_raw.
    2. Нормализует speaker labels.
    3. Разделяет смешанные сегменты.
    4. До-пунктуирует беспунктуационные speaker-clean сегменты.
    5. Определяет target speaker.
    6. Сохраняет результат в asr_postprocess.
    """
    raw_dir = ctx.paths.asr_raw_dir
    out_dir = ctx.paths.asr_postprocess_dir
    audio_dir = ctx.paths.audio_asr_dir

    logger.info("Начинаю asr_postprocess")

    reset_dir(out_dir)

    raw_files = sorted(raw_dir.glob("*.json"))

    logger.info("Найдено ASR raw JSON-файлов: %d", len(raw_files))

    saved_count = 0
    skipped_count = 0

    asr = FasterWhisperASR(
        language=ctx.cfg.lang,
        logger=logger,
        model_size='large-v3',
        vad_filter=False,
    )

    try:
        for raw_path in raw_files:
            logger.debug("Postprocess ASR JSON: %s", raw_path.name)

            try:
                audio_path = audio_dir / f"{raw_path.stem}.wav"

                data = json.loads(raw_path.read_text(encoding="utf-8"))

                processed, reason = postprocess(
                    data=data,
                    audio_path=audio_path,
                    asr=asr,
                )

                if processed is None:
                    skipped_count += 1
                    logger.warning(
                        "Файл %s пропущен. Причина: %s",
                        raw_path.name,
                        reason or "неизвестная причина",
                    )
                    continue

                out_path = out_dir / raw_path.name
                save_json(processed, out_path)

                saved_count += 1
                logger.debug("Сохранён postprocessed JSON: %s", out_path.name)

            except Exception as error:
                skipped_count += 1

                logger.warning(
                    "Ошибка postprocess на файле %s: %s",
                    raw_path.name,
                    error,
                    exc_info=True,
                )

    finally:
        asr.close()

    if saved_count == 0:
        raise RuntimeError(
            "ASR postprocess завершился без сохранённых файлов.\n"
            "Возможные причины: плохая транскрипция начала аудио, неудачная диаризация, "
            "слишком мало спикеров или некорректные ASR JSON."
        )

    logger.info("asr_postprocess завершён")
    logger.info("Postprocessed JSON сохранено: %d", saved_count)
    logger.info("Postprocessed JSON пропущено: %d", skipped_count)