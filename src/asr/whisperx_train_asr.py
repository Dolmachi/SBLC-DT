from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("LIGHTNING_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
import whisperx
import numpy as np
from pyannote.audio import Inference, Model
from pyannote.core import Segment
from whisperx.diarize import DiarizationPipeline

from src.asr.base import ASRJson, TrainASR


TARGET_SPEAKER_ID = "SPEAKER_TARGET"

SPEAKER_EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"

EMBED_END_PAD_SEC = 0.50
MIN_EMBED_SEGMENT_SEC = 2.0
MAX_EMBED_SEGMENTS_PER_SPEAKER = 10
MAX_EMBED_TOTAL_SEC_PER_SPEAKER = 90.0


class WhisperXASR(TrainASR):
    """
    ASR-движок для training pipeline.

    Использует:
    - WhisperX large-v3 для транскрипции;
    - WhisperX alignment model для word-level timestamps;
    - pyannote для диаризации.
    """

    def __init__(
        self,
        language: str,
        hf_token: str,
        logger: logging.Logger,
        device: str | None = None,
        batch_size: int = 8,
        min_total_speech_sec: float = 30.0,
        min_speakers: int = 2,
        max_top_fraction: float = 0.93,
        min_second_speech_sec: float = 15.0,
        min_turns: int = 3,
        target_reference_path: Path | None = None,
    ) -> None:
        self.language = language
        self.hf_token = hf_token
        self.logger = logger
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Параметры фильтра качества диаризации
        self.min_total_speech_sec = min_total_speech_sec
        self.min_speakers = min_speakers
        self.max_top_fraction = max_top_fraction
        self.min_second_speech_sec = min_second_speech_sec
        self.min_turns = min_turns

        self.transcribe_model = whisperx.load_model(
            "large-v3",
            self.device,
            compute_type="float16" if self.device == "cuda" else "int8",
            language=self.language,
        )

        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=self.language,
            device=self.device,
        )

        self.diarize_model = DiarizationPipeline(
            token=self.hf_token,
            device=self.device,
        )

        self.speaker_embedder: Inference | None = None
        self.target_reference_embedding: np.ndarray | None = None

        if target_reference_path is not None:
            self.speaker_embedder = self._load_speaker_embedder()
            self.target_reference_embedding = self._embed_file(Path(target_reference_path))
            self.logger.debug("Target reference embedding готов")

        self.logger.debug("WhisperX train ASR загружен")

    def transcribe_file(self, wav_path: Path) -> ASRJson | None:
        """
        Выполняет полный train-ASR цикл:
        1. transcription;
        2. alignment;
        3. diarization;
        4. target speaker detection по reference embedding;
        5. assign word speakers.
        """
        wav_path = Path(wav_path)

        audio = whisperx.load_audio(str(wav_path))

        transcript = self.transcribe_model.transcribe(
            audio,
            batch_size=self.batch_size,
        )

        aligned = whisperx.align(
            transcript["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
        )
        
        diarize_segments = self.diarize_model(
            audio,
            min_speakers=self.min_speakers,
        )

        if not self._diarization_is_ok(diarize_segments):
            self.logger.warning(
                "Диаризация не прошла фильтр качества: %s",
                wav_path.name,
            )
            return None
        
        try:
            source_target_speaker, target_scores = self._find_target_speaker(
                wav_path=wav_path,
                diarize_df=diarize_segments,
            )
        except RuntimeError as error:
            self.logger.warning(
                "Не удалось определить target speaker для %s: %s",
                wav_path.name,
                error,
            )
            return None

        scores_text = ", ".join(
            f"{speaker}={score:.4f}"
            for speaker, score in sorted(target_scores.items())
        )

        self.logger.info(
            "Target speaker для %s: %s -> %s, similarity=%.4f | scores: %s",
            wav_path.name,
            source_target_speaker,
            TARGET_SPEAKER_ID,
            target_scores[source_target_speaker],
            scores_text,
        )

        diarize_segments = self._mark_target_speaker(
            diarize_df=diarize_segments,
            source_speaker=source_target_speaker,
        )

        result = whisperx.assign_word_speakers(
            diarize_segments,
            aligned,
        )

        result["target_speaker"] = {
            "method": "direct_wespeaker_embedding",
            "source_speaker": source_target_speaker,
            "target_speaker": TARGET_SPEAKER_ID,
            "scores": target_scores,
            "embedding_model": SPEAKER_EMBEDDING_MODEL_ID,
        }

        return result

    def _diarization_is_ok(self, diarize_df: Any) -> bool:
        """
        Проверяет качество диаризации.

        Логика нужна, чтобы отбрасывать записи, где:
        - почти всё говорит один человек;
        - слишком мало речи;
        - нет нормальной смены говорящих;
        - диаризация развалилась.
        """
        if diarize_df is None or len(diarize_df) == 0:
            return False

        df = diarize_df.copy()
        df["dur"] = (df["end"] - df["start"]).clip(lower=0.0)

        durations = df.groupby("speaker")["dur"].sum().sort_values(ascending=False)

        total_speech = float(durations.sum())
        num_speakers = int(durations.shape[0])

        if num_speakers > 0 and total_speech > 0:
            top_fraction = float(durations.iloc[0] / total_speech)
        else:
            top_fraction = 1.0

        second_speech_sec = float(durations.iloc[1]) if num_speakers >= 2 else 0.0
        turns = self._count_speaker_turns(df)

        if total_speech < self.min_total_speech_sec:
            return False

        if num_speakers < self.min_speakers:
            return False

        if top_fraction > self.max_top_fraction and second_speech_sec < self.min_second_speech_sec:
            return False

        if turns < self.min_turns:
            return False

        return True

    @staticmethod
    def _count_speaker_turns(diarize_df: Any) -> int:
        """
        Считает количество смен говорящего в diarization dataframe.
        """
        df_sorted = diarize_df.sort_values("start")
        speakers = df_sorted["speaker"].tolist()

        turns = 0
        previous_speaker = None

        for speaker in speakers:
            if previous_speaker is not None and speaker != previous_speaker:
                turns += 1

            previous_speaker = speaker

        return turns
    
    def _load_speaker_embedder(self) -> Inference:
        """
        Загружает speaker embedding model для speaker verification.

        Важно:
        - это не diarization pipeline;
        - это прямой embedding extractor;
        - модель предназначена для сравнения голосов через cosine similarity.
        """
        model = Model.from_pretrained(
            SPEAKER_EMBEDDING_MODEL_ID,
        )

        inference = Inference(model, window="whole")
        inference.to(torch.device(self.device))

        return inference

    @staticmethod
    def _normalize_embedding(embedding: Any) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float32)

        if arr.ndim > 1:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)

        arr = arr.reshape(-1)

        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            raise RuntimeError("Speaker embedding имеет нулевую норму.")

        return arr / norm

    def _embed_file(self, wav_path: Path) -> np.ndarray:
        """
        Считает speaker embedding для целого reference-файла.
        """
        if self.speaker_embedder is None:
            raise RuntimeError("Speaker embedder не загружен.")

        embedding = self.speaker_embedder(str(wav_path))
        return self._normalize_embedding(embedding)

    def _embed_region(
        self,
        wav_path: Path,
        start: float,
        end: float,
    ) -> np.ndarray:
        """
        Считает speaker embedding для фрагмента аудио.
        """
        region = Segment(start, end)

        embedding = self.speaker_embedder.crop(str(wav_path), region)
        return self._normalize_embedding(embedding)

    def _select_speaker_regions(
        self,
        diarize_df: Any,
        speaker: str,
    ) -> list[tuple[float, float, float]]:
        """
        Берет самые длинные регионы speaker-а для speaker embedding.

        Возвращает список:
        (start, end, duration)
        """
        rows = diarize_df[diarize_df["speaker"] == speaker].copy()

        regions: list[tuple[float, float, float]] = []

        for _, row in rows.iterrows():
            start = float(row["start"])
            end = float(row["end"]) - EMBED_END_PAD_SEC
            
            if end <= start:
                continue
            
            duration = end - start

            if duration < MIN_EMBED_SEGMENT_SEC:
                continue

            regions.append((start, end, duration))

        regions.sort(key=lambda item: item[2], reverse=True)

        selected: list[tuple[float, float, float]] = []
        total_duration = 0.0

        for start, end, duration in regions:
            if len(selected) >= MAX_EMBED_SEGMENTS_PER_SPEAKER:
                break

            if total_duration >= MAX_EMBED_TOTAL_SEC_PER_SPEAKER:
                break

            selected.append((start, end, duration))
            total_duration += duration

        return selected

    def _speaker_embedding_from_diarization(
        self,
        wav_path: Path,
        diarize_df: Any,
        speaker: str,
    ) -> np.ndarray | None:
        """
        Строит centroid speaker embedding для одного diarized speaker.

        Не используем centroids из diarization pipeline.
        Сами берем регионы speaker-а и считаем embeddings прямой speaker embedding model.
        """
        regions = self._select_speaker_regions(
            diarize_df=diarize_df,
            speaker=speaker,
        )

        if not regions:
            return None

        embeddings: list[np.ndarray] = []
        weights: list[float] = []

        for start, end, duration in regions:
            try:
                embedding = self._embed_region(
                    wav_path=wav_path,
                    start=start,
                    end=end,
                )
            except Exception as error:
                self.logger.debug(
                    "Не удалось посчитать speaker embedding: %s %.2f-%.2f, %s",
                    speaker,
                    start,
                    end,
                    error,
                )
                continue

            embeddings.append(embedding)
            weights.append(duration)

        if not embeddings:
            return None

        centroid = np.average(
            np.stack(embeddings, axis=0),
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        )

        return self._normalize_embedding(centroid)

    def _find_target_speaker(
        self,
        wav_path: Path,
        diarize_df: Any,
    ) -> tuple[str, dict[str, float]]:
        """
        Выбирает speaker-кластер, ближайший к reference embedding.

        Threshold пока не используем.
        """
        if self.target_reference_embedding is None:
            raise RuntimeError("Target reference embedding не загружен.")

        speakers = sorted(str(speaker) for speaker in diarize_df["speaker"].dropna().unique())

        if not speakers:
            raise RuntimeError("Diarization не вернула speaker labels.")

        scores: dict[str, float] = {}

        for speaker in speakers:
            speaker_embedding = self._speaker_embedding_from_diarization(
                wav_path=wav_path,
                diarize_df=diarize_df,
                speaker=speaker,
            )

            if speaker_embedding is None:
                continue

            scores[speaker] = float(
                np.dot(self.target_reference_embedding, speaker_embedding)
            )

        if not scores:
            raise RuntimeError("Не удалось посчитать similarity ни для одного speaker.")

        target_speaker = max(scores, key=scores.get)

        return target_speaker, scores

    @staticmethod
    def _mark_target_speaker(diarize_df: Any, source_speaker: str) -> Any:
        """
        Сразу переименовывает speaker-кластер в TARGET_SPEAKER_ID
        до assign_word_speakers.
        """
        diarize_df = diarize_df.copy()
        diarize_df.loc[
            diarize_df["speaker"] == source_speaker,
            "speaker",
        ] = TARGET_SPEAKER_ID

        return diarize_df
    
    def close(self) -> None:
        """
        Освобождает модели WhisperX из памяти.
        """
        self.logger.debug("Выгружаю WhisperX train ASR из памяти")

        self.transcribe_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.speaker_embedder = None
        self.target_reference_embedding = None

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.logger.debug("WhisperX train ASR выгружен")