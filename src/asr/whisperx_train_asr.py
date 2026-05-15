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
from whisperx.diarize import DiarizationPipeline

from src.asr.base import ASRJson, TrainASR


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

        self.logger.debug("WhisperX train ASR загружен")

    def transcribe_file(self, wav_path: Path) -> ASRJson | None:
        """
        Выполняет полный train-ASR цикл:
        1. transcription;
        2. alignment;
        3. diarization;
        4. assign word speakers.
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
            min_speakers=2,
        )

        if not self._diarization_is_ok(diarize_segments):
            self.logger.warning(
                "Диаризация не прошла фильтр качества: %s",
                wav_path.name,
            )
            return None

        result = whisperx.assign_word_speakers(
            diarize_segments,
            aligned,
        )

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
    
    def close(self) -> None:
        """
        Освобождает модели WhisperX из памяти.
        """
        self.logger.debug("Выгружаю WhisperX train ASR из памяти")

        self.transcribe_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.logger.debug("WhisperX train ASR выгружен")