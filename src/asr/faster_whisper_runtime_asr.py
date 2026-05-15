from __future__ import annotations

import gc
import logging
import librosa

import numpy as np
import torch
from faster_whisper import WhisperModel
from numpy.typing import NDArray

from src.asr.base import RuntimeASR


AudioArray = NDArray[np.float32]


class FasterWhisperASR(RuntimeASR):
    """
    Runtime ASR для inference pipeline
    """

    def __init__(
        self,
        language: str,
        logger: logging.Logger,
        model_size: str = "large-v3-turbo",
        sample_rate: int = 16000,
        device: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self.language = language
        self.logger = logger
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        compute_type = "float16" if self.device == "cuda" else "int8"

        self.logger.debug("Загружаю runtime ASR: faster-whisper %s", self.model_size)

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute_type,
        )

        self.logger.debug("Runtime ASR загружен")

    def transcribe(
        self,
        audio: AudioArray,
        sample_rate: int,
    ) -> str:
        """
        Распознаёт audio array и возвращает текст.
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if audio.size == 0:
            return ""

        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
            ).astype(np.float32)

        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        parts = [seg.text.strip() for seg in segments]

        return " ".join(parts).strip()

    def close(self) -> None:
        """
        Выгружает faster-whisper из памяти.
        """
        self.logger.debug("Выгружаю runtime ASR")

        self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.logger.debug("Runtime ASR выгружен")
