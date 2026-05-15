from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray


AudioArray = NDArray[np.float32]


@dataclass(frozen=True)
class RecordedAudio:
    """
    Результат записи с микрофона.
    """
    audio: AudioArray
    sample_rate: int


class SoundDeviceAudioIO:
    """
    Audio I/O через sounddevice.
    """

    def __init__(
        self,
        logger: logging.Logger,
        input_sample_rate: int = 16000,
    ) -> None:
        self.logger = logger
        self.input_sample_rate = input_sample_rate

    def record_until_enter(self) -> RecordedAudio:
        """
        Записывает аудио с микрофона до нажатия Enter.
        """
        frames: list[AudioArray] = []

        def callback(indata, frame_count, time_info, status):  # type: ignore[no-untyped-def]
            if status:
                self.logger.warning("Mic status: %s", status)

            frames.append(indata.copy())

        print("[Микрофон] Запись... Нажми Enter, чтобы остановить.")

        with sd.InputStream(
            samplerate=self.input_sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            input()

        if not frames:
            return RecordedAudio(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self.input_sample_rate,
            )

        audio = np.concatenate(frames, axis=0).flatten()

        return RecordedAudio(
            audio=audio,
            sample_rate=self.input_sample_rate,
        )