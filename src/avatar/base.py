from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


AudioArray = NDArray[np.float32]
VideoFrame = NDArray[np.uint8]


@dataclass(frozen=True)
class AvatarFrame:
    """
    Один синхронизированный output аватара.

    frame:
        BGR кадр для cv2.imshow.

    audio_chunks:
        Аудио, которое нужно проиграть вместе с этим video frame.
        Обычно это один или несколько коротких аудиофрагментов,
        соответствующих текущему кадру. Конкретная нарезка зависит
        от backend'а аватара.
    """
    frame: VideoFrame
    audio_chunks: list[AudioArray]
    sample_rate: int


class StreamingAvatar(ABC):
    """
    Базовый интерфейс avatar engine.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def fps(self) -> int:
        raise NotImplementedError
    
    def idle_frame(self) -> VideoFrame | None:
        """
        Стартовый/ожидающий кадр для preview-окна.
        """
        return None

    def begin_speech(self) -> None:
        return None

    def end_speech(self) -> None:
        return None

    def flush_audio(self) -> None:
        return None
    
    def warm_up(self, runs: int = 5) -> None:
        """
        Прогрев avatar
        """
        return None

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def push_audio(self, audio: AudioArray, sample_rate: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_frame(self, timeout_sec: float = 1.0) -> AvatarFrame | None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.stop()
