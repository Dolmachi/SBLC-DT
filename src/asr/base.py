from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


ASRJson = dict[str, Any]


class TrainASR(ABC):
    """
    Базовый интерфейс ASR для train.
    """

    @abstractmethod
    def transcribe_file(self, wav_path: Path) -> ASRJson | None:
        """
        Распознаёт один wav-файл и возвращает ASR JSON.

        Возвращает None, если файл не прошёл внутреннюю проверку качества
        ASR/диаризации.
        """
        raise NotImplementedError
    
    def close(self) -> None:
        """
        Освобождает ресурсы train ASR.
        """
        return None


class RuntimeASR(ABC):
    """
    Базовый интерфейс ASR для inference.
    """

    @abstractmethod
    def transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> str:
        """
        Распознаёт аудио с микрофона и возвращает текст.
        """
        raise NotImplementedError
    
    def close(self) -> None:
        """
        Освобождает ресурсы inference ASR.
        """
        return None