from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


AudioArray = NDArray[np.float32]


class TTS(ABC):
    """
    Базовый интерфейс TTS
    """
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """
        Частота дискретизации выходного аудио
        """
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, text: str) -> AudioArray:
        """
        Синтезирует полный аудиосигнал для текста.
        """
        raise NotImplementedError
    
    def warm_up(self, runs: int = 5) -> None:
        """
        Прогрев TTS
        """
        return None

    def close(self) -> None:
        """
        Освобождает ресурсы от TTS
        """
        return None