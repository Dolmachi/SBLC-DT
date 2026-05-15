from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLM(ABC):
    """
    Базовый интерфейс LLM-модуля виртуального клона.
    """

    @abstractmethod
    def ask(
        self,
        user_text: str,
        thread_id: str,
    ) -> str:
        """
        Возвращает полный текстовый ответ.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        user_text: str,
        thread_id: str,
    ) -> Iterator[str]:
        """
        Стримит ответ текстовыми delta/chunk.
        """
        raise NotImplementedError
    
    def warm_up(self, runs: int = 5) -> None:
        """
        Прогрев LLM/RAG
        """
        return None

    def close(self) -> None:
        """
        Освобождает ресурсы LLM.
        """
        return None