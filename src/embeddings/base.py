from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class TextEmbedder(ABC):
    """
    Базовый интерфейс embedder'а.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def as_langchain_embeddings(self) -> Embeddings:
        """
        Возвращает объект embeddings, совместимый с LangChain.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Освобождает ресурсы embedder'а.
        """
        return None