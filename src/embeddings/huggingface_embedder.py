from __future__ import annotations

import gc
import logging
from typing import Any

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.embeddings.base import TextEmbedder
from src.embeddings.registry import get_embedder_kwargs


class HuggingFaceTextEmbedder(TextEmbedder):
    """
    Embedder на базе Hugging Face.
    """
    def __init__(
        self,
        model_id: str,
        logger: logging.Logger,
        device: str | None = None,
        use_bf16: bool = True,
    ) -> None:
        self._model_id = model_id
        self.logger = logger
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = use_bf16
        self._embeddings: Embeddings | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def as_langchain_embeddings(self) -> Embeddings:
        """
        Лениво загружает embedding-модель и возвращает LangChain-compatible объект.
        """
        if self._embeddings is not None:
            return self._embeddings

        self.logger.debug("Загружаю embedder: %s", self.model_id)
        self.logger.debug("Embedder device: %s", self.device)

        extra_kwargs = get_embedder_kwargs(self.model_id)

        model_kwargs: dict[str, Any] = {
            "device": self.device,
        }

        if extra_kwargs.get("trust_remote_code"):
            model_kwargs["trust_remote_code"] = True

        if self.use_bf16 and self.device == "cuda":
            model_kwargs["model_kwargs"] = {
                "torch_dtype": torch.bfloat16,
            }

        encode_kwargs = {
            "normalize_embeddings": True,
            **extra_kwargs.get("encode_kwargs", {}),
        }

        kwargs: dict[str, Any] = {
            "model_name": self.model_id,
            "model_kwargs": model_kwargs,
            "encode_kwargs": encode_kwargs,
        }

        if "query_encode_kwargs" in extra_kwargs:
            kwargs["query_encode_kwargs"] = {
                "normalize_embeddings": True,
                **extra_kwargs["query_encode_kwargs"],
            }

        self._embeddings = HuggingFaceEmbeddings(**kwargs)

        self.logger.debug("Embedder загружен: %s", self.model_id)
        return self._embeddings

    def close(self) -> None:
        """
        Освобождает embedding-модель из памяти.
        """
        if self._embeddings is None:
            return

        self.logger.debug("Выгружаю embedder из памяти: %s", self.model_id)

        self._embeddings = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.logger.debug("Embedder выгружен")