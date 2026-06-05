from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.asr.base import RuntimeASR
from src.asr.faster_whisper_runtime_asr import FasterWhisperASR
from src.rag.embeddings.base import TextEmbedder
from src.rag.embeddings.huggingface_embedder import HuggingFaceTextEmbedder
from src.llm.base import LLM
from src.llm.qwen3_5 import QwenCloneLLM
from src.tts.base import TTS
from src.tts.voxcpm2 import VoxCPM2TTS
from src.avatar.base import StreamingAvatar
from src.avatar.flashhead import FlashHeadStreamingAvatar
from src.inference.context import InferenceContext, build_inference_context


AudioArray = NDArray[np.float32]
DEFAULT_THREAD_ID = "default_clone_thread"


class CloneInferencePipeline:
    """
    ML pipeline инференса.

    Связывает модели:
        ASR -> LLM/RAG -> TTS -> Avatar
    """

    def __init__(
        self,
        ctx: InferenceContext,
        asr: RuntimeASR,
        embedder: TextEmbedder,
        llm: LLM,
        tts: TTS,
        avatar: StreamingAvatar,
        logger: logging.Logger,
    ) -> None:
        self.ctx = ctx
        self.asr = asr
        self.embedder = embedder
        self.llm = llm
        self.tts = tts
        self.avatar = avatar
        self.logger = logger

    @classmethod
    def load(
        cls,
        profile_dir: Path,
        logger: logging.Logger,
        llm_model_id: str = "Qwen/Qwen3.5-4B",
        asr_model_size: str = "large-v3-turbo",
        tts_model_id: str = "openbmb/VoxCPM2",
    ) -> "CloneInferencePipeline":
        logger.info("Загружаю inference pipeline")
        logger.info("Профиль: %s", profile_dir)

        ctx = build_inference_context(profile_dir)

        asr = FasterWhisperASR(
            language=ctx.cfg.lang,
            model_size=asr_model_size,
            logger=logger,
        )
        
        embedder = HuggingFaceTextEmbedder(
            model_id=ctx.cfg.embedding_model_id,
            device="cpu",
            use_bf16=False,
            logger=logger,
        )

        llm = QwenCloneLLM(
            profile_dir=ctx.paths.profile_dir,
            model_id=llm_model_id,
            embeddings=embedder.as_langchain_embeddings(),
            logger=logger,
        )

        tts = VoxCPM2TTS.from_artifacts(
            artifacts_tts_dir=ctx.paths.artifacts_tts_dir,
            model_id=tts_model_id,
            logger=logger,
            load_denoiser=False,
            optimize=False,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=False,
            denoise=False,
            retry_badcase=False,
        )

        avatar = FlashHeadStreamingAvatar(
            artifacts_avatar_dir=ctx.paths.artifacts_avatar_dir,
            logger=logger,
            model_type="lite",
            compile_model=True,
            compile_vae=True,
            idle_noise_amplitude=0.003,
        )

        logger.info("Inference pipeline загружен")

        pipeline = cls(
            ctx=ctx,
            asr=asr,
            embedder=embedder,
            llm=llm,
            tts=tts,
            avatar=avatar,
            logger=logger,
        )
        
        pipeline.warm_up(
            llm_runs=5,
            tts_runs=5,
            avatar_runs=5,
        )
        
        logger.info("Inference pipeline готов к работе")
        
        return pipeline
        

    def transcribe_audio(self, audio: AudioArray, sample_rate: int) -> str:
        return self.asr.transcribe(audio=audio, sample_rate=sample_rate)

    def stream_answer_text(self, user_text: str, thread_id: str = DEFAULT_THREAD_ID):
        user_text = self.require_non_empty_text(
            text=user_text,
            error_message="Пустой user_text передан в LLM.",
        )
        yield from self.llm.stream(user_text=user_text, thread_id=thread_id)

    def synthesize_tts_audio(self, text: str) -> AudioArray:
        """
        Text unit -> полный wav.
        """
        text = self.require_non_empty_text(
            text=text,
            error_message="Пустой текст передан в TTS.",
        )
        return self.tts.synthesize(text)
    
    def warm_up(
        self,
        llm_runs: int = 5,
        tts_runs: int = 5,
        avatar_runs: int = 5,
    ) -> None:
        """
        Единая стадия прогрева inference pipeline
        """
        self.logger.info("Прогреваю inference pipeline")

        self.llm.warm_up(runs=llm_runs)
        self.tts.warm_up(runs=tts_runs)
        self.avatar.warm_up(runs=avatar_runs)

        self.logger.info("Inference pipeline прогрет")

    def close(self) -> None:
        self.logger.info("Закрываю inference pipeline")

        self.asr.close()
        self.llm.close()
        self.embedder.close()
        self.tts.close()
        self.avatar.close()

        self.logger.info("Inference pipeline закрыт")

    @staticmethod
    def require_non_empty_text(text: str, error_message: str) -> str:
        normalized = text.strip()
        if not normalized:
            raise RuntimeError(error_message)
        return normalized
