from __future__ import annotations

import logging
from pathlib import Path

from src.training.context import TrainingContext, build_training_context
from src.training.stages import (
    asr_ingest,
    asr_postprocess,
    dialog_pairs,
    init_profile,
    preprocess_dialogs,
    rag_build,
    target_segments,
    tts_dataset_build,
    tts_prepare,
    avatar_prepare
)
from src.utils.app_settings import require_hf_token


class TrainingPipeline:
    """
    Главный train pipeline.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def run(self, name: str, lang: str, data_path: Path) -> Path:
        """
        Запускает полный train pipeline.
        """
        self.logger.info("Запуск train pipeline")

        profile_dir = self._run_init_profile(name, lang, data_path)
        ctx = build_training_context(profile_dir)

        self._run_preprocess_dialogs(ctx)
        self._run_asr_ingest(ctx)
        self._run_asr_postprocess(ctx)
        self._run_dialog_pairs(ctx)
        self._run_rag_build(ctx)
        self._run_target_segments(ctx)
        self._run_tts_dataset_build(ctx)
        self._run_tts_prepare(ctx)
        self._run_avatar_prepare(ctx)
        
        self.logger.info("Train pipeline успешно завершён")
        self.logger.info("Профиль: %s", profile_dir)

        return profile_dir


    def _run_init_profile(self, name: str, lang: str, data_path: Path) -> Path:
        """
        Stage 00: создание профиля.
        """
        self.logger.info("=== Stage 00: init_profile ===")

        profile_dir = init_profile.run(
            name=name,
            lang=lang,
            data_path=data_path,
            logger=self.logger,
        )

        return profile_dir

    def _run_preprocess_dialogs(self, ctx: TrainingContext) -> None:
        """
        Stage 01: подготовка аудио dialogs.
        """
        self.logger.info("=== Stage 01: preprocess_dialogs ===")

        preprocess_dialogs.run(
            ctx=ctx,
            logger=self.logger,
        )

    def _run_asr_ingest(self, ctx: TrainingContext) -> None:
        """
        Stage 02: ASR ingest.
        """
        self.logger.info("=== Stage 02: asr_ingest ===")

        hf_token = require_hf_token()

        from src.asr.whisperx_train_asr import WhisperXASR

        train_asr = WhisperXASR(
            language=ctx.cfg.lang,
            hf_token=hf_token,
            logger=self.logger,
        )

        try:
            asr_ingest.run(
                ctx=ctx,
                asr_engine=train_asr,
                logger=self.logger,
            )
        finally:
            train_asr.close()
            
    def _run_asr_postprocess(self, ctx: TrainingContext) -> None:
        """
        Stage 03: постобработка ASR JSON.
        """
        self.logger.info("=== Stage 03: asr_postprocess ===")

        asr_postprocess.run(
            ctx=ctx,
            logger=self.logger,
        )
        
    def _run_dialog_pairs(self, ctx: TrainingContext) -> None:
        """
        Stage 04: формирование dialog pairs.
        """
        self.logger.info("=== Stage 04: dialog_pairs ===")

        dialog_pairs.run(
            ctx=ctx,
            logger=self.logger,
        )
        
    def _run_rag_build(self, ctx: TrainingContext) -> None:
        """
        Stage 05: построение RAG-базы.
        """
        self.logger.info("=== Stage 05: rag_build ===")

        from src.embeddings.huggingface_embedder import HuggingFaceTextEmbedder

        embedder = HuggingFaceTextEmbedder(
            model_id=ctx.cfg.embedding_model_id,
            device="cuda",
            use_bf16=False, # инференс будет на CPU (fp32)
            logger=self.logger,
        )

        try:
            rag_build.run(
                ctx=ctx,
                embedder=embedder,
                logger=self.logger,
            )
        finally:
            embedder.close()
            
    def _run_target_segments(self, ctx: TrainingContext) -> None:
        """
        Stage 06: выбор target-сегментов для voice cloning.
        """
        self.logger.info("=== Stage 06: target_segments ===")

        target_segments.run(
            ctx=ctx,
            logger=self.logger,
        )
        
    def _run_tts_dataset_build(self, ctx: TrainingContext) -> None:
        """
        Stage 07: построение TTS dataset.
        """
        self.logger.info("=== Stage 07: tts_dataset_build ===")

        tts_dataset_build.run(
            ctx=ctx,
            logger=self.logger,
        )
        
    def _run_tts_prepare(self, ctx: TrainingContext) -> None:
        """
        Stage 08: подготовка TTS artifacts.
        """
        self.logger.info("=== Stage 08: tts_prepare ===")

        tts_prepare.run(
            ctx=ctx,
            logger=self.logger,
        )
        
    def _run_avatar_prepare(self, ctx: TrainingContext) -> None:
        """
        Stage 09: подготовка avatar artifacts.
        """
        self.logger.info("=== Stage 09: avatar_prepare ===")

        avatar_prepare.run(
            ctx=ctx,
            logger=self.logger,
        )