from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tts.base import TTS

from src.utils.output_silencers import patch_tqdm_in_modules, suppress_stdout_stderr

import warnings


AudioArray = NDArray[np.float32]


class VoxCPM2TTS(TTS):
    """
    Runtime TTS engine на базе VoxCPM2.
    """

    def __init__(
        self,
        model_id: str,
        reference_wav_path: Path,
        reference_text: str,
        logger: logging.Logger,
        load_denoiser: bool = False,
        optimize: bool = True,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
    ) -> None:
        self.model_id = model_id
        self.reference_wav_path = Path(reference_wav_path).resolve()
        self.reference_text = reference_text.strip()
        self.logger = logger

        self.load_denoiser = load_denoiser
        self.optimize = optimize

        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.normalize = normalize
        self.denoise = denoise
        self.retry_badcase = retry_badcase

        if not self.reference_wav_path.exists():
            raise FileNotFoundError(
                f"Не найден VoxCPM2 reference wav: {self.reference_wav_path}"
            )

        if not self.reference_text:
            raise RuntimeError(
                f"Пустой reference text для VoxCPM2: {self.reference_wav_path}"
            )

        self.model = self._load_model()
        self._sample_rate = int(self.model.tts_model.sample_rate)

    @classmethod
    def from_artifacts(
        cls,
        artifacts_tts_dir: Path,
        logger: logging.Logger,
        model_id: str = "openbmb/VoxCPM2",
        load_denoiser: bool = False,
        optimize: bool = True,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
    ) -> "VoxCPM2TTS":
        """
        Создаёт VoxCPM2TTS из artifacts/tts профиля.
        """
        artifacts_tts_dir = Path(artifacts_tts_dir).resolve()

        reference_wav_path = artifacts_tts_dir / "reference.wav"
        reference_text_path = artifacts_tts_dir / "reference.txt"

        if not reference_text_path.exists():
            raise FileNotFoundError(
                f"Не найден VoxCPM2 reference text: {reference_text_path}"
            )

        reference_text = reference_text_path.read_text(encoding="utf-8").strip()

        return cls(
            model_id=model_id,
            reference_wav_path=reference_wav_path,
            reference_text=reference_text,
            logger=logger,
            load_denoiser=load_denoiser,
            optimize=optimize,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
        )

    @property
    def sample_rate(self) -> int:
        """
        Частота дискретизации выходного аудио.
        """
        return self._sample_rate

    def synthesize(self, text: str) -> AudioArray:
        """
        Синтезирует полный wav для текста.
        """
        wav = self.model.generate(
            text=text.strip(),
            reference_wav_path=str(self.reference_wav_path),
            prompt_wav_path=str(self.reference_wav_path),
            prompt_text=self.reference_text,
            cfg_value=self.cfg_value,
            inference_timesteps=self.inference_timesteps,
            normalize=self.normalize,
            denoise=self.denoise,
            retry_badcase=self.retry_badcase,
        )

        return np.asarray(wav, dtype=np.float32)
    
    def warm_up(self, runs: int = 5) -> None:
        """
        Ручной warmup VoxCPM2
        """
        runs = max(0, int(runs))

        if runs == 0:
            return

        self.logger.info("Прогреваю VoxCPM2 TTS: %d прогонов", runs)

        warmup_text = "Это короткий прогрев синтеза речи для виртуального клона."

        for index in range(runs):
            _wav = self.synthesize(warmup_text)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def close(self) -> None:
        """
        Выгружает VoxCPM2 из памяти.
        """
        self.logger.debug("Выгружаю VoxCPM2 TTS из памяти")

        self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.logger.debug("VoxCPM2 TTS выгружен")

    def _load_model(self) -> Any:
        """
        Загружает VoxCPM2
        """
        self.logger.info("Загружаю VoxCPM2 TTS: %s", self.model_id)

        from voxcpm import VoxCPM

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r".*torch\.nn\.utils\.weight_norm.*",
            )

            with suppress_stdout_stderr():
                model = VoxCPM.from_pretrained(
                    self.model_id,
                    load_denoiser=self.load_denoiser,
                    optimize=self.optimize,
                )

        patch_tqdm_in_modules([
            "voxcpm.model.voxcpm",
            "voxcpm.model.voxcpm2",
        ])

        self.logger.info("VoxCPM2 TTS загружен")

        return model