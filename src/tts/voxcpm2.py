from __future__ import annotations

import gc
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tts.base import TTS
from src.utils.output_silencers import (
    patch_tqdm_in_modules,
    suppress_stdout_stderr,
)


AudioArray = NDArray[np.float32]


class VoxCPM2TTS(TTS):
    """
    Runtime TTS engine на базе VoxCPM2.
    """

    def __init__(
        self,
        model_id: str,
        lora_weights_path: Path,
        reference_wav_path: Path,
        logger: logging.Logger,
        load_denoiser: bool = False,
        optimize: bool = False,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = False,
    ) -> None:
        self.model_id = model_id
        self.lora_weights_path = Path(lora_weights_path).resolve()
        self.logger = logger

        self.load_denoiser = load_denoiser
        self.optimize = optimize
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.normalize = normalize
        self.denoise = denoise
        self.retry_badcase = retry_badcase
        
        self.reference_wav_path = Path(reference_wav_path).resolve()

        if not self.lora_weights_path.exists():
            raise FileNotFoundError(
                f"Не найден VoxCPM2 LoRA checkpoint: {self.lora_weights_path}"
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
        optimize: bool = False,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = False,
    ) -> "VoxCPM2TTS":
        """
        Создаёт VoxCPM2TTS из artifacts/tts профиля.
        """
        artifacts_tts_dir = Path(artifacts_tts_dir).resolve()
        lora_latest_dir = artifacts_tts_dir / "lora" / "latest"
        reference_wav_path = artifacts_tts_dir / "reference.wav"

        return cls(
            model_id=model_id,
            lora_weights_path=lora_latest_dir,
            reference_wav_path=reference_wav_path,
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
            cfg_value=self.cfg_value,
            inference_timesteps=self.inference_timesteps,
            normalize=self.normalize,
            denoise=self.denoise,
            retry_badcase=self.retry_badcase,
        )

        return np.asarray(wav, dtype=np.float32)
    
    def warm_up(self, runs: int = 5) -> None:
        """
        Ручной warmup VoxCPM2.
        """
        runs = max(0, int(runs))

        if runs == 0:
            return

        self.logger.info("Прогреваю VoxCPM2 TTS: %d прогонов", runs)

        warmup_text = "Это короткий прогрев синтеза речи для виртуального клона."

        for _index in range(runs):
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

    def _load_lora_config(self) -> Any:
        """
        Загружает LoRAConfig с теми же параметрами, с которыми LoRA обучалась.
        """
        config_path = self.lora_weights_path / "lora_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Не найден lora_config.json: {config_path}")

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cfg_payload = payload.get("lora_config", payload)

        from voxcpm.model.voxcpm2 import LoRAConfig

        return LoRAConfig(**cfg_payload)
    
    def _load_model(self) -> Any:
        """
        Загружает VoxCPM2 + LoRA.
        """
        self.logger.info("Загружаю VoxCPM2 TTS: %s", self.model_id)
        self.logger.info("Загружаю VoxCPM2 LoRA: %s", self.lora_weights_path)

        from voxcpm import VoxCPM

        lora_config = self._load_lora_config()

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
                    lora_config=lora_config,
                    lora_weights_path=str(self.lora_weights_path),
                )

        patch_tqdm_in_modules(
            [
                "voxcpm.model.voxcpm",
                "voxcpm.model.voxcpm2",
            ]
        )

        self.logger.info("VoxCPM2 TTS загружен")

        return model