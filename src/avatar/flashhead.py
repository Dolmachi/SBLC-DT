from __future__ import annotations

import gc
import json
import logging
import math
import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from src.avatar.base import AudioArray, AvatarFrame, StreamingAvatar
from src.utils.output_silencers import patch_print_in_modules


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLASHHEAD_PACKAGE_PARENT = PROJECT_ROOT / "src" / "avatar"

MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_FLASHHEAD_CKPT_DIR = MODELS_ROOT / "flashhead" / "SoulX-FlashHead-1_3B"
DEFAULT_WAV2VEC_DIR = MODELS_ROOT / "flashhead" / "wav2vec2-base-960h"
DEFAULT_INFER_PARAMS_PATH = (
    FLASHHEAD_PACKAGE_PARENT
    / "flash_head"
    / "configs"
    / "infer_params.yaml"
)

# Idle не должен забивать очередь: держим маленький запас кадров
IDLE_MIN_QUEUE_FRAMES = 6
IDLE_QUEUE_FACTOR = 0.5

# Тихий режим
DEFAULT_IDLE_NOISE_AMPLITUDE = 0.003


def make_flashhead_importable() -> None:
    """
    Импортирует SoulX-FlashHead
    """
    package_parent = str(FLASHHEAD_PACKAGE_PARENT)

    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)


def load_infer_params(path: Path = DEFAULT_INFER_PARAMS_PATH) -> dict[str, Any]:
    """
    Загружает FlashHead configs/infer_params.yaml.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден FlashHead infer_params.yaml: {path}\n"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise RuntimeError(f"Некорректный FlashHead infer_params.yaml: {path}")

    return data


@dataclass(frozen=True)
class _SpeechJob:
    """
    Один законченный речевой фрагмент для avatar backend.
    """
    audio: AudioArray
    generation_id: int


class FlashHeadStreamingAvatar(StreamingAvatar):
    """
    Sentence-level streaming adapter для SoulX-FlashHead.
    """

    def __init__(
        self,
        artifacts_avatar_dir: Path,
        logger: logging.Logger,
        ckpt_dir: Path = DEFAULT_FLASHHEAD_CKPT_DIR,
        wav2vec_dir: Path = DEFAULT_WAV2VEC_DIR,
        model_type: str = "lite",
        base_seed: int = 9999,
        compile_model: bool = True,
        compile_vae: bool = True,
        idle_noise_amplitude: float = DEFAULT_IDLE_NOISE_AMPLITUDE,
    ) -> None:
        make_flashhead_importable()

        if model_type not in {"lite", "pro", "pretrained"}:
            raise ValueError("model_type должен быть одним из: lite, pro, pretrained")

        if not torch.cuda.is_available():
            raise RuntimeError("SoulX-FlashHead runtime сейчас рассчитан на CUDA GPU.")

        self.artifacts_avatar_dir = Path(artifacts_avatar_dir).resolve()
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.wav2vec_dir = Path(wav2vec_dir).resolve()
        self.model_type = model_type
        self.base_seed = int(base_seed)
        self.compile_model = bool(compile_model)
        self.compile_vae = bool(compile_vae)
        self.idle_noise_amplitude = float(idle_noise_amplitude)
        self.logger = logger

        self.device = torch.device("cuda")

        self.job_queue: queue.Queue[_SpeechJob | None] = queue.Queue()
        self.frame_queue: queue.Queue[AvatarFrame] = queue.Queue(maxsize=96)

        self._stop_event = threading.Event()

        self._speech_thread: threading.Thread | None = None
        self._idle_thread: threading.Thread | None = None

        self._generation_id = 0
        self._generation_lock = threading.Lock()

        self._state_lock = threading.Lock()
        self._speaking = False
        self._speech_end_requested = False
        self._rendering_speech = False

        # Один lock на FlashHead inference
        # speech и idle не должны одновременно лезть в pipeline.generate().
        self._model_lock = threading.Lock()

        self._idle_frame_bgr: NDArray[np.uint8] | None = None
        self._condition_person_name: str | None = None

        # persistent streaming state
        self._audio_dq: deque[float] | None = None
        self._breath_phase = 0.0
        self._initial_latent_motion_frames: torch.Tensor | None = None
        self._latent_motion_frames: torch.Tensor | None = None

        self._load_avatar_artifacts()
        self._load_pipeline()
        self._init_streaming_state()

    @property
    def sample_rate(self) -> int:
        return int(self.infer_params["sample_rate"])

    @property
    def fps(self) -> int:
        return int(self.infer_params["tgt_fps"])

    def idle_frame(self) -> NDArray[np.uint8] | None:
        if self._idle_frame_bgr is None:
            return None

        return self._idle_frame_bgr.copy()

    def start(self) -> None:
        """
        Запускает speech worker и idle worker.
        """
        if self._speech_thread is not None:
            return

        self._stop_event.clear()

        self._speech_thread = threading.Thread(
            target=self._speech_worker_loop,
            name="FlashHeadSpeechWorker",
            daemon=True,
        )

        self._idle_thread = threading.Thread(
            target=self._idle_worker_loop,
            name="FlashHeadIdleWorker",
            daemon=True,
        )

        self._speech_thread.start()
        self._idle_thread.start()

    def begin_speech(self) -> None:
        """
        Начало новой реплики клона.

        Важно:
        - увеличиваем generation_id, чтобы старые кадры/задачи стали неактуальны;
        - чистим idle кадры из очереди;
        - сбрасываем audio sliding window в тишину, чтобы idle noise не попадал
          в speech embedding;
        - НЕ сбрасываем latent motion state, иначе будет визуальный скачок.
        """
        with self._generation_lock:
            self._generation_id += 1

        with self._state_lock:
            self._speaking = True
            self._speech_end_requested = False

        self._clear_queue(self.job_queue)
        self._clear_queue(self.frame_queue)

        with self._model_lock:
            self._reset_audio_deque_to_silence()

    def end_speech(self) -> None:
        """
        TTS больше не будет добавлять audio jobs.

        Но speech может ещё дорендериваться в speech worker, поэтому idle
        включаем только когда job_queue пустая и render_job завершён.
        """
        with self._state_lock:
            self._speech_end_requested = True

        self._maybe_finish_speech()

    def flush_audio(self) -> None:
        return None

    def push_audio(self, audio: AudioArray, sample_rate: int) -> None:
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if audio.size == 0:
            return

        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
            ).astype(np.float32)

        with self._generation_lock:
            generation_id = self._generation_id

        with self._state_lock:
            self._speaking = True
            self._speech_end_requested = False

        self.job_queue.put(
            _SpeechJob(
                audio=audio.astype(np.float32),
                generation_id=generation_id,
            )
        )

    def read_frame(self, timeout_sec: float = 1.0) -> AvatarFrame | None:
        """
        Читает следующий кадр.

        Если очередь пустая, отдаёт static fallback idle frame + silence.
        Это страховка: живой idle обычно приходит из idle worker, но окно
        не должно зависать даже если idle worker временно занят.
        """
        try:
            return self.frame_queue.get(timeout=timeout_sec)
        except queue.Empty:
            return self._fallback_idle_avatar_frame()

    def stop(self) -> None:
        self._stop_event.set()
        self.job_queue.put(None)

        if self._idle_thread is not None:
            self._idle_thread.join(timeout=5.0)
            self._idle_thread = None

        if self._speech_thread is not None:
            self._speech_thread.join(timeout=5.0)
            self._speech_thread = None

    def warm_up(self, runs: int = 5) -> None:
        """
        Прогревает FlashHead несколькими полноценными idle chunk прогонами.

        Важно:
        - warmup идёт через тот же путь, что idle mode;
        - кадры после warmup не оставляем в очереди;
        - после warmup reset audio deque в тишину.
        """
        runs = max(0, int(runs))

        if runs == 0:
            return

        self.logger.info("Прогреваю FlashHead: %d прогонов", runs)

        with self._model_lock:
            for _index in range(runs):
                self._generate_idle_chunk(enqueue=False)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            self._reset_audio_deque_to_silence()

        self._clear_queue(self.frame_queue)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        self.stop()

        self.pipeline = None
        self._latent_motion_frames = None
        self._initial_latent_motion_frames = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def _load_avatar_artifacts(self) -> None:
        metadata_path = self.artifacts_avatar_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Не найден avatar metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        backend = metadata.get("avatar_backend")
        if backend != "flashhead":
            raise RuntimeError(
                f"Avatar artifacts не для FlashHead: avatar_backend={backend!r}. "
                "Перезапусти train pipeline"
            )

        required_names = [
            "condition_image_name",
            "condition_tensor_name",
            "ref_img_latent_name",
            "person_name",
            "model_type",
            "frame_num",
            "target_height",
            "target_width",
        ]

        missing = [name for name in required_names if name not in metadata]
        if missing:
            raise RuntimeError(
                f"FlashHead avatar metadata не содержит precomputed-полей: {missing}\n"
                "Пересоздай профиль через train pipeline."
            )

        self.avatar_metadata = metadata

        self._condition_person_name = str(metadata["person_name"])

        self.condition_image_path = self.artifacts_avatar_dir / str(
            metadata["condition_image_name"]
        )
        self.condition_tensor_path = self.artifacts_avatar_dir / str(
            metadata["condition_tensor_name"]
        )
        self.ref_img_latent_path = self.artifacts_avatar_dir / str(
            metadata["ref_img_latent_name"]
        )

        if not self.condition_image_path.exists():
            raise FileNotFoundError(f"Не найден FlashHead condition image: {self.condition_image_path}")

        if not self.condition_tensor_path.exists():
            raise FileNotFoundError(f"Не найден FlashHead condition tensor: {self.condition_tensor_path}")

        if not self.ref_img_latent_path.exists():
            raise FileNotFoundError(f"Не найден FlashHead ref image latent: {self.ref_img_latent_path}")

        idle = cv2.imread(str(self.condition_image_path))
        if idle is not None:
            self._idle_frame_bgr = cv2.resize(
                idle,
                (512, 512),
                interpolation=cv2.INTER_AREA,
            )

    def _load_pipeline(self) -> None:
        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Не найдены FlashHead weights: {self.ckpt_dir}")

        if not self.wav2vec_dir.exists():
            raise FileNotFoundError(f"Не найден wav2vec dir: {self.wav2vec_dir}")

        patch_print_in_modules([
            "flash_head.src.pipeline.flash_head_pipeline",
        ])

        import flash_head.src.pipeline.flash_head_pipeline as flashhead_pipeline_module
        from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline

        flashhead_pipeline_module.COMPILE_MODEL = self.compile_model
        flashhead_pipeline_module.COMPILE_VAE = self.compile_vae
        flashhead_pipeline_module.USE_PARALLEL_VAE = False

        self.infer_params = load_infer_params()

        self.model_type = str(self.avatar_metadata["model_type"])
        self.frame_num = int(self.avatar_metadata["frame_num"])

        self.infer_params["height"] = int(self.avatar_metadata["target_height"])
        self.infer_params["width"] = int(self.avatar_metadata["target_width"])
        self.infer_params["frame_num"] = self.frame_num

        self.logger.info("Загружаю SoulX-FlashHead: %s", self.ckpt_dir)

        self.pipeline = FlashHeadPipeline(
            checkpoint_dir=str(self.ckpt_dir),
            model_type=self.model_type,
            wav2vec_dir=str(self.wav2vec_dir),
            device=self.device,
            use_usp=False,
        )

        motion_frames_latent_num = int(self.infer_params["motion_frames_latent_num"])

        self.motion_frames_num = (
            (motion_frames_latent_num - 1) * int(self.pipeline.config.vae_stride[0])
            + 1
        )
        self.infer_params["motion_frames_num"] = self.motion_frames_num

        if self.model_type == "pretrained":
            self.infer_params["sample_steps"] = 20
        else:
            self.infer_params["sample_steps"] = 4

        self.slice_len = self.frame_num - self.motion_frames_num

        if self.slice_len <= 0:
            raise RuntimeError(
                f"Некорректные FlashHead params: frame_num={self.frame_num}, "
                f"motion_frames_num={self.motion_frames_num}"
            )

        self.pipeline.frame_num = self.frame_num
        self.pipeline.motion_frames_num = self.motion_frames_num
        self.pipeline.color_correction_strength = float(
            self.infer_params["color_correction_strength"]
        )

        self.pipeline.target_h = int(self.infer_params["height"])
        self.pipeline.target_w = int(self.infer_params["width"])
        self.pipeline.lat_h = self.pipeline.target_h // int(self.pipeline.config.vae_stride[1])
        self.pipeline.lat_w = self.pipeline.target_w // int(self.pipeline.config.vae_stride[2])

        self.pipeline.generator = torch.Generator(device=self.device).manual_seed(
            self.base_seed
        )
        self.pipeline.timesteps = self._build_flashhead_timesteps(
            sampling_steps=int(self.infer_params["sample_steps"])
        )

        self._apply_precomputed_avatar_params()

        self.samples_per_frame = self.sample_rate // self.fps
        self.samples_per_chunk = self.slice_len * self.sample_rate // self.fps

        self.idle_queue_threshold = max(
            IDLE_MIN_QUEUE_FRAMES,
            int(self.slice_len * IDLE_QUEUE_FACTOR),
        )

        self.logger.info(
            "SoulX-FlashHead загружен с precomputed avatar artifacts: "
            "model_type=%s, frame_num=%d, motion_frames_num=%d, "
            "slice_len=%d, idle_threshold=%d, person=%s",
            self.model_type,
            self.frame_num,
            self.motion_frames_num,
            self.slice_len,
            self.idle_queue_threshold,
            self._condition_person_name,
        )

    def _init_streaming_state(self) -> None:
        cached_audio_duration = int(self.infer_params["cached_audio_duration"])

        self.cached_audio_length_sum = self.sample_rate * cached_audio_duration
        self.audio_end_idx = cached_audio_duration * self.fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num

        self._audio_dq = deque(
            [0.0] * self.cached_audio_length_sum,
            maxlen=self.cached_audio_length_sum,
        )

        self._initial_latent_motion_frames = self.pipeline.latent_motion_frames.clone()
        self._latent_motion_frames = self._initial_latent_motion_frames.clone()

    def _build_flashhead_timesteps(self, sampling_steps: int) -> list[torch.Tensor]:
        from flash_head.src.pipeline.flash_head_pipeline import timestep_transform

        if sampling_steps == 2:
            values = [1000, 500]
        elif sampling_steps == 4:
            values = [1000, 750, 500, 250]
        else:
            values = list(
                np.linspace(
                    self.pipeline.num_timesteps,
                    1,
                    sampling_steps,
                    dtype=np.float32,
                )
            )

        values.append(0.0)

        timesteps = [
            torch.tensor([t], device=self.device)
            for t in values
        ]

        if self.pipeline.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    shift=float(self.infer_params["sample_shift"]),
                    num_timesteps=self.pipeline.num_timesteps,
                )
                for t in timesteps
            ]

        return timesteps

    def _apply_precomputed_avatar_params(self) -> None:
        """
        Загружает FlashHead artifacts.
        """
        condition_tensor = torch.load(
            self.condition_tensor_path,
            map_location="cpu",
            weights_only=True,
        )
        ref_img_latent = torch.load(
            self.ref_img_latent_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(condition_tensor, torch.Tensor):
            raise RuntimeError(f"Некорректный condition_tensor.pt: {self.condition_tensor_path}")

        if not isinstance(ref_img_latent, torch.Tensor):
            raise RuntimeError(f"Некорректный ref_img_latent.pt: {self.ref_img_latent_path}")

        condition_tensor = condition_tensor.to(
            device=self.device,
            dtype=self.pipeline.param_dtype,
        )
        ref_img_latent = ref_img_latent.to(
            device=self.device,
            dtype=self.pipeline.param_dtype,
        )

        self.pipeline.cond_image_dict = {
            self._condition_person_name: str(self.condition_image_path)
        }
        self.pipeline.cond_image_tensor_dict = {
            self._condition_person_name: condition_tensor
        }
        self.pipeline.ref_img_latent_dict = {
            self._condition_person_name: ref_img_latent
        }

        self.pipeline.original_color_reference = condition_tensor
        self.pipeline.ref_img_latent = ref_img_latent
        self.pipeline.latent_motion_frames = ref_img_latent[:, :1].clone()
        self.pipeline.person_name = self._condition_person_name

    def _reset_flashhead_motion_state(self) -> None:
        """
        Полный reset motion state.

        В idle-mode это лучше не вызывать на каждую реплику, иначе будет скачок.
        Оставляем метод для debug/full reset.
        """
        if self._condition_person_name is not None:
            self.pipeline.reset_person_name(self._condition_person_name)

        if self._initial_latent_motion_frames is not None:
            self._latent_motion_frames = self._initial_latent_motion_frames.clone()

        self._reset_audio_deque_to_silence()

    def _reset_audio_deque_to_silence(self) -> None:
        self._audio_dq = deque(
            [0.0] * self.cached_audio_length_sum,
            maxlen=self.cached_audio_length_sum,
        )

    def _speech_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                self._maybe_finish_speech()
                continue

            if job is None:
                break

            with self._state_lock:
                self._rendering_speech = True

            try:
                self._render_speech_job(job)
            except Exception:
                self.logger.exception("FlashHead speech render job failed")
            finally:
                with self._state_lock:
                    self._rendering_speech = False

                self._maybe_finish_speech()

    def _render_speech_job(self, job: _SpeechJob) -> None:
        audio = self._pad_audio_to_chunks(job.audio)

        chunks = audio.reshape(-1, self.samples_per_chunk)

        for chunk_idx, audio_chunk in enumerate(chunks):
            if self._stop_event.is_set() or not self._is_current_generation(job.generation_id):
                return

            with self._model_lock:
                frames_rgb = self._generate_chunk_from_audio(
                    audio_chunk=audio_chunk,
                    update_audio_deque=True,
                    update_motion_state=True,
                )

            frames_rgb = frames_rgb[self.motion_frames_num:]
            chunk_start_sample = chunk_idx * self.samples_per_chunk

            self._enqueue_speech_frames(
                frames_rgb=frames_rgb,
                audio=audio,
                chunk_start_sample=chunk_start_sample,
                generation_id=job.generation_id,
            )

    def _idle_worker_loop(self) -> None:
        self.logger.debug(
            "FlashHead idle worker started: threshold=%d",
            self.idle_queue_threshold,
        )

        while not self._stop_event.is_set():
            if not self._need_idle_chunk():
                self._stop_event.wait(timeout=0.05)
                continue

            try:
                with self._model_lock:
                    if not self._need_idle_chunk():
                        continue

                    self._generate_idle_chunk(enqueue=True)

            except Exception:
                self.logger.exception("FlashHead idle generation failed")
                self._stop_event.wait(timeout=0.2)

        self.logger.debug("FlashHead idle worker stopped")

    def _need_idle_chunk(self) -> bool:
        if self.frame_queue.qsize() >= self.idle_queue_threshold:
            return False

        with self._state_lock:
            if self._speaking or self._rendering_speech:
                return False

        return True

    def _maybe_finish_speech(self) -> None:
        with self._state_lock:
            if not self._speech_end_requested:
                return

            if self._rendering_speech:
                return

            if not self.job_queue.empty():
                return

            self._speaking = False
            self._speech_end_requested = False

    def _make_ambient_noise(self, n_samples: int) -> NDArray[np.float32]:
        """
        Генерирует слабый дыхательный noise для idle animation.
        """
        if self.idle_noise_amplitude <= 0:
            return np.zeros(n_samples, dtype=np.float32)

        sr = self.sample_rate
        t = np.arange(n_samples, dtype=np.float32) / sr

        # ~15 вдохов в минуту: один цикл около 4 секунд.
        breath_freq = 0.25
        phase = self._breath_phase

        raw_envelope = np.sin(2 * np.pi * breath_freq * t + phase)
        envelope = np.clip(raw_envelope, 0.0, 1.0) ** 0.5

        self._breath_phase = phase + 2 * np.pi * breath_freq * n_samples / sr

        noise = np.random.randn(n_samples).astype(np.float32)

        # Мягкий low-pass через moving average, чтобы не было резкого белого шума.
        kernel_size = max(1, sr // 2000)
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        noise = np.convolve(noise, kernel, mode="same").astype(np.float32)

        return (noise * envelope * self.idle_noise_amplitude).astype(np.float32)

    def _generate_idle_chunk(self, enqueue: bool) -> None:
        idle_audio = self._make_ambient_noise(self.samples_per_chunk)

        frames_rgb = self._generate_chunk_from_audio(
            audio_chunk=idle_audio,
            update_audio_deque=True,
            update_motion_state=True,
        )

        frames_rgb = frames_rgb[self.motion_frames_num:]

        if not enqueue:
            return

        self._enqueue_idle_frames(frames_rgb)

    def _generate_chunk_from_audio(
        self,
        audio_chunk: AudioArray,
        update_audio_deque: bool,
        update_motion_state: bool,
    ) -> NDArray[np.uint8]:
        if self._audio_dq is None:
            raise RuntimeError("FlashHead audio deque не инициализирован.")

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()

        if update_audio_deque:
            self._audio_dq.extend(audio_chunk.tolist())

        audio_array = np.asarray(self._audio_dq, dtype=np.float32)

        audio_embedding = self._get_audio_embedding(audio_array)

        if update_motion_state and self._latent_motion_frames is not None:
            self.pipeline.latent_motion_frames = self._latent_motion_frames.clone()

        frames_rgb = self._run_pipeline(audio_embedding)

        if update_motion_state:
            self._latent_motion_frames = self.pipeline.latent_motion_frames.clone()

        return frames_rgb

    def _get_audio_embedding(self, audio_array: AudioArray) -> torch.Tensor:
        audio_embedding = self.pipeline.preprocess_audio(
            audio_array,
            sr=self.sample_rate,
            fps=self.fps,
        )

        if audio_embedding is None:
            raise RuntimeError("FlashHead не смог извлечь audio embedding")

        indices = (torch.arange(2 * 2 + 1) - 2) * 1

        center_indices = (
            torch.arange(self.audio_start_idx, self.audio_end_idx, 1).unsqueeze(1)
            + indices.unsqueeze(0)
        )
        center_indices = torch.clamp(
            center_indices,
            min=0,
            max=self.audio_end_idx - 1,
        )

        audio_embedding = audio_embedding[center_indices][None, ...].contiguous()

        return audio_embedding

    @torch.inference_mode()
    def _run_pipeline(self, audio_embedding: torch.Tensor) -> NDArray[np.uint8]:
        audio_embedding = audio_embedding.to(self.pipeline.device)

        sample = self.pipeline.generate(audio_embedding)

        frames = (
            ((sample + 1) / 2)
            .permute(1, 2, 3, 0)
            .clip(0, 1)
            * 255
        ).contiguous()

        frames_np = frames.detach().cpu().numpy().astype(np.uint8)

        del sample
        del frames

        return frames_np

    def _enqueue_speech_frames(
        self,
        frames_rgb: NDArray[np.uint8],
        audio: AudioArray,
        chunk_start_sample: int,
        generation_id: int,
    ) -> None:
        frame_count = int(frames_rgb.shape[0])

        for frame_idx in range(frame_count):
            if self._stop_event.is_set() or not self._is_current_generation(generation_id):
                return

            global_start = chunk_start_sample + frame_idx * self.samples_per_frame
            global_end = global_start + self.samples_per_frame

            frame_audio = audio[global_start:global_end]
            if frame_audio.size < self.samples_per_frame:
                frame_audio = np.pad(
                    frame_audio,
                    (0, self.samples_per_frame - frame_audio.size),
                )

            frame_rgb = frames_rgb[frame_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            avatar_frame = AvatarFrame(
                frame=frame_bgr,
                audio_chunks=[frame_audio.astype(np.float32)],
                sample_rate=self.sample_rate,
            )

            while not self._stop_event.is_set() and self._is_current_generation(generation_id):
                try:
                    self.frame_queue.put(avatar_frame, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def _enqueue_idle_frames(self, frames_rgb: NDArray[np.uint8]) -> None:
        silence = np.zeros(self.samples_per_frame, dtype=np.float32)

        for frame_idx in range(int(frames_rgb.shape[0])):
            if self._stop_event.is_set() or not self._need_idle_chunk():
                return

            frame_rgb = frames_rgb[frame_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            avatar_frame = AvatarFrame(
                frame=frame_bgr,
                audio_chunks=[silence],
                sample_rate=self.sample_rate,
            )

            try:
                self.frame_queue.put(avatar_frame, timeout=0.05)
            except queue.Full:
                return

    def _fallback_idle_avatar_frame(self) -> AvatarFrame | None:
        if self._idle_frame_bgr is None:
            return None

        return AvatarFrame(
            frame=self._idle_frame_bgr.copy(),
            audio_chunks=[np.zeros(self.samples_per_frame, dtype=np.float32)],
            sample_rate=self.sample_rate,
        )

    def _pad_audio_to_chunks(self, audio: AudioArray) -> AudioArray:
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if audio.size == 0:
            audio = np.zeros(self.samples_per_chunk, dtype=np.float32)

        num_chunks = max(1, int(math.ceil(audio.size / self.samples_per_chunk)))
        target_len = num_chunks * self.samples_per_chunk

        if audio.size < target_len:
            audio = np.pad(audio, (0, target_len - audio.size))
        elif audio.size > target_len:
            audio = audio[:target_len]

        return audio.astype(np.float32)

    def _is_current_generation(self, generation_id: int) -> bool:
        with self._generation_lock:
            return generation_id == self._generation_id