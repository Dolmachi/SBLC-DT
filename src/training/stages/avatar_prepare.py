from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from src.training.context import TrainingContext
from src.utils.fs import ensure_dir, reset_dir


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FLASHHEAD_PACKAGE_PARENT = PROJECT_ROOT / "src" / "avatar"

MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_FLASHHEAD_CKPT_DIR = MODELS_ROOT / "flashhead" / "SoulX-FlashHead-1_3B"
DEFAULT_INFER_PARAMS_PATH = (
    FLASHHEAD_PACKAGE_PARENT
    / "flash_head"
    / "configs"
    / "infer_params.yaml"
)

AVATAR_BACKEND = "flashhead"
AVATAR_VERSION = "soulx_flashhead_1_3b_lite_precomputed_ref_latent"

MODEL_TYPE = "lite"

CONDITION_SOURCE_NAME = "condition_source.png"
CONDITION_IMAGE_NAME = "condition.png"
CONDITION_TENSOR_NAME = "condition_tensor.pt"
REF_IMG_LATENT_NAME = "ref_img_latent.pt"

PERSON_NAME = "condition"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_flashhead_importable() -> None:
    package_parent = str(FLASHHEAD_PACKAGE_PARENT)

    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)


def load_infer_params(path: Path = DEFAULT_INFER_PARAMS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден FlashHead infer_params.yaml: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise RuntimeError(f"Некорректный FlashHead infer_params.yaml: {path}")

    return data


def save_rgb_png(src_path: Path, dst_path: Path) -> None:
    image = Image.open(src_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst_path)


def prepare_condition_image(
    source_rgb_path: Path,
    condition_image_path: Path,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """
    Готовит FlashHead condition image заранее:

    - открывает исходный avatar image как RGB;
    - делает resize_and_centercrop до target_size;
    - сохраняет condition.png;
    - возвращает uint8 tensor [1, 3, 1, H, W].

    Автоматический face crop не используется.
    Пользователь сам подаёт подходящую portrait/half-body картинку.
    """
    make_flashhead_importable()

    from flash_head.utils.utils import resize_and_centercrop

    condition_pil = Image.open(source_rgb_path).convert("RGB")

    condition_uint8 = resize_and_centercrop(
        condition_pil,
        target_size,
    )

    arr = (
        condition_uint8[0, :, 0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    Image.fromarray(arr).save(condition_image_path)

    return condition_uint8


def load_flashhead_vae(
    ckpt_dir: Path,
    model_type: str,
    device: torch.device,
    dtype: torch.dtype,
):
    make_flashhead_importable()

    if model_type == "lite":
        from flash_head.ltx_video.ltx_vae import LtxVAE

        return LtxVAE(
            pretrained_model_type_or_path=str(ckpt_dir / "VAE_LTX"),
            dtype=dtype,
            device=str(device),
        )

    if model_type in {"pro", "pretrained"}:
        from flash_head.wan.modules import WanVAE

        return WanVAE(
            vae_path=str(ckpt_dir / "VAE_Wan" / "Wan2.1_VAE.pth"),
            dtype=dtype,
            device=str(device),
            parallel=False,
        )

    raise RuntimeError(f"Неподдерживаемый FlashHead model_type: {model_type}")


@torch.no_grad()
def build_ref_img_latent(
    condition_uint8: torch.Tensor,
    ckpt_dir: Path,
    model_type: str,
    frame_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Делает главный precompute для FlashHead:

    condition image -> normalized tensor -> repeat to video -> VAE encode.

    Возвращает:
    - condition_tensor: [1, 3, 1, H, W], float, [-1, 1]
    - ref_img_latent: VAE latent статичного condition-video
    """
    vae = load_flashhead_vae(
        ckpt_dir=ckpt_dir,
        model_type=model_type,
        device=device,
        dtype=dtype,
    )

    condition_tensor = condition_uint8.to(device=device, dtype=dtype)
    condition_tensor = (condition_tensor / 255.0 - 0.5) * 2.0

    video_frames = condition_tensor.repeat(1, 1, frame_num, 1, 1)

    ref_img_latent = vae.encode(video_frames)

    # Сохраняем на CPU в float32: безопаснее и переносимее.
    condition_tensor_cpu = condition_tensor.detach().cpu().to(torch.float32)
    ref_img_latent_cpu = ref_img_latent.detach().cpu().to(torch.float32)

    del vae
    del condition_tensor
    del video_frames
    del ref_img_latent

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return condition_tensor_cpu, ref_img_latent_cpu


def save_metadata(metadata_path: Path, metadata: dict[str, object]) -> None:
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 09: подготовка avatar artifacts для SoulX-FlashHead.

    Вход:
    - source/avatar.jpg | avatar.jpeg | avatar.png | avatar.webp | avatar.bmp

    Выход:
    - artifacts/avatar/condition_source.png
    - artifacts/avatar/condition.png
    - artifacts/avatar/condition_tensor.pt
    - artifacts/avatar/ref_img_latent.pt
    - artifacts/avatar/metadata.json
    """
    logger.info("Начинаю avatar_prepare для FlashHead")

    make_flashhead_importable()

    source_path = ctx.paths.source_avatar_path
    out_dir = ctx.paths.artifacts_avatar_dir

    if not source_path.exists():
        raise FileNotFoundError(f"Не найден avatar source: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise RuntimeError(
            "FlashHead avatar_prepare поддерживает только avatar image.\n"
            f"Получен файл: {source_path.name}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FlashHead avatar_prepare требует CUDA, потому что заранее считает VAE latent."
        )

    ckpt_dir = DEFAULT_FLASHHEAD_CKPT_DIR
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Не найдены FlashHead weights: {ckpt_dir}")

    infer_params = load_infer_params()

    target_h = int(infer_params["height"])
    target_w = int(infer_params["width"])
    frame_num = int(infer_params["frame_num"])

    reset_dir(out_dir)
    ensure_dir(out_dir)

    source_rgb_path = out_dir / CONDITION_SOURCE_NAME
    condition_image_path = out_dir / CONDITION_IMAGE_NAME
    condition_tensor_path = out_dir / CONDITION_TENSOR_NAME
    ref_img_latent_path = out_dir / REF_IMG_LATENT_NAME
    metadata_path = out_dir / "metadata.json"

    logger.info("Готовлю FlashHead condition image")
    save_rgb_png(source_path, source_rgb_path)

    condition_uint8 = prepare_condition_image(
        source_rgb_path=source_rgb_path,
        condition_image_path=condition_image_path,
        target_size=(target_h, target_w),
    )

    logger.debug("Считаю FlashHead ref_img_latent через VAE")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    condition_tensor, ref_img_latent = build_ref_img_latent(
        condition_uint8=condition_uint8,
        ckpt_dir=ckpt_dir,
        model_type=MODEL_TYPE,
        frame_num=frame_num,
        device=device,
        dtype=dtype,
    )

    torch.save(condition_tensor, condition_tensor_path)
    torch.save(ref_img_latent, ref_img_latent_path)

    metadata = {
        "avatar_backend": AVATAR_BACKEND,
        "version": AVATAR_VERSION,
        "model_type": MODEL_TYPE,
        "person_name": PERSON_NAME,

        "source_path": str(source_path),
        "condition_source_kind": "image",
        "condition_preprocess": "resize_and_centercrop",

        "condition_source_name": CONDITION_SOURCE_NAME,
        "condition_image_name": CONDITION_IMAGE_NAME,
        "condition_tensor_name": CONDITION_TENSOR_NAME,
        "ref_img_latent_name": REF_IMG_LATENT_NAME,

        "condition_source_path": str(source_rgb_path),
        "condition_image_path": str(condition_image_path),
        "condition_tensor_path": str(condition_tensor_path),
        "ref_img_latent_path": str(ref_img_latent_path),

        "target_height": target_h,
        "target_width": target_w,
        "frame_num": frame_num,
        "motion_frames_latent_num": int(infer_params["motion_frames_latent_num"]),
        "tgt_fps": int(infer_params["tgt_fps"]),
        "sample_rate": int(infer_params["sample_rate"]),
        "sample_shift": float(infer_params["sample_shift"]),
        "color_correction_strength": float(infer_params["color_correction_strength"]),
        "cached_audio_duration": int(infer_params["cached_audio_duration"]),

        "condition_tensor_shape": list(condition_tensor.shape),
        "ref_img_latent_shape": list(ref_img_latent.shape),
        "condition_tensor_dtype": str(condition_tensor.dtype),
        "ref_img_latent_dtype": str(ref_img_latent.dtype),
    }

    save_metadata(metadata_path, metadata)

    logger.info("avatar_prepare FlashHead завершён")
    logger.debug("Condition image: %s", condition_image_path)
    logger.debug("Condition tensor: %s", condition_tensor_path)
    logger.debug("Ref image latent: %s", ref_img_latent_path)