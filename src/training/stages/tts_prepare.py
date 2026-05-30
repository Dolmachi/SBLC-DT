from __future__ import annotations

import gc
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.training.context import TrainingContext
from src.utils.fs import ensure_dir, reset_dir


LORA_BATCH_SIZE = 1
LORA_GRAD_ACCUM_STEPS = 16
LORA_NUM_WORKERS = 2

LORA_EPOCHS_WITH_VAL = 10.0
LORA_EPOCHS_NO_VAL = 2.0

LORA_LR = 1e-4
LORA_WEIGHT_DECAY = 1e-2
LORA_WARMUP_RATIO = 0.10
LORA_MAX_GRAD_NORM = 1.0
LORA_MAX_BATCH_TOKENS = 4096

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

CHECKPOINT_COUNT = 10
LOG_INTERVAL = 10

LOSS_WEIGHTS = {
    "loss/diff": 1.0,
    "loss/stop": 1.0,
}


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def resolve_voxcpm2_pretrained_path() -> Path:
    """
    Возвращает локальный snapshot VoxCPM2.
    """
    path = Path(snapshot_download(repo_id="openbmb/VoxCPM2")).resolve()

    if not (path / "config.json").exists():
        raise FileNotFoundError(f"В VoxCPM2 snapshot нет config.json: {path}")

    return path


def lora_config_dict() -> dict[str, Any]:
    return {
        "enable_lm": True,
        "enable_dit": True,
        "enable_proj": False,
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
    }


def serialize_lora_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())

    if hasattr(config, "dict"):
        return dict(config.dict())

    return dict(vars(config))


def choose_num_iters(
    *,
    train_count: int,
    batch_size: int,
    has_validation: bool,
) -> int:
    """
    Считает число optimizer steps.

    При наличии val обучаем 3 эпохи и выбираем лучший из 3 чекпойнтов.
    При малом датасете val не делаем и обучаем 2 эпохи.
    """
    epochs = LORA_EPOCHS_WITH_VAL if has_validation else LORA_EPOCHS_NO_VAL

    effective_batch = batch_size * LORA_GRAD_ACCUM_STEPS
    steps_per_epoch = max(1, math.ceil(train_count / effective_batch))

    return max(1, math.ceil(steps_per_epoch * epochs))


def build_checkpoint_steps(num_iters: int) -> list[int]:
    """
    Выбирает до 3 равномерно расположенных checkpoint steps.
    """
    steps = {
        max(1, math.ceil(num_iters * index / CHECKPOINT_COUNT))
        for index in range(1, CHECKPOINT_COUNT + 1)
    }

    return sorted(steps)


def save_lora_checkpoint(
    *,
    model: Any,
    save_dir: Path,
    step: int,
    pretrained_path: Path,
    val_metrics: dict[str, float] | None,
) -> Path:
    """
    Сохраняет LoRA checkpoint в artifacts/tts/lora/step_XXXXXXX.
    latest здесь не трогаем: latest будет копией лучшего checkpoint.
    """
    ensure_dir(save_dir)

    checkpoint_dir = save_dir / f"step_{step:07d}"
    reset_dir(checkpoint_dir)

    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_" in key
    }

    if not state:
        raise RuntimeError("В модели не найдены LoRA-веса.")

    save_file(state, checkpoint_dir / "lora_weights.safetensors")

    lora_info = {
        "base_model": "openbmb/VoxCPM2",
        "pretrained_path": str(pretrained_path),
        "lora_config": serialize_lora_config(model.lora_config),
    }

    (checkpoint_dir / "lora_config.json").write_text(
        json.dumps(lora_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "step": int(step),
        "val_metrics": val_metrics,
    }

    (checkpoint_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return checkpoint_dir


def update_latest_checkpoint(
    *,
    checkpoint_dir: Path,
    latest_dir: Path,
) -> None:
    if latest_dir.exists():
        shutil.rmtree(latest_dir)

    shutil.copytree(checkpoint_dir, latest_dir)


def filter_by_max_batch_tokens(
    *,
    dataset: Any,
    model: Any,
    batch_size: int,
    logger: logging.Logger,
    split_name: str,
) -> Any:
    """
    Повторяет official max_batch_tokens-фильтр VoxCPM.
    """
    if LORA_MAX_BATCH_TOKENS <= 0:
        return dataset

    from voxcpm.training.data import compute_sample_lengths

    audio_vae_fps = model.audio_vae.sample_rate / model.audio_vae.hop_length

    lengths = compute_sample_lengths(
        dataset,
        audio_vae_fps=audio_vae_fps,
        patch_size=model.config.patch_size,
    )

    max_sample_len = LORA_MAX_BATCH_TOKENS // batch_size

    keep_indices = [
        index
        for index, length in enumerate(lengths)
        if length <= max_sample_len
    ]

    removed = len(dataset) - len(keep_indices)

    if removed > 0:
        logger.info(
            "VoxCPM2 LoRA: отфильтровано длинных %s samples: %d / %d",
            split_name,
            removed,
            len(dataset),
        )

    return dataset.select(keep_indices)


def evaluate_lora(
    *,
    model: Any,
    val_loader: Any,
    batch_processor: Any,
    accelerator: Any,
) -> dict[str, float]:
    """
    Считает validation loss на val_loader.

    Это тот же objective, что и на train:
    val/loss_total = weighted sum(loss/diff, loss/stop).
    """
    losses: list[torch.Tensor] = []
    sub_losses: dict[str, list[torch.Tensor]] = {}

    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            processed = batch_processor(batch)

            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                    sample_generate=False,
                )

                total = None

                for key, value in outputs.items():
                    if not key.startswith("loss/"):
                        continue

                    weighted = value * LOSS_WEIGHTS.get(key, 1.0)
                    total = weighted if total is None else total + weighted
                    sub_losses.setdefault(key, []).append(value.detach())

                if total is None:
                    raise RuntimeError("VoxCPM2 validation не вернул loss/*.")

                losses.append(total.detach())

    model.train()

    total_loss = float(torch.stack(losses).mean().cpu())

    metrics = {
        "loss/total": total_loss,
    }

    for key, values in sub_losses.items():
        metrics[key] = float(torch.stack(values).mean().cpu())

    return metrics


def train_voxcpm2_lora(
    *,
    pretrained_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    lora_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Встроенный VoxCPM2 LoRA training loop.

    Логика:
    - если есть val.jsonl, сохраняем 3 checkpoint-а и выбираем лучший по val/loss;
    - если val.jsonl пустой, сохраняем checkpoints, latest = финальный.
    """
    from voxcpm.model import VoxCPM2Model
    from voxcpm.model.voxcpm2 import LoRAConfig
    from voxcpm.training import (
        Accelerator,
        BatchProcessor,
        build_dataloader,
        load_audio_text_datasets,
    )

    train_count = count_jsonl_rows(train_manifest)
    val_count = count_jsonl_rows(val_manifest)

    if train_count == 0:
        raise RuntimeError(f"TTS train manifest пуст: {train_manifest}")

    has_validation = val_count > 0

    accelerator = Accelerator(amp=True)

    model = VoxCPM2Model.from_local(
        str(pretrained_path),
        optimize=False,
        training=True,
        lora_config=LoRAConfig(**lora_config_dict()),
    )

    expected_sample_rate = int(model.audio_vae.sample_rate)

    if expected_sample_rate != 16000:
        raise RuntimeError(
            f"VoxCPM2 AudioVAE ожидает sample_rate={expected_sample_rate}, "
            "а pipeline готовит 16000."
        )

    tokenizer = model.text_tokenizer

    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=str(train_manifest),
        val_manifest=str(val_manifest) if has_validation else "",
        sample_rate=expected_sample_rate,
    )

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "text_ids": [
                tokenizer(text)
                for text in batch["text"]
            ]
        }

    train_ds = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    if val_ds is not None:
        val_ds = val_ds.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
        )

    batch_size = min(LORA_BATCH_SIZE, max(1, len(train_ds)))

    train_ds = filter_by_max_batch_tokens(
        dataset=train_ds,
        model=model,
        batch_size=batch_size,
        logger=logger,
        split_name="train",
    )

    if val_ds is not None:
        val_ds = filter_by_max_batch_tokens(
            dataset=val_ds,
            model=model,
            batch_size=batch_size,
            logger=logger,
            split_name="val",
        )

        if len(val_ds) == 0:
            val_ds = None
            has_validation = False

    if len(train_ds) == 0:
        raise RuntimeError("После фильтрации TTS train dataset пуст.")

    dataset_count = (
        int(max(train_ds["dataset_id"])) + 1
        if "dataset_id" in train_ds.column_names
        else 1
    )

    batch_processor = BatchProcessor(
        config=model.config,
        audio_vae=model.audio_vae,
        dataset_cnt=dataset_count,
        device=accelerator.device,
    )

    del model.audio_vae

    model = accelerator.prepare_model(model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LORA_LR,
        weight_decay=LORA_WEIGHT_DECAY,
    )

    num_iters = choose_num_iters(
        train_count=len(train_ds),
        batch_size=batch_size,
        has_validation=has_validation,
    )

    warmup_steps = max(1, int(num_iters * LORA_WARMUP_RATIO))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_iters,
    )

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=LORA_NUM_WORKERS,
        drop_last=False,
    )

    val_loader = None

    if val_ds is not None:
        val_loader = build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=LORA_NUM_WORKERS,
            drop_last=False,
        )

    checkpoint_steps = set(build_checkpoint_steps(num_iters))

    logger.info(
        "VoxCPM2 LoRA training: train=%d, val=%d, batch=%d, grad_accum=%d, iters=%d, checkpoints=%s",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        batch_size,
        LORA_GRAD_ACCUM_STEPS,
        num_iters,
        sorted(checkpoint_steps),
    )

    train_iter = iter(train_loader)

    def next_batch() -> Any:
        nonlocal train_iter

        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    saved: list[tuple[int, Path, float | None]] = []

    reset_dir(lora_dir)

    for step in range(num_iters):
        optimizer.zero_grad(set_to_none=True)

        loss_log: dict[str, float] = {}

        for _micro_step in range(LORA_GRAD_ACCUM_STEPS):
            batch = next_batch()
            processed = batch_processor(batch)

            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=step / max(1, num_iters),
                )

                total_loss = None

                for key, value in outputs.items():
                    if not key.startswith("loss/"):
                        continue

                    weighted = value * LOSS_WEIGHTS.get(key, 1.0)
                    loss_value = weighted / LORA_GRAD_ACCUM_STEPS

                    total_loss = loss_value if total_loss is None else total_loss + loss_value
                    loss_log[key] = float(value.detach().cpu())

            if total_loss is None:
                raise RuntimeError("VoxCPM2 forward не вернул loss/*.")

            accelerator.backward(total_loss)

        scaler = getattr(accelerator, "scaler", None)

        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            unwrapped_model.parameters(),
            max_norm=LORA_MAX_GRAD_NORM,
        )

        accelerator.step(optimizer)
        accelerator.update()
        scheduler.step()

        current_step = step + 1

        if step % LOG_INTERVAL == 0 or current_step == num_iters:
            loss_text = ", ".join(
                f"{key}={value:.4f}"
                for key, value in sorted(loss_log.items())
            )

            logger.info(
                "VoxCPM2 LoRA step %d/%d | lr=%.2e | grad_norm=%.3f | %s",
                current_step,
                num_iters,
                optimizer.param_groups[0]["lr"],
                float(grad_norm),
                loss_text,
            )

        if current_step in checkpoint_steps:
            val_metrics = None
            val_loss = None

            if val_loader is not None:
                val_metrics = evaluate_lora(
                    model=unwrapped_model,
                    val_loader=val_loader,
                    batch_processor=batch_processor,
                    accelerator=accelerator,
                )
                val_loss = float(val_metrics["loss/total"])

                logger.info(
                    "VoxCPM2 LoRA checkpoint step %d | val_loss=%.4f | %s",
                    current_step,
                    val_loss,
                    ", ".join(
                        f"{key}={value:.4f}"
                        for key, value in sorted(val_metrics.items())
                    ),
                )
            else:
                logger.info(
                    "VoxCPM2 LoRA checkpoint step %d | validation disabled",
                    current_step,
                )

            checkpoint_dir = save_lora_checkpoint(
                model=unwrapped_model,
                save_dir=lora_dir,
                step=current_step,
                pretrained_path=pretrained_path,
                val_metrics=val_metrics,
            )

            saved.append((current_step, checkpoint_dir, val_loss))

    if not saved:
        raise RuntimeError("VoxCPM2 LoRA не сохранил ни одного checkpoint.")

    if has_validation and any(item[2] is not None for item in saved):
        best_step, best_dir, best_loss = min(
            (item for item in saved if item[2] is not None),
            key=lambda item: float(item[2]),
        )

        logger.info(
            "Лучший VoxCPM2 LoRA checkpoint: step=%d, val_loss=%.4f",
            best_step,
            float(best_loss),
        )
    else:
        best_step, best_dir, best_loss = saved[-1]

        logger.info(
            "Validation недоступна, выбран финальный VoxCPM2 LoRA checkpoint: step=%d",
            best_step,
        )

    update_latest_checkpoint(
        checkpoint_dir=best_dir,
        latest_dir=lora_dir / "latest",
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run(ctx: TrainingContext, logger: logging.Logger) -> None:
    """
    Stage 07: VoxCPM2 LoRA fine-tuning.

    Вход:
    - train_data/processed/tts_dataset/train.jsonl
    - train_data/processed/tts_dataset/val.jsonl
    - train_data/processed/tts_dataset/reference.wav

    Выход:
    - artifacts/tts/reference.wav
    - artifacts/tts/lora/step_*/
    - artifacts/tts/lora/latest/lora_weights.safetensors
    - artifacts/tts/lora/latest/lora_config.json
    """
    logger.info("Начинаю tts_prepare")

    reset_dir(ctx.paths.artifacts_tts_dir)
    ensure_dir(ctx.paths.artifacts_tts_lora_dir)

    dataset_reference_path = ctx.paths.tts_dataset_dir / "reference.wav"

    if not dataset_reference_path.exists():
        raise FileNotFoundError(f"Не найден TTS reference: {dataset_reference_path}")

    shutil.copy2(
        dataset_reference_path,
        ctx.paths.artifacts_tts_reference_wav_path,
    )

    logger.info("VoxCPM2 runtime reference: %s", ctx.paths.artifacts_tts_reference_wav_path)

    pretrained_path = resolve_voxcpm2_pretrained_path()

    train_voxcpm2_lora(
        pretrained_path=pretrained_path,
        train_manifest=ctx.paths.tts_train_manifest_path,
        val_manifest=ctx.paths.tts_val_manifest_path,
        lora_dir=ctx.paths.artifacts_tts_lora_dir,
        logger=logger,
    )

    logger.info("tts_prepare завершён")
    logger.info("VoxCPM2 LoRA latest: %s", ctx.paths.artifacts_tts_lora_latest_dir)