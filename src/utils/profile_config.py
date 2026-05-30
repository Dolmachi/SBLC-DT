from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ProfileConfig:
    """
    Конфиг профиля конкретного двойника.
    """
    name: str
    slug: str
    lang: str
    avatar_file_name: str
    reference_file_name: str
    embedding_model_id: str


@dataclass
class ProfilePaths:
    """
    Все основные пути профиля.
    """

    profile_dir: Path
    config_path: Path

    source_dir: Path
    source_dialogs_dir: Path
    source_profile_txt: Path
    source_reference_path: Path
    source_avatar_path: Path

    train_data_dir: Path
    interim_dir: Path
    processed_dir: Path

    audio_asr_dir: Path
    audio_tts_dir: Path
    asr_raw_dir: Path
    asr_postprocess_dir: Path
    reference_wav_path: Path

    dialog_pairs_dir: Path
    dialog_pairs_path: Path

    tts_dataset_dir: Path
    tts_wavs_dir: Path
    tts_train_manifest_path: Path
    tts_val_manifest_path: Path

    artifacts_dir: Path
    artifacts_rag_dir: Path
    artifacts_rag_chroma_dir: Path

    artifacts_tts_dir: Path
    artifacts_tts_lora_dir: Path
    artifacts_tts_lora_latest_dir: Path
    artifacts_tts_reference_wav_path: Path

    artifacts_avatar_dir: Path

    memory_dir: Path
    memory_sqlite_path: Path

    logs_dir: Path


def build_profile_paths(profile_dir: Path, cfg: ProfileConfig) -> ProfilePaths:
    """
    Строит все основные пути для профиля.
    """
    profile_dir = Path(profile_dir).resolve()

    source_dir = profile_dir / "source"
    train_data_dir = profile_dir / "train_data"
    interim_dir = train_data_dir / "interim"
    processed_dir = train_data_dir / "processed"

    dialog_pairs_dir = processed_dir / "dialog_pairs"
    tts_dataset_dir = processed_dir / "tts_dataset"

    artifacts_dir = profile_dir / "artifacts"
    artifacts_tts_dir = artifacts_dir / "tts"

    memory_dir = profile_dir / "memory"

    return ProfilePaths(
        profile_dir=profile_dir,
        config_path=profile_dir / "config.json",

        source_dir=source_dir,
        source_dialogs_dir=source_dir / "dialogs",
        source_profile_txt=source_dir / "profile.txt",
        source_reference_path=source_dir / cfg.reference_file_name,
        source_avatar_path=source_dir / cfg.avatar_file_name,

        train_data_dir=train_data_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,

        audio_asr_dir=interim_dir / "audio_asr",
        audio_tts_dir=interim_dir / "audio_tts",
        asr_raw_dir=interim_dir / "asr_raw",
        asr_postprocess_dir=interim_dir / "asr_postprocess",
        reference_wav_path=interim_dir / "reference.wav",

        dialog_pairs_dir=dialog_pairs_dir,
        dialog_pairs_path=dialog_pairs_dir / "dialog_pairs.jsonl",

        tts_dataset_dir=tts_dataset_dir,
        tts_wavs_dir=tts_dataset_dir / "wavs",
        tts_train_manifest_path=tts_dataset_dir / "train.jsonl",
        tts_val_manifest_path=tts_dataset_dir / "val.jsonl",
        artifacts_dir=artifacts_dir,
        artifacts_rag_dir=artifacts_dir / "rag",
        artifacts_rag_chroma_dir=artifacts_dir / "rag" / "chroma",

        artifacts_tts_dir=artifacts_tts_dir,
        artifacts_tts_lora_dir=artifacts_tts_dir / "lora",
        artifacts_tts_lora_latest_dir=artifacts_tts_dir / "lora" / "latest",
        artifacts_tts_reference_wav_path=artifacts_tts_dir / "reference.wav",

        artifacts_avatar_dir=artifacts_dir / "avatar",

        memory_dir=memory_dir,
        memory_sqlite_path=memory_dir / "memory.sqlite",

        logs_dir=profile_dir / "logs",
    )


def save_profile_config(cfg: ProfileConfig, profile_dir: Path) -> Path:
    """
    Сохраняет ProfileConfig в profiles/<slug>/config.json.
    """
    profile_dir = Path(profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    config_path = profile_dir / "config.json"
    payload = asdict(cfg)

    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return config_path


def load_profile_config(profile_dir: Path) -> ProfileConfig:
    """
    Загружает ProfileConfig из profiles/<slug>/config.json.
    """
    profile_dir = Path(profile_dir).resolve()
    config_path = profile_dir / "config.json"

    payload = json.loads(config_path.read_text(encoding="utf-8"))

    expected_keys = set(ProfileConfig.__dataclass_fields__.keys())
    payload_keys = set(payload.keys())

    missing_keys = expected_keys - payload_keys
    extra_keys = payload_keys - expected_keys

    if missing_keys:
        raise RuntimeError(
            "Конфиг профиля устарел или поврежден.\n"
            f"Не хватает полей: {sorted(missing_keys)}\n"
            "После изменения формата профиля его нужно пересобрать."
        )

    if extra_keys:
        raise RuntimeError(
            f"Конфиг профиля содержит устаревшие поля: {sorted(extra_keys)}\n"
        )

    return ProfileConfig(**payload)