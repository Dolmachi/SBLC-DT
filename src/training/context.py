from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.profile_config import (
    ProfileConfig,
    ProfilePaths,
    build_profile_paths,
    load_profile_config,
)


@dataclass
class TrainingContext:
    """
    Общий контекст train pipeline.
    """
    cfg: ProfileConfig
    paths: ProfilePaths


def build_training_context(profile_dir: Path) -> TrainingContext:
    """
    Создаёт TrainingContext из директории профиля.
    """
    profile_dir = Path(profile_dir).resolve()

    cfg = load_profile_config(profile_dir)
    paths = build_profile_paths(profile_dir, cfg)

    return TrainingContext(
        cfg=cfg,
        paths=paths,
    )