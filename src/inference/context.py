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
class InferenceContext:
    """
    Общий контекст inference pipeline.

    Содержит:
    - config профиля;
    - все пути профиля;
    - доступ к artifacts, source, memory.
    """
    cfg: ProfileConfig
    paths: ProfilePaths


def build_inference_context(profile_dir: Path) -> InferenceContext:
    """
    Создаёт InferenceContext из директории профиля.
    """
    profile_dir = Path(profile_dir).expanduser().resolve()

    cfg = load_profile_config(profile_dir)
    paths = build_profile_paths(profile_dir, cfg)

    return InferenceContext(
        cfg=cfg,
        paths=paths,
    )