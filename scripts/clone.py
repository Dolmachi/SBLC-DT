from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


BASE_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_ROOT / ".env"

load_dotenv(ENV_PATH)

if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from src.inference.session import CloneInferenceSession
from src.utils.logger import setup_logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
        help="Path to profiles/<slug>",
    )

    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen3.5-4B",
        help="LLM model id",
    )

    parser.add_argument(
        "--asr-model",
        default="large-v3-turbo",
        help="faster-whisper runtime ASR model size",
    )

    parser.add_argument(
        "--tts-model",
        default="openbmb/VoxCPM2",
        help="VoxCPM2 model id or local path",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logger = setup_logger()

    session = CloneInferenceSession.load(
        profile_dir=Path(args.profile),
        llm_model_id=args.llm_model,
        asr_model_size=args.asr_model,
        tts_model_id=args.tts_model,
        logger=logger,
    )

    try:
        session.run()
    finally:
        session.close()


if __name__ == "__main__":
    main()