from __future__ import annotations

import argparse
from pathlib import Path


BASE_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_ROOT / ".env"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token для WhisperX diarization / pyannote",
    )

    return parser


def upsert_env_value(env_path: Path, key: str, value: str) -> None:
    lines: list[str] = []

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    prefix = f"{key}="
    new_line = f"{key}={value}"

    replaced = False
    new_lines: list[str] = []

    for line in lines:
        if line.startswith(prefix):
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append(new_line)

    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.hf_token:
        upsert_env_value(
            env_path=ENV_PATH,
            key="HF_TOKEN",
            value=args.hf_token.strip(),
        )
        print(f"HF_TOKEN сохранён в {ENV_PATH}")
    else:
        print(f".env находится здесь: {ENV_PATH}")
        print("Чтобы сохранить HF_TOKEN:")
        print("  python scripts/setup.py --hf-token hf_xxx")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
