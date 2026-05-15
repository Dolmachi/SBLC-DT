from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASE_ROOT = Path(__file__).resolve().parent.parent
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from src.training.pipeline import TrainingPipeline
from src.utils.logger import setup_logger


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Parser аргументов для train entrypoint.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", required=True, help="Имя клонируемого человека")
    parser.add_argument("--lang", required=True, help="Язык профиля, например ru")
    parser.add_argument("--data", required=True, help="Путь до входной папки данных")

    return parser


def main() -> None:
    """
    Train entrypoint.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    logger = setup_logger()

    pipeline = TrainingPipeline(logger=logger)

    try:
        profile_dir = pipeline.run(
            name=args.name,
            lang=args.lang,
            data_path=Path(args.data)
        )

        logger.info("Профиль успешно создан и обработан: %s", profile_dir)

    except Exception:
        logger.exception("Train завершился с ошибкой")
        raise


if __name__ == "__main__":
    main()