from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver


BASE_ROOT = Path(__file__).resolve().parent.parent
PROFILES_ROOT = BASE_ROOT / "profiles"
ENV_PATH = BASE_ROOT / ".env"

DIALOGS_FILE_NAME = "dialogs.json"

load_dotenv(ENV_PATH)

if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))


from src.inference.session import CloneInferenceSession
from src.utils.logger import setup_logger
from src.utils.profile_config import (
    ProfileConfig,
    ProfilePaths,
    build_profile_paths,
    load_profile_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        "--profiles",
        dest="profile",
        default=None,
        help="Path to profile directory, for example profiles/ernest.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Clone name from profile config.",
    )
    parser.add_argument(
        "--dialog-title",
        "--dialog_title",
        dest="dialog_title",
        default=None,
        help="Existing dialog title, for example dialog_0.",
    )
    parser.add_argument(
        "--delete-profile",
        action="store_true",
        help="Delete selected clone profile and exit.",
    )
    parser.add_argument(
        "--delete-dialog",
        action="store_true",
        help="Delete selected dialog memory and exit.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip deletion confirmation.",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen3.5-4B",
        help="LLM model id.",
    )
    parser.add_argument(
        "--asr-model",
        default="large-v3-turbo",
        help="faster-whisper runtime ASR model size.",
    )
    parser.add_argument(
        "--tts-model",
        default="openbmb/VoxCPM2",
        help="VoxCPM2 model id or local path.",
    )

    return parser


def load_profiles() -> list[tuple[Path, ProfileConfig]]:
    profiles: list[tuple[Path, ProfileConfig]] = []

    if not PROFILES_ROOT.exists():
        return profiles

    for profile_dir in sorted(PROFILES_ROOT.iterdir()):
        if not profile_dir.is_dir():
            continue

        if not (profile_dir / "config.json").exists():
            continue

        cfg = load_profile_config(profile_dir)
        profiles.append((profile_dir, cfg))

    return profiles


def select_index(title: str, labels: list[str]) -> int:
    print(f"\n{title}")

    for index, label in enumerate(labels):
        print(f"  {index}) {label}")

    while True:
        value = input("\nВыбери номер: ").strip()

        try:
            index = int(value)
        except ValueError:
            print("Нужно ввести номер из списка.")
            continue

        if 0 <= index < len(labels):
            return index

        print("Такого номера нет.")
        
        
def resolve_profile_by_path(profile: str) -> Path:
    profile_dir = Path(profile).expanduser()

    if not profile_dir.is_absolute():
        profile_dir = BASE_ROOT / profile_dir

    profile_dir = profile_dir.resolve()

    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Профиль не найден: {profile_dir}")

    if not (profile_dir / "config.json").exists():
        raise FileNotFoundError(f"В профиле нет config.json: {profile_dir}")

    return profile_dir


def resolve_profile_by_name(name: str) -> Path:
    name = name.strip().casefold()
    profiles = load_profiles()

    matches = [
        (profile_dir, cfg)
        for profile_dir, cfg in profiles
        if cfg.name.strip().casefold() == name
    ]

    if not matches:
        available = ", ".join(cfg.name for _, cfg in profiles) or "нет профилей"
        raise RuntimeError(
            f"Клон с таким именем не найден.\n"
            f"Доступные клоны: {available}"
        )

    if len(matches) > 1:
        variants = "\n".join(
            f"- {cfg.name}: {profile_dir}"
            for profile_dir, cfg in matches
        )
        raise RuntimeError(
            "Найдено несколько профилей с таким именем.\n"
            f"{variants}\n"
            "Выбери нужный явно через --profile."
        )

    return matches[0][0]


def select_profile_interactively() -> Path:
    profiles = load_profiles()

    if not profiles:
        raise RuntimeError(
            f"Нет доступных профилей в {PROFILES_ROOT}.\n"
            "Сначала запусти train pipeline."
        )

    labels = [
        f"{cfg.name} ({cfg.slug})"
        for _, cfg in profiles
    ]

    index = select_index("Доступные клоны:", labels)
    return profiles[index][0]


def resolve_profile(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.profile and args.name:
        parser.error("Укажи либо --profile/--profiles, либо --name, но не оба.")

    if args.delete_profile and args.delete_dialog:
        parser.error(
            "Нельзя одновременно использовать --delete-profile и --delete-dialog."
        )

    if args.delete_profile and args.dialog_title:
        parser.error("--dialog-title нельзя использовать вместе с --delete-profile.")

    if args.dialog_title and not args.profile and not args.name:
        parser.error(
            "--dialog-title нельзя использовать без --profile/--profiles или --name."
        )

    if args.profile:
        return resolve_profile_by_path(args.profile)

    if args.name:
        return resolve_profile_by_name(args.name)

    return select_profile_interactively()


def get_profile_paths(profile_dir: Path) -> ProfilePaths:
    cfg = load_profile_config(profile_dir)
    return build_profile_paths(profile_dir, cfg)


def get_memory_dir(profile_dir: Path) -> Path:
    paths = get_profile_paths(profile_dir)
    paths.memory_dir.mkdir(parents=True, exist_ok=True)
    return paths.memory_dir


def load_dialog_titles(memory_dir: Path) -> list[str]:
    path = memory_dir / DIALOGS_FILE_NAME

    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_dialogs = payload.get("dialogs", [])

    titles: list[str] = []

    for item in raw_dialogs:
        if isinstance(item, str):
            title = item.strip()
        else:
            title = str(item.get("title", "")).strip()

        if title:
            titles.append(title)

    return titles


def save_dialog_titles(memory_dir: Path, titles: list[str]) -> None:
    path = memory_dir / DIALOGS_FILE_NAME

    payload = {
        "dialogs": titles,
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_next_dialog(memory_dir: Path, titles: list[str]) -> str:
    existing = set(titles)

    index = 0
    while f"dialog_{index}" in existing:
        index += 1

    title = f"dialog_{index}"
    titles.append(title)

    save_dialog_titles(memory_dir, titles)

    return title


def resolve_dialog_title(profile_dir: Path, requested_title: str | None) -> str:
    memory_dir = get_memory_dir(profile_dir)
    titles = load_dialog_titles(memory_dir)

    if requested_title:
        if requested_title not in titles:
            available = ", ".join(titles) or "нет диалогов"
            raise RuntimeError(
                f"Диалог '{requested_title}' не найден.\n"
                f"Доступные диалоги: {available}"
            )

        return requested_title

    if not titles:
        title = create_next_dialog(memory_dir, titles)
        print(f"\nДиалогов ещё нет. Создан новый диалог: {title}")
        return title

    labels = [*titles, "Создать новый диалог"]
    index = select_index("Доступные диалоги:", labels)

    if index == len(titles):
        title = create_next_dialog(memory_dir, titles)
        print(f"\nСоздан новый диалог: {title}")
        return title

    return titles[index]


def resolve_existing_dialog_title(
    titles: list[str],
    requested_title: str | None,
) -> str:
    if not titles:
        raise RuntimeError("У выбранного профиля нет диалогов для удаления.")

    if requested_title:
        if requested_title not in titles:
            available = ", ".join(titles)
            raise RuntimeError(
                f"Диалог '{requested_title}' не найден.\n"
                f"Доступные диалоги: {available}"
            )

        return requested_title

    index = select_index("Какой диалог удалить:", titles)
    return titles[index]


def confirm_delete(entity: str, name: str, yes: bool) -> None:
    if yes:
        return

    print(f"\nБудет удалено: {entity} '{name}'")
    value = input(f"Для подтверждения введи '{name}': ").strip()

    if value != name:
        raise RuntimeError("Удаление отменено.")
    
    
def ensure_profile_inside_profiles_root(profile_dir: Path) -> None:
    profiles_root = PROFILES_ROOT.resolve()
    profile_dir = profile_dir.resolve()

    try:
        profile_dir.relative_to(profiles_root)
    except ValueError:
        raise RuntimeError(
            "Удалять через clone.py можно только профили внутри папки profiles/.\n"
            f"Переданный путь: {profile_dir}"
        )

    if profile_dir == profiles_root:
        raise RuntimeError("Нельзя удалить саму папку profiles/.")


def delete_profile(profile_dir: Path, yes: bool) -> None:
    cfg = load_profile_config(profile_dir)

    ensure_profile_inside_profiles_root(profile_dir)
    confirm_delete(
        entity="профиль",
        name=cfg.slug,
        yes=yes,
    )

    shutil.rmtree(profile_dir)

    print(f"\nПрофиль удалён: {profile_dir}")
        

def delete_thread_from_memory(sqlite_path: Path, thread_id: str) -> None:
    if not sqlite_path.exists():
        return

    with sqlite3.connect(sqlite_path.as_posix(), check_same_thread=False) as conn:
        memory = SqliteSaver(conn)
        memory.delete_thread(thread_id)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def delete_dialog(profile_dir: Path, requested_title: str | None, yes: bool) -> None:
    paths = get_profile_paths(profile_dir)
    paths.memory_dir.mkdir(parents=True, exist_ok=True)

    titles = load_dialog_titles(paths.memory_dir)
    title = resolve_existing_dialog_title(
        titles=titles,
        requested_title=requested_title,
    )

    confirm_delete(
        entity="диалог",
        name=title,
        yes=yes,
    )

    delete_thread_from_memory(
        sqlite_path=paths.memory_sqlite_path,
        thread_id=title,
    )
    
    remaining_titles = [
        existing_title
        for existing_title in titles
        if existing_title != title
    ]

    save_dialog_titles(
        memory_dir=paths.memory_dir,
        titles=remaining_titles,
    )

    print(f"\nДиалог удалён: {title}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logger = setup_logger()

    profile_dir = resolve_profile(args, parser)

    if args.delete_profile:
        delete_profile(
            profile_dir=profile_dir,
            yes=args.yes,
        )
        return

    if args.delete_dialog:
        delete_dialog(
            profile_dir=profile_dir,
            requested_title=args.dialog_title,
            yes=args.yes,
        )
        return

    dialog_title = resolve_dialog_title(
        profile_dir=profile_dir,
        requested_title=args.dialog_title,
    )

    session = CloneInferenceSession.load(
        profile_dir=profile_dir,
        thread_id=dialog_title,
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