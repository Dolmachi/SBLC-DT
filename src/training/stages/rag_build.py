from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings.base import TextEmbedder
from src.training.context import TrainingContext
from src.utils.fs import reset_dir


DialogPair = dict[str, str]

# Сколько dialog pairs кладём в один RAG-документ.
WINDOW_SIZE = 3
# С каким шагом двигаем окно.
STEP = 2

RAG_LABELS_BY_LANG: dict[str, tuple[str, str]] = {
    "ru": ("Фрагмент", "Собеседник"),
    "en": ("Fragment", "Interlocutor"),
    "es": ("Fragmento", "Interlocutor"),
    "de": ("Fragment", "Gesprächspartner"),
    "fr": ("Fragment", "Interlocuteur"),
    "it": ("Frammento", "Interlocutore"),
    "el": ("Απόσπασμα", "Συνομιλητής"),
    "pl": ("Fragment", "Rozmówca"),
    "pt": ("Fragmento", "Interlocutor"),
    "fi": ("Katkelma", "Keskustelukumppani"),
    "sv": ("Fragment", "Samtalspartner"),
    "nl": ("Fragment", "Gesprekspartner"),
    "da": ("Fragment", "Samtalepartner"),
    "no": ("Fragment", "Samtalepartner"),
    "he": ("קטע", "בן שיח"),
    "tr": ("Parça", "Konuşma partneri"),
    "ar": ("مقطع", "المحاور"),
    "hi": ("अंश", "वार्ताकार"),
    "zh": ("片段", "对话者"),
    "ja": ("断片", "話し相手"),
    "ko": ("조각", "대화 상대"),
    "tl": ("Fragment", "Kausap"),
    "vi": ("Đoạn trích", "Người đối thoại"),
}


def get_rag_labels(lang: str) -> tuple[str, str]:
    normalized_lang = lang.strip().lower()

    if normalized_lang not in RAG_LABELS_BY_LANG:
        raise RuntimeError(
            f"Для языка '{lang}' не заданы RAG labels.\n"
        )

    return RAG_LABELS_BY_LANG[normalized_lang]


def load_dialog_pairs(path: Path) -> list[DialogPair]:
    pairs: list[DialogPair] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            pairs.append(json.loads(line))

    return pairs


def build_rag_docs(
    pairs: list[DialogPair],
    target_name: str,
    lang: str,
) -> list[Document]:
    """
    Собирает RAG-документы из dialog pairs скользящим окном.

    Каждый документ содержит небольшой фрагмент диалога:
    [Собеседник]: ...
    [Имя target]: ...
    """
    docs: list[Document] = []
    total_pairs = len(pairs)

    if total_pairs == 0:
        return docs

    fragment_label, interlocutor_label = get_rag_labels(lang)

    starts = list(range(0, max(1, total_pairs - WINDOW_SIZE + 1), STEP))

    for start_idx in starts:
        chunk = pairs[start_idx : start_idx + WINDOW_SIZE]

        lines = [f"{fragment_label}:"]
        for pair in chunk:
            lines.append(f"[{interlocutor_label}]: {pair['user']}")
            lines.append(f"[{target_name}]: {pair['assistant']}")

        docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "start_pair": start_idx,
                    "end_pair": start_idx + len(chunk) - 1,
                },
            )
        )

    # Хвост
    last_start = starts[-1] if starts else 0
    tail_start = max(0, total_pairs - WINDOW_SIZE)

    if tail_start > last_start:
        chunk = pairs[tail_start:]

        lines = [f"{fragment_label}:"]
        for pair in chunk:
            lines.append(f"[{interlocutor_label}]: {pair['user']}")
            lines.append(f"[{target_name}]: {pair['assistant']}")

        docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "start_pair": tail_start,
                    "end_pair": total_pairs - 1,
                },
            )
        )

    return docs


def save_chroma_db(
    docs: list[Document],
    embedder: TextEmbedder,
    persist_dir: Path,
) -> None:
    """
    Строит и сохраняет Chroma DB.
    """
    embeddings = embedder.as_langchain_embeddings()

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="dialogs",
        persist_directory=str(persist_dir),
    )


def run(
    ctx: TrainingContext,
    embedder: TextEmbedder,
    logger: logging.Logger,
) -> None:
    """
    Stage 05: построение RAG-базы.

    Вход:
    - train_data/processed/dialog_pairs/dialog_pairs.jsonl

    Выход:
    - artifacts/rag/chroma/
    """
    pairs_path = ctx.paths.dialog_pairs_path
    chroma_dir = ctx.paths.artifacts_rag_chroma_dir

    logger.info("Начинаю rag_build")
    logger.info("Кол-во диалоговых пар: %s", pairs_path)
    logger.debug("Chroma dir: %s", chroma_dir)
    logger.debug("Embedding model: %s", embedder.model_id)

    if not pairs_path.exists():
        raise FileNotFoundError(f"Не найден файл dialog pairs: {pairs_path}")

    pairs = load_dialog_pairs(pairs_path)
    docs = build_rag_docs(
        pairs=pairs,
        target_name=ctx.cfg.name,
        lang=ctx.cfg.lang,
    )

    if not docs:
        raise RuntimeError("Не удалось построить ни одного RAG-документа.")

    reset_dir(chroma_dir)

    save_chroma_db(
        docs=docs,
        embedder=embedder,
        persist_dir=chroma_dir,
    )

    logger.info("rag_build завершён")
    logger.debug("Dialog pairs загружено: %d", len(pairs))
    logger.info("RAG документов создано: %d", len(docs))
    logger.info("Chroma RAG база успешно сохранена: %s", chroma_dir)