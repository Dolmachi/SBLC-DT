from __future__ import annotations


DialogPair = dict[str, str]


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
            f"Для языка '{lang}' не заданы RAG labels."
        )

    return RAG_LABELS_BY_LANG[normalized_lang]


def format_rag_fragment(
    pairs: list[DialogPair],
    target_name: str,
    lang: str,
) -> str:
    """
    Форматирует RAG-документ.
    """

    fragment_label, interlocutor_label = get_rag_labels(lang)

    lines = [f"{fragment_label}:"]

    for pair in pairs:
        lines.append(f"[{interlocutor_label}]: {pair['user']}")
        lines.append(f"[{target_name}]: {pair['assistant']}")

    return "\n".join(lines)


def format_retrieval_query(
    previous_pairs: list[DialogPair],
    current_user_text: str,
    target_name: str,
    lang: str,
) -> str:
    """
    Форматирует inference query для retriever.
    """

    _, interlocutor_label = get_rag_labels(lang)

    lines: list[str] = []

    for pair in previous_pairs:
        lines.append(f"[{interlocutor_label}]: {pair['user']}")
        lines.append(f"[{target_name}]: {pair['assistant']}")

    lines.append(f"[{interlocutor_label}]: {current_user_text}")

    return "\n".join(lines)