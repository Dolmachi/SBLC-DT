from __future__ import annotations

from typing import Any


DEFAULT_EMBEDDERS_BY_LANG: dict[str, str] = {
    "en": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "ru": "ai-forever/FRIDA",
    "es": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "de": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "fr": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "it": "DeepMount00/Ita-Search",
    "el": "Alibaba-NLP/gte-multilingual-base",
    "pl": "ibm-granite/granite-embedding-311m-multilingual-r2",
    "pt": "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",
    "fi": "ibm-granite/granite-embedding-311m-multilingual-r2",
    "sv": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "nl": "clips/e5-large-trm-nl",
    "da": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "no": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "he": "dicta-il/neodictabert-bilingual-embed",
    "tr": "newmindai/TurkEmbed4Retrieval",
    "ar": "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    "hi": "ibm-granite/granite-embedding-311m-multilingual-r2",
    "zh": "richinfoai/ritrieve_zh_v1",
    "ja": "cl-nagoya/ruri-v3-310m",
    "ko": "telepix/PIXIE-Rune-Preview",
    "tl": "aisingapore/SEA-LION-E5-Embedding-600M",
    "vi": "AITeamVN/Vietnamese_Embedding_v2",
}


EMBEDDER_KWARGS_BY_MODEL: dict[str, dict[str, Any]] = {
    "jinaai/jina-embeddings-v5-text-small-retrieval": {
        "query_encode_kwargs": {
            "prompt_name": "query",
        },
        "encode_kwargs": {
            "prompt_name": "document",
        },
    },

    "ai-forever/FRIDA": {
        "query_encode_kwargs": {
            "prompt": "search_query: ",
        },
        "encode_kwargs": {
            "prompt": "search_document: ",
        },
    },

    "DeepMount00/Ita-Search": {
        "query_encode_kwargs": {
            "prompt": "Represent this search query for finding relevant passages: ",
        },
        "encode_kwargs": {
            "prompt": "Represent this passage for retrieval: ",
        },
    },

    "Alibaba-NLP/gte-multilingual-base": {
        "trust_remote_code": True,
    },

    "dicta-il/neodictabert-bilingual-embed": {
        "trust_remote_code": True,
        "query_encode_kwargs": {
            "prompt": "query: ",
        },
    },

    "clips/e5-large-trm-nl": {
        "query_encode_kwargs": {
            "prompt": "query: ",
        },
        "encode_kwargs": {
            "prompt": "passage: ",
        },
    },

    "cl-nagoya/ruri-v3-310m": {
        "query_encode_kwargs": {
            "prompt": "検索クエリ: ",
        },
        "encode_kwargs": {
            "prompt": "検索文書: ",
        },
    },

    "telepix/PIXIE-Rune-Preview": {
        "query_encode_kwargs": {
            "prompt_name": "query",
        },
    },

    "aisingapore/SEA-LION-E5-Embedding-600M": {
        "query_encode_kwargs": {
            "prompt_name": "Retrieval",
        },
    },
}


def resolve_embedding_model_id(
    lang: str,
    explicit_model_id: str | None = None,
) -> str:
    """
    Выбирает embedding-модель.

    Приоритет:
    1. Если пользователь/код явно передал model_id — используем его.
    2. Иначе выбираем дефолтную модель по языку.
    """
    if explicit_model_id:
        return explicit_model_id

    normalized_lang = lang.strip().lower()

    if normalized_lang not in DEFAULT_EMBEDDERS_BY_LANG:
        raise RuntimeError(
            f"Для языка '{lang}' не задан embedding model.\n"
        )

    return DEFAULT_EMBEDDERS_BY_LANG[normalized_lang]


def get_embedder_kwargs(model_id: str) -> dict[str, Any]:
    """
    Возвращает доп настройки для конкретной embedding-модели
    """
    return EMBEDDER_KWARGS_BY_MODEL.get(model_id.strip(), {})