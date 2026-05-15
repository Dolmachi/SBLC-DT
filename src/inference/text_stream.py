from __future__ import annotations

import re
from collections.abc import Iterator


SENTENCE_END_RE = re.compile(r"([.!?…]+)(\s+|$)")


class SentenceStreamBuffer:
    """
    Собирает поток LLM-текста в более крупные TTS-чанки.
    """

    def __init__(
        self,
        min_chars: int = 120,
        max_chars: int = 260,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

        self._buffer = ""
        self._ready_sentences: list[str] = []

    def push(self, text_delta: str) -> Iterator[str]:
        if not text_delta:
            return

        self._buffer += text_delta

        while True:
            sentence = self._pop_ready_sentence()
            if sentence is None:
                break

            self._ready_sentences.append(sentence)

            ready_text = self._ready_text()
            if len(ready_text) >= self.min_chars:
                yield self._flush_ready()

        ready_text = self._ready_text()

        if len(ready_text) >= self.max_chars:
            yield self._flush_ready()
            return

        if len(self._buffer) >= self.max_chars:
            if self._ready_sentences:
                yield self._flush_ready()
            else:
                yield self._flush_soft()

    def flush(self) -> str | None:
        parts: list[str] = []

        if self._ready_sentences:
            parts.append(self._flush_ready())

        tail = self._buffer.strip()
        self._buffer = ""

        if tail:
            parts.append(tail)

        text = " ".join(part.strip() for part in parts if part.strip()).strip()
        return text or None

    def _pop_ready_sentence(self) -> str | None:
        match = SENTENCE_END_RE.search(self._buffer)
        if not match:
            return None

        end_index = match.end()
        sentence = self._buffer[:end_index].strip()
        self._buffer = self._buffer[end_index:]

        return sentence or None

    def _ready_text(self) -> str:
        return " ".join(self._ready_sentences).strip()

    def _flush_ready(self) -> str:
        text = self._ready_text()
        self._ready_sentences.clear()
        return text

    def _flush_soft(self) -> str:
        cut_index = max(
            self._buffer.rfind(","),
            self._buffer.rfind(";"),
            self._buffer.rfind(" "),
        )

        if cut_index <= 0:
            cut_index = len(self._buffer)

        text = self._buffer[:cut_index].strip()
        self._buffer = self._buffer[cut_index:]

        return text