"""Split documents into overlapping, boundary-aware chunks.

Why chunk at all: embedding a whole file blurs every topic in it into one vector, and the LLM prompt has
a budget. Small passages (~800 characters) give precise hits; the overlap keeps a fact that straddles a
cut retrievable from either side. Cuts are snapped to the nearest paragraph, line or sentence end so a
chunk rarely stops mid-thought, and every chunk remembers where it came from (file, page, offsets).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .loaders import Document, title_from_source
from .utils import stable_id

# Tried in order: paragraph break, line break, sentence end, clause, word.
SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.metadata.get("title") or title_from_source(self.source)

    @property
    def page(self) -> int | None:
        return self.metadata.get("page")

    @property
    def indexed_text(self) -> str:
        """What gets embedded and BM25-indexed. The document title is prepended so a passage that
        never repeats its subject ("... the limit is 2,000 USD") still carries it into the vector."""
        return f"{self.title}\n{self.text}"

    def label(self) -> str:
        """Short citation label, e.g. 'travel_policy.pdf, page 2, chunk 5'."""
        parts = [self.source]
        if self.page is not None:
            parts.append(f"page {self.page}")
        parts.append(f"chunk {self.chunk_index}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "source": self.source,
                "chunk_index": self.chunk_index, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(id=data["id"], text=data["text"], source=data["source"],
                   chunk_index=int(data["chunk_index"]), metadata=dict(data.get("metadata") or {}))


_SPACE_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Unify newlines, collapse runs of spaces and blank lines, strip line ends. Keeps paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = _SPACE_RUN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) character spans covering `text`; each span is at most chunk_size long."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    length = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _snap_end(text, start, end)
        spans.append((start, end))
        if end >= length:
            break
        # Next window begins `chunk_overlap` back - capped at half the chunk so we always move forward -
        # then nudged to a word boundary.
        next_start = max(end - chunk_overlap, start + (end - start) // 2, start + 1)
        start = _snap_start(text, next_start, end)
    return spans


def _snap_end(text: str, start: int, end: int) -> int:
    """Prefer cutting at a natural boundary in the second half of the window (never in the first half,
    which would produce tiny chunks)."""
    window = text[start:end]
    floor = len(window) // 2
    for separator in SEPARATORS:
        index = window.rfind(separator)
        if index >= floor:
            return start + index + len(separator)
    return end


def _snap_start(text: str, position: int, limit: int) -> int:
    """Move forward to the next word boundary so a chunk never starts mid-word."""
    index = position
    while index < limit and not text[index - 1].isspace():
        index += 1
    return index if index < limit else position


def chunk_documents(documents: Iterable[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every document; chunk_index counts per source file (across pages for PDFs)."""
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for document in documents:
        text = normalize_text(document.text)
        if not text:
            continue
        for start, end in split_text(text, chunk_size, chunk_overlap):
            piece = text[start:end]
            lead = len(piece) - len(piece.lstrip())
            body = piece.strip()
            if not body:
                continue
            index = counters.get(document.source, 0)
            counters[document.source] = index + 1
            metadata = dict(document.metadata)
            metadata["start_char"] = start + lead
            metadata["end_char"] = start + lead + len(body)
            chunks.append(Chunk(
                id=stable_id(document.source, metadata.get("page"), index, body),
                text=body,
                source=document.source,
                chunk_index=index,
                metadata=metadata,
            ))
    return chunks
