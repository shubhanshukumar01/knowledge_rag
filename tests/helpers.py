"""Helpers shared by the tests: temporary config files and a tiny PDF writer."""
from __future__ import annotations

from pathlib import Path

import yaml

from rag.config import Config, load_config


def write_config(folder: Path, **overrides) -> Config:
    """Write a config.yaml in `folder` that uses the offline builtin embedder, then load it."""
    settings = {
        "llm": {"provider": "ollama", "temperature": 0.0, "max_tokens": 256},
        "embeddings": {"provider": "builtin", "batch_size": 16},
        "providers": {"ollama": {"base_url": "http://localhost:1", "model": "test"},
                      "openai": {"api_key": ""}},
        "paths": {"knowledge_dir": str(folder / "knowledge"), "index_dir": str(folder / "index")},
        "ingestion": {"chunk_size": 200, "chunk_overlap": 40},
        "retrieval": {"top_k": 3, "hybrid": True, "candidates": 10, "max_context_chars": 4000},
        "chat": {"max_history_turns": 3, "rewrite_followups": True, "stream": False},
    }
    for key, value in overrides.items():
        settings[key].update(value)
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return load_config(path)


def write_minimal_pdf(path: Path, lines: list[str]) -> None:
    """A valid single-page PDF with Helvetica text, enough for pypdf to extract the lines."""
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    ops = ["BT", "/F1 12 Tf", "16 TL", "50 750 Td"] + [f"({escape(line)}) Tj T*" for line in lines] + ["ET"]
    stream = "\n".join(ops).encode("latin-1", "replace")
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
        5: b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    }
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for object_id, body in objects.items():
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for object_id in objects:
        out += f"{offsets[object_id]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
