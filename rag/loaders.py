"""Turn files in the knowledge folder into plain-text Documents.

One loader per format; each returns Document(text, source, metadata). PDFs give one Document per page so
answers can cite page numbers. Optional parsers (pypdf, python-docx) are imported only when needed.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable

from .errors import MissingDependency

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx", ".csv", ".html", ".htm", ".json"}


@dataclass
class Document:
    text: str
    source: str                                     # path relative to the knowledge folder, e.g. "policies/handbook.md"
    metadata: dict = field(default_factory=dict)    # page, format, title ...


def title_from_source(source: str) -> str:
    """'travel_and_expense_policy.pdf' -> 'travel and expense policy' (prepended to every chunk)."""
    return Path(source).stem.replace("_", " ").replace("-", " ").strip()


def discover_files(knowledge_dir: Path, extensions: Iterable[str]) -> list[Path]:
    """All supported, non-empty, non-hidden files under the knowledge folder (recursive, sorted)."""
    knowledge_dir = Path(knowledge_dir)
    if not knowledge_dir.exists():
        return []
    wanted = {e.lower() if e.startswith(".") else "." + e.lower() for e in extensions}
    files: list[Path] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(knowledge_dir).parts
        if any(part.startswith(".") for part in relative_parts):
            continue                                    # hidden files/folders
        if path.name.startswith("~$"):
            continue                                    # Office lock files
        if path.suffix.lower() in wanted and path.stat().st_size > 0:
            files.append(path)
    return files


def load_file(path: Path, knowledge_dir: Path) -> list[Document]:
    """Parse one file into one or more Documents (several for a multi-page PDF)."""
    path = Path(path)
    source = path.relative_to(knowledge_dir).as_posix()
    extension = path.suffix.lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        raise ValueError(f"Unsupported file type '{extension}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    documents = loader(path, source)
    for document in documents:
        document.metadata.setdefault("format", extension.lstrip("."))
        document.metadata.setdefault("title", title_from_source(source))
    return [d for d in documents if d.text and d.text.strip()]


def load_documents(knowledge_dir: Path, extensions: Iterable[str],
                   on_error: Callable[[Path, Exception], None] | None = None) -> list[Document]:
    """Load every supported file. A broken file is reported through on_error and skipped, not fatal."""
    documents: list[Document] = []
    for path in discover_files(knowledge_dir, extensions):
        try:
            documents.extend(load_file(path, knowledge_dir))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort ingestion
            if on_error is None:
                raise
            on_error(path, exc)
    return documents


# ------------------------------------------------------------------------------------
# individual loaders

def _read_text(path: Path) -> str:
    # utf-8-sig also swallows a BOM; errors="replace" keeps going on the odd bad byte.
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _load_text(path: Path, source: str) -> list[Document]:
    return [Document(_read_text(path), source)]


def _load_pdf(path: Path, source: str) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise MissingDependency("Reading PDFs needs pypdf:  pip install pypdf") from None
    reader = PdfReader(str(path))
    documents: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(Document(text, source, {"page": number}))
    return documents


def _load_docx(path: Path, source: str) -> list[Document]:
    try:
        import docx  # python-docx
    except ImportError:
        raise MissingDependency("Reading .docx needs python-docx:  pip install python-docx") from None
    document = docx.Document(str(path))
    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
        parts.append(("# " + text) if style_name.startswith("Heading") else text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [Document("\n\n".join(parts), source)]


def _load_csv(path: Path, source: str) -> list[Document]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = [c.strip() for c in (reader.fieldnames or []) if c]
        lines = [f"Columns: {', '.join(columns)}"] if columns else []
        for row in reader:
            cells = [f"{(key or '').strip()}: {str(value).strip()}"
                     for key, value in row.items() if key and value not in (None, "")]
            if cells:
                lines.append("; ".join(cells))
    return [Document("\n".join(lines), source)]


class _HTMLText(HTMLParser):
    """Strip tags, drop script/style, keep block boundaries as newlines."""

    SKIP = {"script", "style", "noscript", "head", "template"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
             "article", "table", "ul", "ol", "blockquote", "pre", "header", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skipping += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skipping:
            self._skipping -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skipping:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def _load_html(path: Path, source: str) -> list[Document]:
    parser = _HTMLText()
    parser.feed(_read_text(path))
    parser.close()
    return [Document(parser.text(), source)]


def _load_json(path: Path, source: str) -> list[Document]:
    raw = _read_text(path)
    try:
        text = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except ValueError:
        text = raw
    return [Document(text, source)]


_LOADERS: dict[str, Callable[[Path, str], list[Document]]] = {
    **{ext: _load_text for ext in TEXT_EXTENSIONS},
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".csv": _load_csv,
    ".html": _load_html,
    ".htm": _load_html,
    ".json": _load_json,
}
