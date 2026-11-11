"""Wiring for both halves of the pipeline.

    ingest(cfg)              index time: discover -> load -> chunk -> embed -> store (incremental by file hash)
    build_chat_engine(cfg)   query time: open the index, build the hybrid retriever and the LLM
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .bm25 import BM25, tokenize
from .chat_engine import ChatEngine
from .chunking import chunk_documents
from .config import Config
from .embeddings import Embedder, Progress, get_embedder
from .errors import RAGError
from .llm import get_llm
from .loaders import discover_files, load_file
from .retriever import HybridRetriever
from .utils import sha256_file
from .vector_store import VectorStore

MANIFEST_VERSION = 1


@dataclass
class IngestReport:
    embedder: str = ""
    full_rebuild: bool = False
    files_total: int = 0
    files_indexed: int = 0                                 # new or changed in this run
    files_removed: int = 0
    files_skipped: list[str] = field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_total: int = 0
    dims: int | None = None
    seconds: float = 0.0


def _chunking_signature(cfg: Config) -> dict:
    return {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap}


def ingest(cfg: Config, *, rebuild: bool = False, log: Callable[[str], None] = print,
           progress: Progress | None = None) -> IngestReport:
    started = time.perf_counter()
    report = IngestReport()

    embedder = get_embedder(cfg)
    report.embedder = embedder.name

    store = VectorStore(cfg.index_dir)
    if store.exists() and not rebuild:
        store.load()
        previous = store.manifest
        if previous.get("embedder") != embedder.name or previous.get("chunking") != _chunking_signature(cfg):
            log(f"Index settings changed (was {previous.get('embedder')}, {previous.get('chunking')}); "
                "rebuilding from scratch.")
            store = VectorStore(cfg.index_dir)
            report.full_rebuild = True
    else:
        report.full_rebuild = True

    files = discover_files(cfg.knowledge_dir, cfg.include_extensions)
    report.files_total = len(files)
    if not files:
        raise RAGError(f"No documents found in {cfg.knowledge_dir}. "
                       f"Supported types: {', '.join(sorted(cfg.include_extensions))}")

    by_name = {path.relative_to(cfg.knowledge_dir).as_posix(): path for path in files}
    current_hashes = {name: sha256_file(path) for name, path in by_name.items()}
    previous_hashes = dict(store.manifest.get("files", {}))

    changed = [name for name, digest in current_hashes.items() if previous_hashes.get(name) != digest]
    removed = [name for name in previous_hashes if name not in current_hashes]
    report.files_indexed = len(changed)
    report.files_removed = len(removed)
    report.chunks_removed = store.remove_sources(set(changed) | set(removed))
    for name in removed:
        log(f"  - {name}  (deleted)")

    indexed_hashes = {name: digest for name, digest in previous_hashes.items() if name in current_hashes}
    documents = []
    for name in changed:
        try:
            docs = load_file(by_name[name], cfg.knowledge_dir)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
            log(f"  ! skipped {name}: {exc}")
            report.files_skipped.append(name)
            indexed_hashes.pop(name, None)
            continue
        if not docs:
            log(f"  ! skipped {name}: no text could be extracted (scanned PDF?)")
            report.files_skipped.append(name)
            indexed_hashes.pop(name, None)
            continue
        documents.extend(docs)
        indexed_hashes[name] = current_hashes[name]
        characters = sum(len(d.text) for d in docs)
        log(f"  + {name}  ({characters:,} chars{', ' + str(len(docs)) + ' pages' if len(docs) > 1 else ''})")

    chunks = chunk_documents(documents, cfg.chunk_size, cfg.chunk_overlap)
    if chunks:
        vectors = embedder.embed_documents([chunk.indexed_text for chunk in chunks], progress=progress)
        store.add(chunks, vectors)
    report.chunks_added = len(chunks)

    store.manifest = {
        "version": MANIFEST_VERSION,
        "embedder": embedder.name,
        "dims": store.dims,
        "chunking": _chunking_signature(cfg),
        "files": indexed_hashes,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    store.save()

    report.chunks_total = len(store)
    report.dims = store.dims
    report.seconds = time.perf_counter() - started
    return report


def load_retriever(cfg: Config) -> tuple[HybridRetriever, VectorStore, Embedder]:
    """Open the index and build the hybrid retriever (BM25 is rebuilt from the stored chunks at start-up)."""
    store = VectorStore(cfg.index_dir)
    if not store.exists():
        raise RAGError(f"No index at {cfg.index_dir}. Build it first:  python ingest.py")
    store.load()
    embedder = get_embedder(cfg)
    built_with = store.manifest.get("embedder")
    if built_with and built_with != embedder.name:
        raise RAGError(
            f"The index was built with '{built_with}' but config.yaml now selects '{embedder.name}'. Vectors from "
            f"different models are not comparable - run:  python ingest.py --rebuild")
    bm25 = BM25([tokenize(chunk.indexed_text) for chunk in store.chunks]) if cfg.hybrid else None
    retriever = HybridRetriever(store, embedder, bm25, top_k=cfg.top_k, candidates=cfg.candidates,
                                hybrid=cfg.hybrid, rrf_k=cfg.rrf_k, min_score=cfg.min_score)
    return retriever, store, embedder


def build_chat_engine(cfg: Config) -> tuple[ChatEngine, VectorStore]:
    retriever, store, _ = load_retriever(cfg)
    llm = get_llm(cfg)
    engine = ChatEngine(llm, retriever, max_history_turns=cfg.max_history_turns,
                        rewrite_followups=cfg.rewrite_followups, max_context_chars=cfg.max_context_chars)
    return engine, store
