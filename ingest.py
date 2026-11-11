#!/usr/bin/env python
"""Build or update the search index from the files in the knowledge folder.

    python ingest.py              incremental: only new or changed files are re-embedded, deleted files are dropped
    python ingest.py --rebuild    start from scratch (needed after changing the embedding model)
"""
from __future__ import annotations

import argparse
import os
import sys

from rag.config import load_config
from rag.errors import RAGError
from rag.pipeline import ingest
from rag.utils import Style


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="ignore the existing index and re-embed every file")
    parser.add_argument("--config", help="path to a config file (default: config.yaml next to this script)")
    args = parser.parse_args(argv)
    Style.init()

    cfg = load_config(args.config)
    print(Style.bold("Knowledge RAG - ingest"))
    print(f"  knowledge folder : {cfg.knowledge_dir}")
    print(f"  index folder     : {cfg.index_dir}")
    print(f"  embeddings       : {cfg.embedding_provider} / "
          f"{cfg.embedding_settings.get('embedding_model') or ('builtin' if cfg.embedding_provider == 'builtin' else '-')}")
    print(f"  chunking         : {cfg.chunk_size} chars, overlap {cfg.chunk_overlap}")
    print()

    def progress(done: int, total: int) -> None:
        print(f"\r  embedding chunks {done}/{total}", end="", flush=True)
        if done >= total:
            print()

    report = ingest(cfg, rebuild=args.rebuild, log=print, progress=progress)

    print()
    print(Style.green("Index updated") + (" (full rebuild)" if report.full_rebuild else " (incremental)"))
    print(f"  files            : {report.files_total} total, {report.files_indexed} indexed now, "
          f"{report.files_removed} removed")
    print(f"  chunks           : {report.chunks_total} total  (+{report.chunks_added} / -{report.chunks_removed})")
    print(f"  vectors          : {report.dims} dimensions, embedder {report.embedder}")
    print(f"  time             : {report.seconds:.1f}s")
    if report.files_skipped:
        print(Style.yellow(f"  skipped          : {', '.join(report.files_skipped)}"))
    if report.files_indexed == 0 and report.files_removed == 0:
        print(Style.dim("  nothing changed since the last run"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RAGError as exc:
        print(Style.red(f"\nError: {exc}"), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - show provider errors as one line unless debugging
        if os.environ.get("RAG_DEBUG"):
            raise
        print(Style.red(f"\n{type(exc).__name__}: {exc}"), file=sys.stderr)
        sys.exit(1)
