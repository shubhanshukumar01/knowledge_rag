#!/usr/bin/env python
"""Chat with the documents in your knowledge folder.

    python chat.py                         interactive chat
    python chat.py --ask "question"        answer one question and exit
    python chat.py --search "question"     show what retrieval returns, without calling the LLM
    python chat.py --verbose               also show the rewritten query, timings and retrieval scores

Inside the chat:  /sources  /reset  /verbose  /config  /help  /exit
"""
from __future__ import annotations

import argparse
import os
import sys

from rag.chat_engine import ChatEngine, ChatTurn
from rag.config import Config, load_config
from rag.errors import RAGError
from rag.pipeline import build_chat_engine, load_retriever
from rag.utils import Style
from rag.vector_store import VectorStore

HELP = """Commands:
  /sources   show the full text of the passages used for the last answer
  /reset     forget the conversation (start a new topic)
  /verbose   toggle rewritten query, timings and retrieval scores
  /config    show the active configuration (keys masked)
  /help      this list
  /exit      quit (Ctrl+C also works)"""


def print_banner(cfg: Config, engine: ChatEngine, store: VectorStore) -> None:
    mode = "hybrid (dense + BM25, fused with RRF)" if engine.retriever.hybrid else "dense only"
    print(Style.bold("Knowledge RAG chat"))
    print(f"  LLM        : {engine.llm.name}")
    print(f"  embeddings : {engine.retriever.embedder.name}")
    print(f"  index      : {len(store)} chunks from {len(store.source_counts())} files  ({cfg.knowledge_dir})")
    print(f"  retrieval  : {mode}, top {cfg.top_k}")
    print(Style.dim("Ask a question, or /help for commands. /exit to quit."))
    print()


def print_sources(turn: ChatTurn, verbose: bool) -> None:
    if not turn.sources:
        print(Style.dim("Sources: none retrieved"))
        return
    print(Style.dim("Sources:"))
    for number, result in enumerate(turn.sources, start=1):
        line = f"  [{number}] {result.chunk.label()}"
        if verbose:
            line += f"    <- {result.why()}"
        print(Style.dim(line))


def print_verbose(turn: ChatTurn) -> None:
    timings = turn.timings
    if turn.rewritten:
        print(Style.dim(f"  search query : {turn.search_query}"))
    print(Style.dim(f"  context      : {turn.context_chars:,} chars (~{turn.context_chars // 4:,} tokens) "
                    f"in {len(turn.sources)} passages"))
    print(Style.dim(f"  timings      : rewrite {timings.get('rewrite', 0):.2f}s, retrieve {timings.get('retrieve', 0):.2f}s, "
                    f"generate {timings.get('generate', 0):.2f}s, total {timings.get('total', 0):.2f}s"))


def answer(engine: ChatEngine, question: str, *, stream: bool, verbose: bool, show_sources: bool) -> ChatTurn:
    print(Style.bold("Assistant: "), end="", flush=True)
    if stream:
        turn = engine.ask(question, on_token=lambda piece: print(piece, end="", flush=True))
        print()
    else:
        turn = engine.ask(question)
        print(turn.answer)
    if show_sources:
        print_sources(turn, verbose)
    if verbose:
        print_verbose(turn)
    return turn


def show_full_sources(turn: ChatTurn | None) -> None:
    if turn is None or not turn.sources:
        print("No sources yet - ask a question first.")
        return
    for number, result in enumerate(turn.sources, start=1):
        print(Style.bold(f"[{number}] {result.chunk.label()}") + Style.dim(f"   {result.why()}"))
        print(result.chunk.text)
        print()


def run_search(cfg: Config, query: str) -> int:
    retriever, _, _ = load_retriever(cfg)
    results = retriever.retrieve(query)
    print(Style.bold(f"Top {len(results)} passages for: ") + query)
    for number, result in enumerate(results, start=1):
        preview = " ".join(result.chunk.text.split())
        if len(preview) > 240:
            preview = preview[:240] + "..."
        print(f"\n{number}. {result.chunk.label()}   " + Style.dim(result.why()))
        print("   " + preview)
    return 0


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--search", metavar="QUERY", help="show retrieved passages only (no LLM call)")
    parser.add_argument("--verbose", action="store_true", help="show rewritten query, timings and scores")
    parser.add_argument("--no-stream", action="store_true", help="print the answer only when complete")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--top-k", type=int, help="override retrieval.top_k from the config")
    parser.add_argument("--config", help="path to a config file (default: config.yaml next to this script)")
    args = parser.parse_args(argv)
    Style.init(enabled=False if args.no_color else None)
    _utf8_console()

    cfg = load_config(args.config)
    if args.top_k:
        cfg.top_k = args.top_k
    if args.search:
        return run_search(cfg, args.search)

    engine, store = build_chat_engine(cfg)
    stream = cfg.stream and not args.no_stream
    verbose = args.verbose
    if args.ask:
        answer(engine, args.ask, stream=stream, verbose=verbose, show_sources=cfg.show_sources)
        return 0

    print_banner(cfg, engine, store)
    last_turn: ChatTurn | None = None
    while True:
        try:
            question = input(Style.cyan("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.startswith("/"):
            command = question.split()[0].lower()
            if command in ("/exit", "/quit", "/q"):
                break
            if command == "/reset":
                engine.reset()
                print(Style.dim("(conversation cleared)"))
            elif command == "/verbose":
                verbose = not verbose
                print(Style.dim(f"(verbose {'on' if verbose else 'off'})"))
            elif command == "/sources":
                show_full_sources(last_turn)
            elif command == "/config":
                print(cfg.describe())
            elif command == "/help":
                print(HELP)
            else:
                print(Style.yellow("Unknown command.") + "\n" + HELP)
            print()
            continue
        try:
            last_turn = answer(engine, question, stream=stream, verbose=verbose, show_sources=cfg.show_sources)
        except KeyboardInterrupt:
            print(Style.yellow("\n(interrupted)"))
        except RAGError as exc:
            print(Style.red(f"\nError: {exc}"))
        except Exception as exc:  # noqa: BLE001 - keep the chat alive and show the provider's message
            if os.environ.get("RAG_DEBUG"):
                raise
            print(Style.red(f"\n{type(exc).__name__}: {exc}"))
        print()
    print(Style.dim("Bye."))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RAGError as exc:
        print(Style.red(f"Error: {exc}"), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("RAG_DEBUG"):
            raise
        print(Style.red(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        sys.exit(1)
