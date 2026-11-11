#!/usr/bin/env python
"""Measure retrieval quality (and optionally answer quality) on a list of known questions.

    python evaluate.py                     Recall@k and MRR for eval/questions.jsonl (no LLM calls)
    python evaluate.py --k 3               use a different cut-off
    python evaluate.py --with-llm          also generate answers and check the expected keywords

eval/questions.jsonl holds one JSON object per line:
    {"question": "...", "expected_source": "file.md" or ["a.md", "b.csv"], "expected_keywords": ["24"]}
A retrieval "hit" means at least one expected source appears in the top-k passages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from rag.chat_engine import ChatEngine
from rag.config import PROJECT_ROOT, load_config
from rag.errors import RAGError
from rag.llm import get_llm
from rag.pipeline import load_retriever
from rag.utils import Style


def load_questions(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            expected = item.get("expected_source") or item.get("expected_sources") or []
            item["expected_sources"] = [expected] if isinstance(expected, str) else list(expected)
            item.setdefault("expected_keywords", [])
            items.append(item)
    return items


def source_matches(source: str, expected: list[str]) -> bool:
    return any(source == name or source.endswith("/" + name) for name in expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default=str(PROJECT_ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--k", type=int, help="cut-off (default: retrieval.top_k from the config)")
    parser.add_argument("--with-llm", action="store_true", help="also generate answers and check keywords")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    Style.init()

    cfg = load_config(args.config)
    retriever, store, _ = load_retriever(cfg)
    k = args.k or cfg.top_k
    questions = load_questions(Path(args.file))
    if not questions:
        raise RAGError(f"No questions found in {args.file}")

    print(Style.bold(f"Retrieval evaluation: {len(questions)} questions, k={k}, index of {len(store)} chunks"))
    print()
    hits = 0
    reciprocal = 0.0
    for item in questions:
        results = retriever.retrieve(item["question"], top_k=k)
        rank = next((position for position, result in enumerate(results, start=1)
                     if source_matches(result.chunk.source, item["expected_sources"])), None)
        if rank:
            hits += 1
            reciprocal += 1.0 / rank
        mark = Style.green("hit ") if rank else Style.red("miss")
        top = results[0].chunk.source if results else "-"
        print(f"  {mark}  rank {str(rank or '-'):>2}  {item['question'][:62]:<62}  top: {top}")
    count = len(questions)
    print()
    print(Style.bold(f"  Recall@{k} = {hits}/{count} = {hits / count:.0%}     MRR = {reciprocal / count:.2f}"))

    if args.with_llm:
        llm = get_llm(cfg)
        engine = ChatEngine(llm, retriever, max_history_turns=0, rewrite_followups=False,
                            max_context_chars=cfg.max_context_chars)
        scored = [item for item in questions if item["expected_keywords"]]
        print()
        print(Style.bold(f"Answer check with {llm.name}: {len(scored)} questions"))
        print()
        passed = 0
        started = time.perf_counter()
        for item in scored:
            turn = engine.ask(item["question"])
            lowered = turn.answer.lower()
            ok = all(keyword.lower() in lowered for keyword in item["expected_keywords"])
            passed += int(ok)
            print(f"  {Style.green('pass') if ok else Style.red('fail')}  {item['question'][:62]:<62}  "
                  f"expects {item['expected_keywords']}")
            if not ok:
                print(Style.dim("        answer: " + " ".join(turn.answer.split())[:220]))
        print()
        print(Style.bold(f"  keyword accuracy = {passed}/{len(scored)} = {passed / max(1, len(scored)):.0%}"
                         f"   ({time.perf_counter() - started:.1f}s)"))
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
