#!/usr/bin/env python
"""Preflight check before a demo: config, packages, knowledge folder, providers and index.

    python check_setup.py                  run every check for the providers selected in config.yaml
    python check_setup.py --list-models    also list the models your key (or local Ollama) can serve
    python check_setup.py --skip-llm       do not call the chat model (saves one request)
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from collections import Counter

from rag.config import Config, load_config, require_api_key
from rag.embeddings import get_embedder
from rag.errors import RAGError
from rag.llm import get_llm
from rag.loaders import discover_files
from rag.utils import Style, ollama_get
from rag.vector_store import VectorStore

# python package needed per provider (None = standard library only)
PACKAGES = {"openai": "openai", "grok": "openai", "anthropic": "anthropic", "gemini": "google.genai",
            "local": "sentence_transformers", "ollama": None, "builtin": None}
PIP_NAMES = {"google.genai": "google-genai", "sentence_transformers": "sentence-transformers", "docx": "python-docx",
             "yaml": "pyyaml"}


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  {Style.green('PASS')}  {label:<20} {detail}")

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  {Style.yellow('WARN')}  {label:<20} {detail}")

    def fail(self, label: str, detail: str = "") -> None:
        self.failures += 1
        print(f"  {Style.red('FAIL')}  {label:<20} {detail}")


def check_package(report: Report, label: str, module: str, required: bool = True) -> bool:
    try:
        importlib.import_module(module)
        report.ok(label, module)
        return True
    except ImportError:
        pip_name = PIP_NAMES.get(module, module)
        (report.fail if required else report.warn)(label, f"{module} is not installed:  pip install {pip_name}")
        return False


def list_models(cfg: Config) -> None:
    provider = cfg.llm_provider
    settings = cfg.llm_settings
    print()
    print(Style.bold(f"Models available for '{provider}':"))
    try:
        if provider in ("openai", "grok"):
            from openai import OpenAI
            default_url = "https://api.x.ai/v1" if provider == "grok" else None
            client = OpenAI(api_key=require_api_key(cfg, provider), base_url=settings.get("base_url") or default_url)
            names = sorted(model.id for model in client.models.list())
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=cfg.provider("anthropic").get("api_key") or None)
            names = [model.id for model in client.models.list()]
        elif provider == "gemini":
            from google import genai
            client = genai.Client(api_key=require_api_key(cfg, "gemini"))
            names = [model.name.replace("models/", "") for model in client.models.list()]
        else:
            data = ollama_get(settings.get("base_url") or "http://localhost:11434", "/api/tags")
            names = [model["name"] for model in data.get("models", [])]
    except Exception as exc:  # noqa: BLE001
        print(f"  could not list models: {type(exc).__name__}: {exc}")
        return
    for name in names:
        print(f"  {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    Style.init()
    report = Report()
    print(Style.bold("Knowledge RAG - preflight check"))
    print()

    # 1. configuration
    try:
        cfg = load_config(args.config)
    except RAGError as exc:
        report.fail("config", str(exc))
        return 1
    report.ok("config", f"{cfg.path.name}: LLM {cfg.llm_provider}/{cfg.llm_settings.get('model')}, "
                        f"embeddings {cfg.embedding_provider}/{cfg.embedding_settings.get('embedding_model') or 'builtin'}")
    print(Style.dim("\n".join("        " + line for line in cfg.describe().splitlines()[-6:])))

    # 2. packages
    check_package(report, "config package", "yaml")
    check_package(report, "numpy", "numpy")
    for role, provider in (("LLM package", cfg.llm_provider), ("embedding package", cfg.embedding_provider)):
        module = PACKAGES.get(provider)
        if module is None:
            report.ok(role, f"{provider}: standard library only")
        else:
            check_package(report, role, module)
    check_package(report, "PDF support", "pypdf", required=False)
    check_package(report, "DOCX support", "docx", required=False)

    # 3. knowledge folder
    files = discover_files(cfg.knowledge_dir, cfg.include_extensions)
    if files:
        counts = Counter(path.suffix.lower() for path in files)
        report.ok("knowledge folder", f"{len(files)} files: " + ", ".join(f"{n} {ext}" for ext, n in sorted(counts.items())))
    else:
        report.warn("knowledge folder", f"no supported files in {cfg.knowledge_dir}")

    # 4. embeddings provider
    embedder = None
    try:
        started = time.perf_counter()
        embedder = get_embedder(cfg)
        vector = embedder.embed_query("hello world")
        report.ok("embeddings call", f"{embedder.name}: {vector.shape[0]} dims in {time.perf_counter() - started:.2f}s")
    except Exception as exc:  # noqa: BLE001
        report.fail("embeddings call", f"{type(exc).__name__}: {exc}")

    # 5. chat model
    if args.skip_llm:
        report.warn("LLM call", "skipped (--skip-llm)")
    else:
        try:
            started = time.perf_counter()
            llm = get_llm(cfg)
            reply = llm.generate([{"role": "user", "content": "Reply with the single word OK."}])
            report.ok("LLM call", f"{llm.name} replied {reply.strip()[:40]!r} in {time.perf_counter() - started:.2f}s")
        except Exception as exc:  # noqa: BLE001
            report.fail("LLM call", f"{type(exc).__name__}: {exc}")

    # 6. index
    store = VectorStore(cfg.index_dir)
    if not store.exists():
        report.warn("index", "not built yet - run:  python ingest.py")
    else:
        try:
            store.load()
            built_with = store.manifest.get("embedder")
            if embedder is not None and built_with != embedder.name:
                report.fail("index", f"built with {built_with} but config selects {embedder.name} - run:  python ingest.py --rebuild")
            else:
                report.ok("index", f"{len(store)} chunks from {len(store.source_counts())} files, built with {built_with}")
        except RAGError as exc:
            report.fail("index", str(exc))

    if args.list_models:
        list_models(cfg)

    print()
    print(Style.green("All checks passed - ready to demo.") if report.failures == 0
          else Style.red(f"{report.failures} check(s) failed."))
    return 1 if report.failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("RAG_DEBUG"):
            raise
        print(Style.red(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        sys.exit(1)
