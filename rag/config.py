"""Load and validate config.yaml.

Everything the pipeline can be tuned with lives in config.yaml. This module turns it into a typed
Config object, fills API keys from environment variables when the YAML leaves them blank, resolves
relative paths against the project folder, and rejects impossible settings early with a clear message.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError, MissingDependency

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLM_PROVIDERS = ("openai", "anthropic", "gemini", "grok", "ollama")
EMBEDDING_PROVIDERS = ("openai", "gemini", "ollama", "local", "builtin")

# Environment variables consulted when providers.<name>.api_key is blank in config.yaml.
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
}

# Providers that cannot work without a key. (anthropic is lenient: its SDK can also pick up an
# `ant auth login` profile; ollama / builtin / local run on this machine.)
KEY_REQUIRED = {"openai", "gemini", "grok"}

_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass
class Config:
    path: Path
    llm_provider: str
    embedding_provider: str
    providers: dict[str, dict[str, Any]]
    knowledge_dir: Path
    index_dir: Path
    include_extensions: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    top_k: int
    candidates: int
    hybrid: bool
    rrf_k: int
    min_score: float
    max_context_chars: int
    temperature: float
    max_tokens: int
    max_history_turns: int
    rewrite_followups: bool
    stream: bool
    show_sources: bool

    # -- convenience ---------------------------------------------------------------------
    def provider(self, name: str) -> dict[str, Any]:
        return self.providers.get(name, {})

    @property
    def llm_settings(self) -> dict[str, Any]:
        return self.provider(self.llm_provider)

    @property
    def embedding_settings(self) -> dict[str, Any]:
        return self.provider(self.embedding_provider)

    def describe(self) -> str:
        """Human-readable summary with secrets masked (used by `/config` in the chat and by check_setup)."""
        llm = self.llm_settings
        emb = self.embedding_settings
        lines = [
            f"config file      : {self.path}",
            f"LLM              : {self.llm_provider} / {llm.get('model', '-')}",
            f"embeddings       : {self.embedding_provider} / {emb.get('embedding_model', '-') or '-'}",
            f"knowledge folder : {self.knowledge_dir}",
            f"index folder     : {self.index_dir}",
            f"chunking         : {self.chunk_size} chars, overlap {self.chunk_overlap}",
            f"retrieval        : top_k={self.top_k}, hybrid={'on' if self.hybrid else 'off'}, "
            f"candidates={self.candidates}, context budget {self.max_context_chars} chars",
            f"chat             : history {self.max_history_turns} turns, rewrite follow-ups "
            f"{'on' if self.rewrite_followups else 'off'}, streaming {'on' if self.stream else 'off'}",
            "api keys         :",
        ]
        for name in LLM_PROVIDERS:
            if name == "ollama":
                continue
            settings = self.provider(name)
            source = settings.get("api_key_source") or "not set"
            lines.append(f"  {name:<10} {source}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# loading

def default_config_path() -> Path:
    """config.yaml if present, otherwise config.example.yaml (so a fresh clone still runs)."""
    main = PROJECT_ROOT / "config.yaml"
    if main.exists():
        return main
    example = PROJECT_ROOT / "config.example.yaml"
    if example.exists():
        return example
    raise ConfigError(f"No config.yaml found in {PROJECT_ROOT}. Copy config.example.yaml to config.yaml.")


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path).expanduser() if path else default_config_path()
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        import yaml  # PyYAML
    except ImportError:  # pragma: no cover - depends on the environment
        raise MissingDependency("PyYAML is required to read config.yaml:  pip install pyyaml") from None

    with open(cfg_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a YAML mapping at the top level.")

    llm = _section(raw, "llm")
    embeddings = _section(raw, "embeddings")
    paths = _section(raw, "paths")
    ingestion = _section(raw, "ingestion")
    retrieval = _section(raw, "retrieval")
    chat = _section(raw, "chat")
    providers_raw = _section(raw, "providers")

    llm_provider = str(llm.get("provider", "")).strip().lower()
    embedding_provider = str(embeddings.get("provider", "")).strip().lower()
    if llm_provider not in LLM_PROVIDERS:
        raise ConfigError(f"llm.provider must be one of {', '.join(LLM_PROVIDERS)} (got '{llm_provider}').")
    if embedding_provider not in EMBEDDING_PROVIDERS:
        raise ConfigError(
            f"embeddings.provider must be one of {', '.join(EMBEDDING_PROVIDERS)} (got '{embedding_provider}'). "
            "Note: anthropic and grok have no embedding models - pair them with openai, gemini, ollama or builtin.")

    providers = {name: _resolve_provider(name, settings or {}) for name, settings in providers_raw.items()
                 if isinstance(settings, dict) or settings is None}
    for name in LLM_PROVIDERS + EMBEDDING_PROVIDERS:
        providers.setdefault(name, _resolve_provider(name, {}))

    extensions = ingestion.get("include_extensions") or [".txt", ".md", ".pdf", ".docx", ".csv", ".html", ".json"]
    if isinstance(extensions, str):
        extensions = [extensions]

    cfg = Config(
        path=cfg_path,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        providers=providers,
        knowledge_dir=_resolve_path(paths.get("knowledge_dir", "knowledge")),
        index_dir=_resolve_path(paths.get("index_dir", "index")),
        include_extensions=tuple(_norm_ext(e) for e in extensions),
        chunk_size=_as_int(ingestion.get("chunk_size", 800), "ingestion.chunk_size"),
        chunk_overlap=_as_int(ingestion.get("chunk_overlap", 150), "ingestion.chunk_overlap"),
        embedding_batch_size=_as_int(embeddings.get("batch_size", 64), "embeddings.batch_size"),
        top_k=_as_int(retrieval.get("top_k", 5), "retrieval.top_k"),
        candidates=_as_int(retrieval.get("candidates", 20), "retrieval.candidates"),
        hybrid=bool(retrieval.get("hybrid", True)),
        rrf_k=_as_int(retrieval.get("rrf_k", 60), "retrieval.rrf_k"),
        min_score=float(retrieval.get("min_score", 0.0) or 0.0),
        max_context_chars=_as_int(retrieval.get("max_context_chars", 12000), "retrieval.max_context_chars"),
        temperature=float(llm.get("temperature", 0.2) if llm.get("temperature") is not None else 0.2),
        max_tokens=_as_int(llm.get("max_tokens", 8192), "llm.max_tokens"),
        max_history_turns=_as_int(chat.get("max_history_turns", 6), "chat.max_history_turns"),
        rewrite_followups=bool(chat.get("rewrite_followups", True)),
        stream=bool(chat.get("stream", True)),
        show_sources=bool(chat.get("show_sources", True)),
    )
    _validate(cfg)
    return cfg


def require_api_key(cfg: Config, provider: str) -> str | None:
    """Return the key for a provider, or raise a ConfigError that says exactly where to put one."""
    key = (cfg.provider(provider).get("api_key") or "").strip()
    if key:
        return key
    if provider in KEY_REQUIRED:
        env = " or ".join(ENV_KEYS.get(provider, ())) or "the provider's environment variable"
        raise ConfigError(
            f"No API key for '{provider}'. Paste it in config.yaml under providers.{provider}.api_key, "
            f"or set the environment variable {env}.")
    return None


# ---------------------------------------------------------------------------------------
# helpers

def _section(raw: dict, name: str) -> dict:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' in config.yaml must be a mapping.")
    return value


def _resolve_provider(name: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Copy the provider block and resolve its api_key: literal value, ${ENV_VAR} reference, or env fallback."""
    resolved = dict(settings)
    key = str(settings.get("api_key") or "").strip()
    source = ""
    match = _ENV_REFERENCE.match(key)
    if match:
        env_name = match.group(1)
        key = os.environ.get(env_name, "").strip()
        source = f"from ${env_name}" if key else f"${env_name} is not set"
    elif key:
        source = f"from config.yaml (ends with ...{key[-4:]})"
    else:
        for env_name in ENV_KEYS.get(name, ()):
            value = os.environ.get(env_name, "").strip()
            if value:
                key, source = value, f"from environment variable {env_name}"
                break
    resolved["api_key"] = key
    resolved["api_key_source"] = source
    return resolved


def _resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _norm_ext(ext: str) -> str:
    ext = str(ext).strip().lower()
    return ext if ext.startswith(".") else "." + ext


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{field} must be an integer (got {value!r}).") from None


def _validate(cfg: Config) -> None:
    if cfg.chunk_size < 50:
        raise ConfigError("ingestion.chunk_size must be at least 50 characters.")
    if not 0 <= cfg.chunk_overlap < cfg.chunk_size:
        raise ConfigError("ingestion.chunk_overlap must be >= 0 and smaller than chunk_size.")
    if cfg.top_k < 1:
        raise ConfigError("retrieval.top_k must be at least 1.")
    if cfg.candidates < cfg.top_k:
        cfg.candidates = cfg.top_k
    if cfg.max_context_chars < 500:
        raise ConfigError("retrieval.max_context_chars is too small to hold even one chunk.")
    if cfg.max_history_turns < 0:
        raise ConfigError("chat.max_history_turns cannot be negative.")
    if not 0.0 <= cfg.temperature <= 2.0:
        raise ConfigError("llm.temperature must be between 0 and 2.")
    if cfg.embedding_batch_size < 1:
        raise ConfigError("embeddings.batch_size must be at least 1.")
