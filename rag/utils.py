"""Small helpers shared by several stages: hashing, retries, timing, console colours, Ollama HTTP."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from .errors import ProviderError

T = TypeVar("T")


# --------------------------------------------------------------------- hashing

def sha256_file(path, block_size: int = 1 << 20) -> str:
    """Content hash of a file; the ingest manifest uses it to detect new or changed documents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: Any, length: int = 16) -> str:
    """Deterministic short id. (Python's built-in hash() is salted per process, so it is unusable here.)"""
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------- retries

def is_retryable(exc: BaseException) -> bool:
    """Transient = rate limit, server error, timeout or connection problem. Auth/validation errors are not."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    name = type(exc).__name__.lower()
    return any(word in name for word in ("timeout", "connection", "unavailable", "overloaded"))


def retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0, label: str = "") -> T:
    """Call fn(); on a transient error wait 1s, 2s, 4s ... and try again. Permanent errors raise at once."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - anything non-transient is re-raised below
            if attempt == attempts or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  [retry] {label or 'call'} failed ({type(exc).__name__}); retrying in {delay:.0f}s",
                  file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------- timing

@contextmanager
def timed(timings: dict[str, float], key: str) -> Iterator[None]:
    """with timed(t, "retrieve"): ...   ->   t["retrieve"] = seconds elapsed."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start)


def approx_tokens(text: str) -> int:
    """Rough token estimate: about four characters per token for English prose."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------- console colours

class Style:
    """Tiny ANSI helper. Disabled automatically when output is not a terminal."""

    enabled = sys.stdout.isatty()

    @classmethod
    def init(cls, enabled: bool | None = None) -> None:
        if os.name == "nt":
            os.system("")  # switches the classic Windows console into ANSI escape mode
        if enabled is not None:
            cls.enabled = enabled

    @classmethod
    def paint(cls, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if cls.enabled else text

    @classmethod
    def bold(cls, text: str) -> str:
        return cls.paint(text, "1")

    @classmethod
    def dim(cls, text: str) -> str:
        return cls.paint(text, "2")

    @classmethod
    def cyan(cls, text: str) -> str:
        return cls.paint(text, "36")

    @classmethod
    def green(cls, text: str) -> str:
        return cls.paint(text, "32")

    @classmethod
    def yellow(cls, text: str) -> str:
        return cls.paint(text, "33")

    @classmethod
    def red(cls, text: str) -> str:
        return cls.paint(text, "31")


# --------------------------------------------------------------------- Ollama HTTP (standard library only)

def ollama_post(base_url: str, path: str, payload: dict, *, timeout: float = 600) -> dict:
    """POST JSON to the local Ollama server and return the parsed JSON reply."""
    response = _ollama_open(base_url, path, payload, timeout)
    with response:
        return json.loads(response.read().decode("utf-8"))


def ollama_stream(base_url: str, path: str, payload: dict, *, timeout: float = 600) -> Iterator[dict]:
    """POST with "stream": true and yield one parsed JSON object per NDJSON line."""
    response = _ollama_open(base_url, path, payload, timeout)
    with response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def ollama_get(base_url: str, path: str, *, timeout: float = 10) -> dict:
    url = base_url.rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ProviderError(f"Cannot reach Ollama at {base_url} ({exc.reason}). Start it with:  ollama serve") from None


def _ollama_open(base_url: str, path: str, payload: dict, timeout: float):
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error", body)
        except ValueError:
            detail = body
        hint = ""
        if exc.code == 404 and "not found" in str(detail):
            hint = f"\n  Pull it first:  ollama pull {payload.get('model', '<model>')}"
        raise ProviderError(f"Ollama returned HTTP {exc.code}: {detail}{hint}") from None
    except urllib.error.URLError as exc:
        raise ProviderError(f"Cannot reach Ollama at {base_url} ({exc.reason}). Start it with:  ollama serve") from None
