"""Test doubles: a deterministic LLM that needs no network."""
from __future__ import annotations

import re

from rag.llm import LLM


class FakeLLM(LLM):
    """Echoes which passages it was given, so tests can assert grounding without a real model.
    Rewrite prompts (recognised by their opening words) return `rewrite` when one is configured."""

    name = "fake/echo"

    def __init__(self, rewrite: str | None = None) -> None:
        self.rewrite = rewrite
        self.calls: list[list[dict]] = []

    def generate(self, messages):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if prompt.startswith("Below is a conversation") and self.rewrite:
            return self.rewrite
        sources = re.findall(r"\[(\d+)\] Source: ([^\n,]+)", prompt)
        if not sources:
            return "I couldn't find that in the knowledge base."
        return "Based on " + ", ".join(f"[{number}] {source}" for number, source in sources) + "."

    def stream(self, messages):
        text = self.generate(messages)
        for start in range(0, len(text), 7):
            yield text[start:start + 7]
