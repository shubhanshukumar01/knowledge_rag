"""Prompt templates: the grounding rules, the numbered-context block, and the follow-up rewrite prompt."""
from __future__ import annotations

from .retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a precise assistant for a private knowledge base. Answer using ONLY the numbered context passages supplied in the user's message.

Rules:
1. Ground every statement in the passages. Quote numbers, dates, names and limits exactly as written.
2. After each sentence or bullet that uses a passage, cite it with its number in square brackets, like [1] or [2][3]. Always write the answer in words first; never reply with citations alone.
3. If the passages do not contain the answer, say: "I couldn't find that in the knowledge base." Then, if the passages contain something related, say briefly what they do cover.
4. If passages disagree, say so and cite both.
5. Be concise: a short paragraph or a few bullets. No preamble and do not repeat the question."""

REWRITE_PROMPT = """Below is a conversation between a user and an assistant, followed by the user's new question.
Rewrite the new question as ONE self-contained question that includes every name, topic and constraint it refers to, so a search engine could answer it without seeing the conversation.
If the new question is already self-contained, or is about a different topic from the conversation, return it EXACTLY as written.
Reply with the rewritten question only - no explanation, no quotes.

Conversation:
{history}

New question: {question}"""


def format_context(results: list[RetrievedChunk], max_chars: int) -> tuple[str, list[RetrievedChunk]]:
    """Number the passages and stop adding when the character budget is reached (always keeps the first)."""
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0
    for number, result in enumerate(results, start=1):
        block = f"[{number}] Source: {result.chunk.label()}\n{result.chunk.text}"
        if used and total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(result)
        total += len(block) + 2
    return "\n\n".join(blocks), used


def build_user_message(question: str, context: str, search_query: str | None = None) -> str:
    note = ""
    if search_query and search_query.strip() != question.strip():
        note = f"\n(The question was interpreted as: {search_query})"
    if not context:
        context = "(no passages were retrieved)"
    return f"Context passages:\n\n{context}\n\n---\nQuestion: {question}{note}"


def format_history(history: list[dict], max_turns: int = 6) -> str:
    recent = history[-2 * max_turns:] if max_turns else []
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'].strip()}" for m in recent]
    return "\n".join(lines) or "(none)"
