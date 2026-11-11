"""The conversational loop: rewrite -> retrieve -> augment -> generate, with bounded memory.

ask() does, in order:
  1. rewrite a follow-up ("and for P2?") into a standalone search query using the recent conversation;
  2. retrieve the best chunks for that query (hybrid search);
  3. build the prompt = system rules + recent history + numbered context + the question;
  4. call the LLM (streaming through on_token when the caller wants it);
  5. remember the plain question/answer pair.
The retrieved context is injected per turn and never stored in the history, so the prompt size stays
bounded no matter how long the conversation runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .bm25 import tokenize
from .llm import LLM, Message
from .prompts import REWRITE_PROMPT, SYSTEM_PROMPT, build_user_message, format_context, format_history
from .retriever import HybridRetriever, RetrievedChunk
from .utils import timed

# Words that usually mean "this question leans on the previous turn".
FOLLOWUP_CUES = frozenset("it its that those these this they them their he she him her same also else too then".split())
FOLLOWUP_OPENERS = ("and ", "what about", "how about", "also ", "then ", "but ", "or ", "so ", "plus ")


def looks_like_followup(question: str) -> bool:
    """Cheap gate in front of the rewrite call: short questions, or ones that lean on the conversation.
    Long, self-contained questions skip the extra LLM call (and cannot be over-rewritten by a small model)."""
    text = question.strip().lower()
    words = re.findall(r"[a-z0-9']+", text)
    return len(words) <= 6 or text.startswith(FOLLOWUP_OPENERS) or any(word in FOLLOWUP_CUES for word in words)


@dataclass
class ChatTurn:
    question: str
    search_query: str
    answer: str
    sources: list[RetrievedChunk]
    context_chars: int
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def rewritten(self) -> bool:
        return self.search_query.strip() != self.question.strip()


class ChatEngine:
    def __init__(self, llm: LLM, retriever: HybridRetriever, *, max_history_turns: int = 6,
                 rewrite_followups: bool = True, max_context_chars: int = 12000,
                 system_prompt: str = SYSTEM_PROMPT) -> None:
        self.llm = llm
        self.retriever = retriever
        self.max_history_turns = max_history_turns
        self.rewrite_followups = rewrite_followups
        self.max_context_chars = max_context_chars
        self.system_prompt = system_prompt
        self.history: list[Message] = []

    def reset(self) -> None:
        self.history = []

    def _recent_history(self) -> list[Message]:
        return self.history[-2 * self.max_history_turns:] if self.max_history_turns else []

    def rewrite_query(self, question: str) -> str:
        """Turn a follow-up into a standalone query; the first question of a conversation is used as is."""
        if not self.rewrite_followups or not self.history or not looks_like_followup(question):
            return question
        prompt = REWRITE_PROMPT.format(history=format_history(self.history, self.max_history_turns), question=question)
        rewritten = self.llm.generate([{"role": "user", "content": prompt}]).strip()
        rewritten = rewritten.splitlines()[0].strip().strip('"').strip("'") if rewritten else ""
        if not rewritten or len(rewritten) > 400:
            return question
        # Guard against over-rewriting: the rewrite must keep at least one content word of the original question.
        original_terms = set(tokenize(question))
        if original_terms and not original_terms & set(tokenize(rewritten)):
            return question
        return rewritten

    def ask(self, question: str, on_token: Callable[[str], None] | None = None) -> ChatTurn:
        question = question.strip()
        timings: dict[str, float] = {}

        with timed(timings, "rewrite"):
            search_query = self.rewrite_query(question)
        with timed(timings, "retrieve"):
            results = self.retriever.retrieve(search_query)
        context, used = format_context(results, self.max_context_chars)
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            *self._recent_history(),
            {"role": "user", "content": build_user_message(question, context, search_query)},
        ]

        with timed(timings, "generate"):
            if on_token is not None:
                pieces: list[str] = []
                for piece in self.llm.stream(messages):
                    pieces.append(piece)
                    on_token(piece)
                answer = "".join(pieces).strip()
            else:
                answer = self.llm.generate(messages).strip()
        timings["total"] = sum(timings.values())

        self.history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        self.history = self._recent_history()
        return ChatTurn(question, search_query, answer, used, len(context), timings)
