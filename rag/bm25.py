"""BM25 keyword (sparse) retrieval, implemented directly from the formula.

Dense embeddings are good at meaning ("holiday" ~ "annual leave") but weak on exact tokens: product
codes, numbers, rare names ("P1", "SOC 2", "3.2"). BM25 scores documents by the query terms they share,
weighting rare terms higher (IDF) and saturating repeated terms (k1) with document-length
normalisation (b). The retriever fuses both rankings.

    score(D, Q) = sum over terms t in Q of  IDF(t) * tf(t, D) * (k1 + 1) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# Words too common to help ranking.
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my no not of on or our shall should so than that the their them then there these they this those to
was we were what when where which who whom why will with would you your
""".split())

# Words, plus numbers/versions that keep their inner dots ("3.2", "99.9", "1,200" -> "1", "200").
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def _stem(word: str) -> str:
    """Very light English suffix stripping so 'policies' ~ 'policy' and 'meetings' ~ 'meeting'."""
    if len(word) <= 3 or word[-1].isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word/number tokens, drop stopwords, lightly stem."""
    return [_stem(token) for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """documents: one token list per chunk (see tokenize)."""
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_len = [len(tokens) for tokens in documents]
        self.avg_len = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.term_freqs = [Counter(tokens) for tokens in documents]
        self.postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, freqs in enumerate(self.term_freqs):
            for term in freqs:
                self.postings[term].append(doc_id)
        self.idf = {term: math.log(1.0 + (self.n_docs - len(ids) + 0.5) / (len(ids) + 0.5))
                    for term, ids in self.postings.items()}

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (doc_id, score) for the query; only documents sharing a term with it are scored."""
        terms = tokenize(query)
        if not terms or not self.n_docs:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id in self.postings[term]:
                tf = self.term_freqs[doc_id][term]
                length_norm = 1.0 - self.b + self.b * self.doc_len[doc_id] / (self.avg_len or 1.0)
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
