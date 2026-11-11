"""Hybrid retrieval: dense (embedding) search + sparse (BM25) search, fused with Reciprocal Rank Fusion.

Each search returns its own ranked candidate list; RRF merges them by rank alone,
    score(d) = sum over lists of 1 / (rrf_k + rank_in_list(d)),
so the two incomparable score scales never have to be calibrated against each other, and a chunk that
both searches like rises to the top.
"""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .chunking import Chunk
from .embeddings import Embedder
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float                        # fused RRF score (or cosine similarity when hybrid is off)
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None

    def why(self) -> str:
        """One-line explanation for --verbose: which search found it and where it ranked."""
        bits = []
        if self.dense_rank is not None:
            bits.append(f"dense #{self.dense_rank} (cos {self.dense_score:.3f})")
        if self.sparse_rank is not None:
            bits.append(f"bm25 #{self.sparse_rank} ({self.sparse_score:.2f})")
        return ", ".join(bits) or "-"


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Merge several ranked id lists. k=60 is the value from the original paper; it keeps the top ranks
    from dominating completely."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, bm25: BM25 | None = None, *,
                 top_k: int = 5, candidates: int = 20, hybrid: bool = True, rrf_k: int = 60,
                 min_score: float = 0.0) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.top_k = top_k
        self.candidates = candidates
        self.hybrid = hybrid and bm25 is not None
        self.rrf_k = rrf_k
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        if len(self.store) == 0:
            return []
        pool = max(self.candidates, top_k)

        dense = self.store.search(self.embedder.embed_query(query), pool)
        if self.min_score > 0:
            dense = [(i, s) for i, s in dense if s >= self.min_score]
        dense_pos = {i: (rank, s) for rank, (i, s) in enumerate(dense, start=1)}

        if not self.hybrid:
            return [RetrievedChunk(self.store.chunks[i], s, rank, s) for i, (rank, s) in dense_pos.items()][:top_k]

        sparse = self.bm25.search(query, pool)
        sparse_pos = {i: (rank, s) for rank, (i, s) in enumerate(sparse, start=1)}
        fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]], k=self.rrf_k)
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]

        results: list[RetrievedChunk] = []
        for i, score in ordered:
            d = dense_pos.get(i)
            s = sparse_pos.get(i)
            results.append(RetrievedChunk(
                chunk=self.store.chunks[i], score=score,
                dense_rank=d[0] if d else None, dense_score=d[1] if d else None,
                sparse_rank=s[0] if s else None, sparse_score=s[1] if s else None,
            ))
        return results
