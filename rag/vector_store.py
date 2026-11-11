"""A small persistent vector index: chunk metadata + a NumPy matrix of unit vectors.

Files in the index folder:
    chunks.jsonl    one chunk per line (text, source, page, offsets)
    vectors.npy     float32 matrix, row i is the embedding of chunk i
    manifest.json   embedder name, dimensions, chunk settings and a content hash per source file

Search is exact (brute-force dot product), which is fast enough for tens of thousands of chunks. For
millions you would swap this class for FAISS, pgvector, Qdrant ... behind the same three methods.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .errors import RAGError


class VectorStore:
    CHUNKS = "chunks.jsonl"
    VECTORS = "vectors.npy"
    MANIFEST = "manifest.json"

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.manifest: dict = {}

    # -- basics ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dims(self) -> int | None:
        return int(self.vectors.shape[1]) if self.vectors.size else None

    def exists(self) -> bool:
        return (self.index_dir / self.MANIFEST).exists()

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return counts

    # -- persistence -----------------------------------------------------------------
    def load(self) -> "VectorStore":
        manifest_path = self.index_dir / self.MANIFEST
        if not manifest_path.exists():
            raise RAGError(f"No index at {self.index_dir}. Build it first:  python ingest.py")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks_path = self.index_dir / self.CHUNKS
        self.chunks = []
        if chunks_path.exists():
            with open(chunks_path, encoding="utf-8") as handle:
                self.chunks = [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]
        vectors_path = self.index_dir / self.VECTORS
        self.vectors = np.load(vectors_path) if vectors_path.exists() else np.zeros((0, 0), dtype=np.float32)
        if self.chunks and self.vectors.shape[0] != len(self.chunks):
            raise RAGError("The index is inconsistent (chunk and vector counts differ). Run:  python ingest.py --rebuild")
        return self

    def save(self) -> None:
        """Write all three files atomically (temp file + rename) so an interrupted save cannot corrupt the index."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_tmp = self.index_dir / (self.CHUNKS + ".tmp")
        with open(chunks_tmp, "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        vectors_tmp = self.index_dir / (self.VECTORS + ".tmp.npy")
        np.save(vectors_tmp, self.vectors)
        manifest_tmp = self.index_dir / (self.MANIFEST + ".tmp")
        manifest_tmp.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(chunks_tmp, self.index_dir / self.CHUNKS)
        os.replace(vectors_tmp, self.index_dir / self.VECTORS)
        os.replace(manifest_tmp, self.index_dir / self.MANIFEST)

    # -- mutation --------------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) == 0:
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(f"Expected {len(chunks)} vectors, got array of shape {vectors.shape}")
        if self.dims is not None and vectors.shape[1] != self.dims:
            raise ValueError(f"Vector size {vectors.shape[1]} does not match the index ({self.dims})")
        self.vectors = vectors.copy() if not self.vectors.size else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def remove_sources(self, sources: set[str]) -> int:
        """Drop every chunk that came from one of these files; returns how many were removed."""
        if not sources or not self.chunks:
            return 0
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.source not in sources]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        return removed

    # -- search ----------------------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (chunk_index, cosine_similarity), best first."""
        if not self.chunks or not self.vectors.size:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dims:
            raise RAGError(
                f"The query vector has {query.shape[0]} dimensions but the index has {self.dims}: the embedding "
                f"model changed since the index was built. Run:  python ingest.py --rebuild")
        scores = self.vectors @ query
        k = max(1, min(int(k), scores.shape[0]))
        top = np.argpartition(-scores, k - 1)[:k]          # O(n) partial sort ...
        top = top[np.argsort(-scores[top])]                # ... then order just the k winners
        return [(int(i), float(scores[i])) for i in top]
