"""Embedding models: text -> unit-length vectors whose dot product is cosine similarity.

Every provider sits behind the same Embedder interface (embed_documents / embed_query), so the rest
of the pipeline never knows which model is in use. Vectors are L2-normalised once here, which turns
cosine similarity into a plain matrix product in the vector store.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .bm25 import tokenize
from .config import Config, require_api_key
from .errors import ConfigError, MissingDependency
from .utils import ollama_post, retry

Progress = Callable[[int, int], None]


def normalize(matrix) -> np.ndarray:
    """Scale every row to unit length (zero rows are left as zeros)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embedder(ABC):
    """Common interface. Subclasses only implement _embed_batch; batching, retries and normalisation live here."""

    name: str = "base"

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = max(1, int(batch_size))

    @abstractmethod
    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """Return one vector per text. is_query lets asymmetric models use their query-side encoding."""

    def embed_documents(self, texts: list[str], progress: Progress | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text if text.strip() else " " for text in texts[start:start + self.batch_size]]
            vectors.extend(retry(lambda b=batch: self._embed_batch(b, is_query=False), label=f"{self.name}"))
            if progress:
                progress(min(start + len(batch), len(texts)), len(texts))
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vector = retry(lambda: self._embed_batch([text if text.strip() else " "], is_query=True), label=self.name)
        return normalize(np.asarray(vector, dtype=np.float32))[0]


# ---------------------------------------------------------------------------------------
# providers

class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small (1536 dims) / -large (3072). `dimensions` can shrink the vectors."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimensions: int | None = None,
                 base_url: str | None = None, batch_size: int = 64) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise MissingDependency("The openai package is needed for OpenAI embeddings:  pip install openai") from None
        super().__init__(batch_size)
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.dimensions = int(dimensions) if dimensions else None
        self.name = f"openai/{model}" + (f"@{self.dimensions}" if self.dimensions else "")

    def _embed_batch(self, texts, *, is_query):
        kwargs = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


class GeminiEmbedder(Embedder):
    """Google gemini-embedding-001 (default 3072 dims; 768 or 1536 recommended for storage). Uses the
    asymmetric task types RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY the model was trained with."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dimensions: int | None = 768,
                 batch_size: int = 64) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise MissingDependency("The google-genai package is needed for Gemini:  pip install google-genai") from None
        super().__init__(min(batch_size, 100))        # the API accepts at most 100 texts per call
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = int(dimensions) if dimensions else None
        self.name = f"gemini/{model}" + (f"@{self.dimensions}" if self.dimensions else "")

    def _embed_batch(self, texts, *, is_query):
        config = self.types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
            output_dimensionality=self.dimensions,
        )
        response = self.client.models.embed_content(model=self.model, contents=texts, config=config)
        return [embedding.values for embedding in response.embeddings]


class OllamaEmbedder(Embedder):
    """A local embedding model served by Ollama (nomic-embed-text, mxbai-embed-large, bge-m3 ...)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text",
                 batch_size: int = 32) -> None:
        super().__init__(batch_size)
        self.base_url = base_url
        self.model = model
        self.name = f"ollama/{model}"
        self._prefixed = "nomic" in model.lower()   # nomic models were trained with these task prefixes

    def _embed_batch(self, texts, *, is_query):
        if self._prefixed:
            prefix = "search_query: " if is_query else "search_document: "
            texts = [prefix + text for text in texts]
        data = ollama_post(self.base_url, "/api/embed", {"model": self.model, "input": texts})
        return data["embeddings"]


class LocalEmbedder(Embedder):
    """sentence-transformers running in this process (optional: pip install sentence-transformers).
    Downloads the model once (~90 MB for all-MiniLM-L6-v2); after that it is fully offline."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise MissingDependency(
                "embeddings.provider 'local' needs sentence-transformers:  pip install sentence-transformers") from None
        super().__init__(batch_size)
        self.model = SentenceTransformer(model)
        self.name = f"local/{model}"

    def _embed_batch(self, texts, *, is_query):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()


class BuiltinEmbedder(Embedder):
    """Zero-dependency fallback: hashed word, bigram and character-trigram features in 1024 dimensions.

    Deterministic and offline, so the test suite and a no-network demo run without any model. It is
    LEXICAL, not semantic ("holiday" and "annual leave" do not match) - use a real embedding model for
    anything that matters. It exists to show what an embedding is: a fixed-size vector whose dot product
    measures similarity.
    """

    dims = 1024
    name = "builtin/hashed-ngrams-1024"

    def _embed_batch(self, texts, *, is_query):
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dims, dtype=np.float32)
        tokens = tokenize(text)
        features: list[tuple[str, float]] = [(token, 2.0) for token in tokens]
        features += [(f"{a}_{b}", 1.5) for a, b in zip(tokens, tokens[1:])]
        for token in tokens:
            padded = f"#{token}#"
            features += [(padded[i:i + 3], 0.5) for i in range(len(padded) - 2)]
        for feature, weight in features:
            digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
            sign = 1.0 if digest & 1 else -1.0
            vector[(digest >> 1) % self.dims] += sign * weight
        return (np.sign(vector) * np.log1p(np.abs(vector))).tolist()   # dampen very frequent features


# ---------------------------------------------------------------------------------------

def get_embedder(cfg: Config) -> Embedder:
    """Build the embedder selected by embeddings.provider in config.yaml."""
    provider = cfg.embedding_provider
    settings = cfg.embedding_settings
    batch = cfg.embedding_batch_size
    if provider == "openai":
        return OpenAIEmbedder(api_key=require_api_key(cfg, "openai"),
                              model=settings.get("embedding_model") or "text-embedding-3-small",
                              dimensions=settings.get("embedding_dimensions"), batch_size=batch)
    if provider == "gemini":
        return GeminiEmbedder(api_key=require_api_key(cfg, "gemini"),
                              model=settings.get("embedding_model") or "gemini-embedding-001",
                              dimensions=settings.get("embedding_dimensions", 768), batch_size=batch)
    if provider == "ollama":
        return OllamaEmbedder(base_url=settings.get("base_url") or "http://localhost:11434",
                              model=settings.get("embedding_model") or "nomic-embed-text", batch_size=batch)
    if provider == "local":
        return LocalEmbedder(model=settings.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2",
                             batch_size=batch)
    if provider == "builtin":
        return BuiltinEmbedder(batch_size=batch)
    raise ConfigError(f"Unknown embeddings.provider '{provider}'.")
