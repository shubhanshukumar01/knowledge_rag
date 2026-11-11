import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.bm25 import BM25, tokenize
from rag.chunking import Chunk
from rag.embeddings import BuiltinEmbedder, normalize
from rag.errors import RAGError
from rag.retriever import HybridRetriever, reciprocal_rank_fusion
from rag.vector_store import VectorStore

TEXTS = [
    "Annual leave is 24 working days per year, accrued monthly.",
    "The per diem for international travel is 95 USD per day.",
    "Business class flights require Vice President approval.",
    "Passwords must be at least 14 characters and MFA is mandatory.",
    "Atlas 3.2 introduced Ask Atlas and row-level security.",
]


def make_chunks(texts, source="doc.md"):
    return [Chunk(id=f"c{i}", text=text, source=source, chunk_index=i) for i, text in enumerate(texts)]


class TokenizeTests(unittest.TestCase):
    def test_lowercases_stems_and_drops_stopwords(self):
        tokens = tokenize("The Policies of expenses, meetings and 99.9% uptime in Atlas 3.2")
        for expected in ("policy", "expense", "meeting", "99.9", "uptime", "3.2"):
            self.assertIn(expected, tokens)
        for stopword in ("the", "of", "and", "in"):
            self.assertNotIn(stopword, tokens)


class BuiltinEmbedderTests(unittest.TestCase):
    def test_unit_vectors_and_determinism(self):
        embedder = BuiltinEmbedder()
        vectors = embedder.embed_documents(TEXTS)
        self.assertEqual(vectors.shape, (5, BuiltinEmbedder.dims))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(vectors, embedder.embed_documents(TEXTS))

    def test_similar_text_scores_highest(self):
        embedder = BuiltinEmbedder()
        vectors = embedder.embed_documents(TEXTS)
        query = embedder.embed_query("how many days of annual leave do I get")
        self.assertEqual(int(np.argmax(vectors @ query)), 0)


class BM25Tests(unittest.TestCase):
    def test_ranks_keyword_matches(self):
        bm25 = BM25([tokenize(text) for text in TEXTS])
        top = bm25.search("per diem international travel", k=3)
        self.assertEqual(top[0][0], 1)
        self.assertEqual(bm25.search("zzzz qqqq"), [])
        self.assertEqual(BM25([]).search("anything"), [])


class VectorStoreTests(unittest.TestCase):
    def test_add_search_save_load_remove(self):
        with tempfile.TemporaryDirectory() as folder:
            store = VectorStore(Path(folder) / "index")
            chunks = make_chunks(TEXTS[:3], "a.md") + make_chunks(TEXTS[3:], "b.md")
            rng = np.random.default_rng(0)
            vectors = normalize(rng.normal(size=(5, 8)))
            store.add(chunks, vectors)
            self.assertEqual(len(store), 5)
            self.assertEqual(store.dims, 8)
            best, score = store.search(vectors[2], k=2)[0]
            self.assertEqual(best, 2)
            self.assertAlmostEqual(score, 1.0, places=5)

            store.manifest = {"embedder": "test"}
            store.save()
            loaded = VectorStore(Path(folder) / "index").load()
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded.manifest["embedder"], "test")
            self.assertEqual(loaded.source_counts(), {"a.md": 3, "b.md": 2})

            self.assertEqual(loaded.remove_sources({"a.md"}), 3)
            self.assertEqual([c.source for c in loaded.chunks], ["b.md", "b.md"])
            self.assertEqual(loaded.vectors.shape, (2, 8))
            self.assertEqual(loaded.remove_sources({"b.md"}), 2)
            self.assertEqual(loaded.search(vectors[0], k=3), [])

    def test_dimension_mismatch_is_explained(self):
        store = VectorStore(Path("unused"))
        store.add(make_chunks(TEXTS[:2]), normalize(np.ones((2, 4))))
        with self.assertRaises(RAGError):
            store.search(np.ones(5), k=1)
        with self.assertRaises(ValueError):
            store.add(make_chunks(TEXTS[:1]), normalize(np.ones((1, 3))))


class FusionTests(unittest.TestCase):
    def test_items_in_both_lists_win(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]], k=60)
        self.assertEqual(max(fused, key=fused.get), 3)
        self.assertGreater(fused[1], fused[2])


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.embedder = BuiltinEmbedder()
        self.store = VectorStore(Path("unused"))
        chunks = make_chunks(TEXTS)
        self.store.add(chunks, self.embedder.embed_documents([c.indexed_text for c in chunks]))
        self.bm25 = BM25([tokenize(c.indexed_text) for c in chunks])

    def test_hybrid_puts_the_right_chunk_first(self):
        retriever = HybridRetriever(self.store, self.embedder, self.bm25, top_k=3, candidates=5)
        results = retriever.retrieve("international travel per diem")
        self.assertEqual(results[0].chunk.id, "c1")
        self.assertEqual(len(results), 3)
        self.assertIn("dense", results[0].why())
        self.assertIn("bm25", results[0].why())

    def test_dense_only_path(self):
        retriever = HybridRetriever(self.store, self.embedder, None, top_k=2, hybrid=False)
        results = retriever.retrieve("password length and MFA")
        self.assertEqual(results[0].chunk.id, "c3")
        self.assertIsNone(results[0].sparse_rank)

    def test_empty_store(self):
        retriever = HybridRetriever(VectorStore(Path("unused")), self.embedder, None)
        self.assertEqual(retriever.retrieve("anything"), [])


if __name__ == "__main__":
    unittest.main()
