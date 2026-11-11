import json
import tempfile
import unittest
from pathlib import Path

from rag.chat_engine import ChatEngine
from rag.errors import RAGError
from rag.pipeline import build_chat_engine, ingest, load_retriever
from rag.prompts import SYSTEM_PROMPT, format_context
from rag.retriever import RetrievedChunk
from rag.chunking import Chunk
from tests.fakes import FakeLLM
from tests.helpers import write_config

DOCS = {
    "travel.md": "# Travel\n\nInternational travel per diem is 95 USD per day. Domestic per diem is 60 USD per day. "
                 "Business class needs Vice President approval.",
    "leave.md": "# Leave\n\nAnnual leave is 24 working days per year. Sick leave is 10 paid days per year.",
    "security.md": "# Security\n\nP1 incidents are acknowledged within 15 minutes. Passwords need 14 characters.",
}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.knowledge = self.root / "knowledge"
        self.knowledge.mkdir()
        for name, text in DOCS.items():
            (self.knowledge / name).write_text(text, encoding="utf-8")
        self.cfg = write_config(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def quiet_ingest(self, **kwargs):
        return ingest(self.cfg, log=lambda _line: None, **kwargs)

    def test_ingest_is_incremental(self):
        first = self.quiet_ingest()
        self.assertTrue(first.full_rebuild)
        self.assertEqual((first.files_total, first.files_indexed, first.files_removed), (3, 3, 0))
        self.assertGreater(first.chunks_total, 0)
        self.assertEqual(first.dims, 1024)

        second = self.quiet_ingest()
        self.assertFalse(second.full_rebuild)
        self.assertEqual((second.files_indexed, second.chunks_added, second.files_removed), (0, 0, 0))
        self.assertEqual(second.chunks_total, first.chunks_total)

        (self.knowledge / "leave.md").write_text(DOCS["leave.md"] + "\n\nVolunteering: 2 paid days.", encoding="utf-8")
        (self.knowledge / "security.md").unlink()
        (self.knowledge / "new.txt").write_text("Learning budget is 1,200 USD per year.", encoding="utf-8")
        third = self.quiet_ingest()
        self.assertEqual((third.files_total, third.files_indexed, third.files_removed), (3, 2, 1))
        self.assertGreater(third.chunks_removed, 0)

        manifest = json.loads((self.root / "index" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["files"]), {"travel.md", "leave.md", "new.txt"})
        self.assertEqual(manifest["embedder"], "builtin/hashed-ngrams-1024")
        _, store, _ = load_retriever(self.cfg)
        self.assertEqual(set(store.source_counts()), {"travel.md", "leave.md", "new.txt"})
        self.assertEqual(len(store), third.chunks_total)

    def test_changed_chunk_settings_force_a_rebuild(self):
        self.quiet_ingest()
        self.cfg.chunk_size = 120
        report = self.quiet_ingest()
        self.assertTrue(report.full_rebuild)
        self.assertEqual(report.files_indexed, 3)

    def test_embedder_mismatch_is_refused_at_query_time(self):
        self.quiet_ingest()
        self.cfg.embedding_provider = "ollama"                  # constructing the Ollama embedder makes no network call
        with self.assertRaises(RAGError) as caught:
            load_retriever(self.cfg)
        self.assertIn("--rebuild", str(caught.exception))

    def test_missing_index_and_empty_folder_are_explained(self):
        with self.assertRaises(RAGError):
            load_retriever(self.cfg)
        for path in self.knowledge.iterdir():
            path.unlink()
        with self.assertRaises(RAGError):
            self.quiet_ingest()

    def test_chat_engine_grounds_rewrites_and_remembers(self):
        self.quiet_ingest()
        retriever, _, _ = load_retriever(self.cfg)
        llm = FakeLLM(rewrite="What is the domestic travel per diem?")
        engine = ChatEngine(llm, retriever, max_history_turns=3, rewrite_followups=True, max_context_chars=4000)

        first = engine.ask("What is the per diem for international travel?")
        self.assertFalse(first.rewritten)
        self.assertEqual(first.sources[0].chunk.source, "travel.md")
        self.assertTrue(first.answer.startswith("Based on [1] travel.md"))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertIn("[1] Source: travel.md", llm.calls[0][-1]["content"])
        self.assertIn("Question: What is the per diem for international travel?", llm.calls[0][-1]["content"])

        pieces = []
        second = engine.ask("and domestic?", on_token=pieces.append)
        self.assertTrue(second.rewritten)
        self.assertEqual(second.search_query, "What is the domestic travel per diem?")
        self.assertEqual("".join(pieces), second.answer)
        self.assertEqual(len(llm.calls), 3, "one rewrite call plus one answer call")
        answer_messages = llm.calls[-1]
        self.assertEqual([m["role"] for m in answer_messages], ["system", "user", "assistant", "user"])
        self.assertEqual(answer_messages[1]["content"], "What is the per diem for international travel?")
        self.assertNotIn("Context passages", answer_messages[1]["content"], "history stores the plain question")
        self.assertIn("interpreted as: What is the domestic travel per diem?", answer_messages[-1]["content"])
        self.assertEqual(len(engine.history), 4)

        calls_before = len(llm.calls)
        third = engine.ask("Who won the football world cup final in the year 2022?")
        self.assertFalse(third.rewritten, "a long, self-contained question skips the rewrite")
        self.assertEqual(len(llm.calls), calls_before + 1, "no rewrite call was made")

        llm.rewrite = "Completely unrelated rewritten text"
        fourth = engine.ask("and the hotel cap?")
        self.assertFalse(fourth.rewritten, "a rewrite sharing no content word with the question is discarded")

        for _ in range(4):
            engine.ask("How many days of annual leave?")
        self.assertEqual(len(engine.history), 6, "history is capped at max_history_turns pairs")
        engine.reset()
        self.assertEqual(engine.history, [])

    def test_followup_gate(self):
        from rag.chat_engine import looks_like_followup
        for question in ("and for P2?", "what about domestic trips", "Who approves it?", "How long does that take?"):
            self.assertTrue(looks_like_followup(question), question)
        for question in ("Who won the football world cup final in the year 2022?",
                         "What is the maximum hotel rate for international travel in London?"):
            self.assertFalse(looks_like_followup(question), question)

    def test_build_chat_engine_wires_everything(self):
        self.quiet_ingest()
        engine, store = build_chat_engine(self.cfg)
        self.assertEqual(len(store), engine.retriever.store.__len__())
        self.assertEqual(engine.llm.name, "ollama/test")
        self.assertTrue(engine.retriever.hybrid)


class ContextBudgetTests(unittest.TestCase):
    def test_budget_keeps_at_least_one_passage_and_numbers_contiguously(self):
        chunks = [Chunk(id=str(i), text="x" * 300, source=f"f{i}.md", chunk_index=0) for i in range(4)]
        results = [RetrievedChunk(chunk, 1.0) for chunk in chunks]
        context, used = format_context(results, max_chars=700)
        self.assertEqual(len(used), 2)
        self.assertIn("[1] Source: f0.md", context)
        self.assertIn("[2] Source: f1.md", context)
        self.assertNotIn("[3]", context)
        context, used = format_context(results, max_chars=10)
        self.assertEqual(len(used), 1)


if __name__ == "__main__":
    unittest.main()
