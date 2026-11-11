import unittest

from rag.chunking import Chunk, chunk_documents, normalize_text, split_text
from rag.loaders import Document


class SplitTextTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("hello world", 100, 20), [(0, 11)])

    def test_spans_cover_the_text_respect_size_and_overlap(self):
        text = " ".join(f"word{i}" for i in range(400))
        spans = split_text(text, 120, 30)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        for start, end in spans:
            self.assertLess(start, end)
            self.assertLessEqual(end - start, 120)
        for (start_a, end_a), (start_b, _) in zip(spans, spans[1:]):
            self.assertLess(start_b, end_a, "consecutive chunks should overlap")
            self.assertGreater(start_b, start_a, "chunks must move forward")
            self.assertTrue(text[start_b - 1].isspace(), "chunks start on a word boundary")

    def test_cuts_prefer_sentence_or_paragraph_boundaries(self):
        paragraph = ("Sentence one is here. " * 5).strip()
        text = "\n\n".join([paragraph] * 4)
        spans = split_text(text, 300, 50)
        self.assertGreater(len(spans), 1)
        for start, end in spans[:-1]:
            piece = text[start:end]
            self.assertTrue(piece.endswith(("\n\n", ". ", ".")), repr(piece[-20:]))

    def test_validation(self):
        with self.assertRaises(ValueError):
            split_text("x" * 10, 5, 5)
        with self.assertRaises(ValueError):
            split_text("x", 0, 0)

    def test_text_without_whitespace_still_terminates(self):
        text = "a" * 1000
        spans = split_text(text, 100, 20)
        self.assertEqual(spans[-1][1], 1000)
        self.assertLess(len(spans), 60)


class ChunkDocumentsTests(unittest.TestCase):
    def test_indexes_ids_and_offsets(self):
        doc = Document("Para one.\n\nPara two is a bit longer than paragraph one.\n\nPara three ends it.", "notes/a.md",
                       {"title": "a"})
        chunks = chunk_documents([doc], 40, 8)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))
        self.assertEqual(len({c.id for c in chunks}), len(chunks))
        self.assertEqual([c.id for c in chunks], [c.id for c in chunk_documents([doc], 40, 8)], "ids are stable")
        normalized = normalize_text(doc.text)
        for chunk in chunks:
            self.assertEqual(normalized[chunk.metadata["start_char"]:chunk.metadata["end_char"]], chunk.text)
            self.assertEqual(chunk.source, "notes/a.md")

    def test_pdf_pages_share_one_counter_and_keep_page_numbers(self):
        pages = [Document("page one text " * 10, "x.pdf", {"page": 1}),
                 Document("page two text " * 10, "x.pdf", {"page": 2})]
        chunks = chunk_documents(pages, 80, 10)
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[-1].page, 2)
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))

    def test_indexed_text_label_and_roundtrip(self):
        chunk = Chunk(id="1", text="body", source="travel_policy.pdf", chunk_index=0, metadata={"page": 2})
        self.assertTrue(chunk.indexed_text.startswith("travel policy\n"))
        self.assertEqual(chunk.label(), "travel_policy.pdf, page 2, chunk 0")
        self.assertEqual(Chunk.from_dict(chunk.to_dict()), chunk)

    def test_normalize_text(self):
        self.assertEqual(normalize_text("a  b\r\n\r\n\r\n\r\nc \t d  "), "a b\n\nc d")


if __name__ == "__main__":
    unittest.main()
