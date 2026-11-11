import importlib.util
import tempfile
import unittest
from pathlib import Path

from rag.loaders import discover_files, load_documents, load_file
from tests.helpers import write_minimal_pdf

HAS_PYPDF = importlib.util.find_spec("pypdf") is not None
HAS_DOCX = importlib.util.find_spec("docx") is not None


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "sub").mkdir()
        (self.root / "notes.md").write_text("# Title\n\nHello **world**.", encoding="utf-8")
        (self.root / "sub" / "plain.txt").write_text("plain text file", encoding="utf-8")
        (self.root / "table.csv").write_text("tier,price\nPro,49\nBusiness,89\n", encoding="utf-8")
        (self.root / "page.html").write_text(
            "<html><head><style>p{}</style><script>var x=1;</script></head>"
            "<body><h1>Heading</h1><p>First &amp; second</p><ul><li>item</li></ul></body></html>", encoding="utf-8")
        (self.root / "data.json").write_text('{"a": 1, "b": [1, 2]}', encoding="utf-8")
        (self.root / ".hidden.md").write_text("hidden", encoding="utf-8")
        (self.root / "~$lock.docx").write_bytes(b"lock")
        (self.root / "empty.txt").write_text("", encoding="utf-8")
        (self.root / "image.png").write_bytes(b"\x89PNG")

    def tearDown(self):
        self._tmp.cleanup()

    def test_discover_filters_hidden_lock_empty_and_unknown_files(self):
        names = [p.relative_to(self.root).as_posix()
                 for p in discover_files(self.root, [".md", ".txt", ".csv", ".html", ".json", ".docx", ".pdf"])]
        self.assertEqual(names, ["data.json", "notes.md", "page.html", "sub/plain.txt", "table.csv"])

    def test_text_markdown_and_metadata(self):
        docs = load_file(self.root / "notes.md", self.root)
        self.assertEqual(len(docs), 1)
        self.assertIn("Hello **world**", docs[0].text)
        self.assertEqual(docs[0].source, "notes.md")
        self.assertEqual(docs[0].metadata["format"], "md")
        self.assertEqual(docs[0].metadata["title"], "notes")
        self.assertEqual(load_file(self.root / "sub" / "plain.txt", self.root)[0].source, "sub/plain.txt")

    def test_csv_rows_keep_column_names(self):
        text = load_file(self.root / "table.csv", self.root)[0].text
        self.assertIn("Columns: tier, price", text)
        self.assertIn("tier: Business; price: 89", text)

    def test_html_strips_tags_scripts_and_styles(self):
        text = load_file(self.root / "page.html", self.root)[0].text
        self.assertIn("Heading", text)
        self.assertIn("First & second", text)
        self.assertNotIn("var x", text)
        self.assertNotIn("p{}", text)

    def test_json_is_pretty_printed(self):
        text = load_file(self.root / "data.json", self.root)[0].text
        self.assertIn('"a": 1', text)

    @unittest.skipUnless(HAS_PYPDF, "pypdf not installed")
    def test_pdf_pages_become_documents_with_page_numbers(self):
        write_minimal_pdf(self.root / "policy.pdf", ["Per diem is 95 USD per day.", "Second line."])
        docs = load_file(self.root / "policy.pdf", self.root)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertIn("95 USD", docs[0].text)

    @unittest.skipUnless(HAS_DOCX, "python-docx not installed")
    def test_docx_keeps_headings_paragraphs_and_tables(self):
        import docx
        document = docx.Document()
        document.add_heading("Onboarding", level=1)
        document.add_paragraph("Finish training within 14 days.")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Milestone"
        table.rows[0].cells[1].text = "Deadline"
        document.save(self.root / "guide.docx")
        text = load_file(self.root / "guide.docx", self.root)[0].text
        self.assertIn("# Onboarding", text)
        self.assertIn("14 days", text)
        self.assertIn("Milestone | Deadline", text)

    @unittest.skipUnless(HAS_DOCX, "python-docx not installed")
    def test_load_documents_reports_broken_files_instead_of_failing(self):
        (self.root / "broken.docx").write_bytes(b"not a docx")
        errors = []
        docs = load_documents(self.root, [".md", ".docx"], on_error=lambda path, exc: errors.append(path.name))
        self.assertEqual([d.source for d in docs], ["notes.md"])
        self.assertEqual(errors, ["broken.docx"])
        with self.assertRaises(Exception):
            load_documents(self.root, [".docx"])


if __name__ == "__main__":
    unittest.main()
