import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from rag.config import load_config, require_api_key
from rag.errors import ConfigError
from tests.helpers import write_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_defaults_paths_and_providers(self):
        cfg = write_config(self.root)
        self.assertEqual(cfg.llm_provider, "ollama")
        self.assertEqual(cfg.embedding_provider, "builtin")
        self.assertEqual(cfg.knowledge_dir, self.root / "knowledge")
        self.assertEqual(cfg.chunk_size, 200)
        self.assertEqual(cfg.rrf_k, 60)                     # default filled in
        self.assertIn("gemini", cfg.providers)              # every provider block exists even if absent in YAML
        self.assertIn("api keys", cfg.describe())

    def test_blank_key_falls_back_to_environment(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-1234"}):
            cfg = write_config(self.root)
            self.assertEqual(require_api_key(cfg, "openai"), "sk-env-1234")
            self.assertIn("OPENAI_API_KEY", cfg.provider("openai")["api_key_source"])
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}):
            cfg = write_config(self.root)
            with self.assertRaises(ConfigError) as caught:
                require_api_key(cfg, "openai")
            self.assertIn("providers.openai.api_key", str(caught.exception))
            self.assertIsNone(require_api_key(cfg, "anthropic"), "anthropic may rely on an SDK profile")

    def test_env_reference_syntax_and_masking(self):
        with mock.patch.dict(os.environ, {"MY_KEY": "sk-secret-abcd"}):
            cfg = write_config(self.root, providers={"openai": {"api_key": "${MY_KEY}"}})
            self.assertEqual(cfg.provider("openai")["api_key"], "sk-secret-abcd")
            self.assertNotIn("sk-secret-abcd", cfg.describe())
        cfg = write_config(self.root, providers={"openai": {"api_key": "sk-literal-wxyz"}})
        self.assertNotIn("sk-literal-wxyz", cfg.describe())
        self.assertIn("wxyz", cfg.provider("openai")["api_key_source"])

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ConfigError):
            write_config(self.root, llm={"provider": "mistral"})
        with self.assertRaises(ConfigError):
            write_config(self.root, embeddings={"provider": "anthropic"})
        with self.assertRaises(ConfigError):
            write_config(self.root, ingestion={"chunk_size": 200, "chunk_overlap": 200})
        with self.assertRaises(ConfigError):
            write_config(self.root, retrieval={"top_k": 0})

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config(self.root / "nope.yaml")
        (self.root / "list.yaml").write_text(yaml.safe_dump([1, 2]), encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.root / "list.yaml")


if __name__ == "__main__":
    unittest.main()
