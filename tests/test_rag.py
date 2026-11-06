import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import Mock, patch

from rag import chunks, index, model, search


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        def vector(text):
            return FakeVector([1.0, 0.0] if "kot" in text.lower() else [0.0, 1.0])

        return (
            vector(texts)
            if isinstance(texts, str)
            else [vector(text) for text in texts]
        )


class RagTest(unittest.TestCase):
    def test_model_uses_cuda_when_available_and_cpu_otherwise(self):
        for available, device in ((True, "cuda"), (False, "cpu")):
            with self.subTest(device=device):
                torch = types.ModuleType("torch")
                torch.__spec__ = ModuleSpec("torch", loader=None)
                torch.cuda = types.SimpleNamespace(is_available=lambda: available)
                transformers = types.ModuleType("transformers")
                transformers.__path__ = []
                transformers_utils = types.ModuleType("transformers.utils")
                transformers_utils.logging = types.SimpleNamespace(
                    disable_progress_bar=Mock()
                )
                sentence_transformers = types.ModuleType("sentence_transformers")
                sentence_transformers.SentenceTransformer = Mock()
                model.cache_clear()
                with patch.dict(
                    sys.modules,
                    {
                        "torch": torch,
                        "transformers": transformers,
                        "transformers.utils": transformers_utils,
                        "sentence_transformers": sentence_transformers,
                    },
                ):
                    model()
                sentence_transformers.SentenceTransformer.assert_called_once_with(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    device=device,
                )
        model.cache_clear()

    def test_chunks_are_bounded_and_preserve_text(self):
        text = "a" * 2500
        parts = list(chunks(text))
        self.assertTrue(all(len(part) <= 1000 for part in parts))
        self.assertEqual(parts[0] + "".join(part[150:] for part in parts[1:]), text)

    @patch("rag.model", return_value=FakeModel())
    def test_index_search_rebuild_and_empty(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "knowledge"
            source.mkdir()
            self.assertIn("Brak indeksu", search(root, "kot"))
            (source / "kot.md").write_text("Kot lubi mleko.", encoding="utf-8")
            (source / "pies.md").write_text("Pies lubi spacery.", encoding="utf-8")
            self.assertEqual(index(root), 2)
            _model.reset_mock()
            self.assertEqual(index(root), 0)
            _model.assert_not_called()
            self.assertTrue(search(root, "kot").startswith("Źródło: knowledge/kot.md"))
            (source / "pies.md").write_text("Pies śpi w domu.", encoding="utf-8")
            self.assertEqual(index(root), 1)
            self.assertIn("Pies śpi w domu.", search(root, "pies"))
            self.assertNotIn("Pies lubi spacery.", search(root, "pies"))
            (source / "kot.md").unlink()
            index(root)
            self.assertNotIn("kot.md", search(root, "kot"))
            (source / "pies.md").unlink()
            self.assertEqual(index(root), 0)
            self.assertIn("Brak dokumentów", search(root, "kot"))

    def test_batching_and_recovery_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "knowledge"
            source.mkdir()
            (source / "large.md").write_text("kot " * 20000, encoding="utf-8")
            fake = FakeModel()
            calls = []

            def encode(texts, normalize_embeddings=True):
                calls.append(len(texts))
                if len(calls) == 2:
                    raise RuntimeError("interrupted")
                return fake.encode(texts, normalize_embeddings)

            with patch("rag.model") as mocked:
                mocked.return_value.encode.side_effect = encode
                with self.assertRaises(RuntimeError):
                    index(root)
            self.assertEqual(calls[0], 64)
            with patch("rag.model", return_value=fake):
                count = index(root)
                import lancedb

                table = lancedb.connect(str(root / ".rag" / "lancedb")).open_table(
                    "knowledge"
                )
                self.assertEqual(table.count_rows(), count)
                self.assertGreater(count, 64)


if __name__ == "__main__":
    unittest.main()
