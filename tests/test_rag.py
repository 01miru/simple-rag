import json
import re
import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import Mock, patch

from rag import chunks, index, model, search
from rag.indexing import embedding_text


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    max_seq_length = 128
    tokenizer = types.SimpleNamespace(
        encode=lambda text, **kwargs: [0, *re.findall(r"\w+|[^\w\s]", text), 1]
    )

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
                torch.cuda = types.SimpleNamespace(
                    is_available=Mock(return_value=available)
                )
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
                    "intfloat/multilingual-e5-large",
                    device=device,
                )
        model.cache_clear()

    def test_chunks_preserve_sections_blocks_and_token_limit(self):
        text = (
            "# Główny\nWstęp.\n\n## Kod\n```python\n# nie nagłówek\nprint('kot')\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            + "kot " * 80
        )
        parts = list(chunks(text, FakeModel.tokenizer, 30))
        self.assertTrue(
            all(len(FakeModel.tokenizer.encode(p["text"])) <= 30 for p in parts)
        )
        self.assertEqual(parts[0]["section"], "Główny")
        self.assertTrue(all(p["section"] == "Główny > Kod" for p in parts[1:]))
        self.assertTrue(
            any(
                "```python\n# nie nagłówek\nprint('kot')\n```" in p["text"]
                for p in parts
            )
        )
        self.assertTrue(
            any("| A | B |\n|---|---|\n| 1 | 2 |" in p["text"] for p in parts)
        )
        self.assertEqual("".join(p["text"] for p in parts).count("kot"), 81)

    @patch("rag._shared.model", return_value=FakeModel())
    def test_search_returns_section_metadata(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            (root / "knowledge" / "note.md").write_text(
                "# Zwierzęta\n## Kot\nKot śpi.", encoding="utf-8"
            )
            index(root)
            self.assertIn("Sekcja: Zwierzęta > Kot", search(root, "kot"))

    def test_setext_headings(self):
        parts = list(
            chunks(
                "Tytuł\n======\nWstęp\n\nSekcja\n------\nTreść", FakeModel.tokenizer, 30
            )
        )
        self.assertEqual(
            [part["section"] for part in parts], ["Tytuł", "Tytuł > Sekcja"]
        )

    def test_embedding_text_respects_section_and_limit(self):
        tokenizer = FakeModel.tokenizer
        text = "Bez nagłówka.\n\n# Zwierzęta\n## Kot\n" + "kot " * 80
        parts = list(chunks(text, tokenizer, 24))
        self.assertEqual(
            embedding_text(parts[0]["text"], "", tokenizer, 24),
            "passage: " + parts[0]["text"],
        )
        self.assertGreater(len(parts), 3)
        for part in parts[2:]:
            encoded = embedding_text(part["text"], part["section"], tokenizer, 24)
            self.assertTrue(encoded.startswith("passage: Zwierzęta > Kot\n\n"))
            self.assertTrue(encoded.endswith(part["text"]))
            self.assertLessEqual(len(tokenizer.encode(encoded)), 24)

    def test_long_candidates_are_measured_without_tokenizer_warnings(self):
        calls = []

        def encode(text, **kwargs):
            calls.append((text, kwargs))
            return [0, *re.findall(r"\w+|[^\w\s]", text), 1]

        tokenizer = types.SimpleNamespace(encode=encode)
        parts = list(
            chunks("# " + "nagłówek " * 50 + "\n" + "tekst " * 100, tokenizer, 24)
        )
        self.assertTrue(any(len(encode(text)) > 24 for text, _ in list(calls)))
        for _, options in calls:
            if options:
                self.assertFalse(options["verbose"])
                self.assertFalse(options["truncation"])
        for part in parts:
            value = embedding_text(part["text"], part["section"], tokenizer, 24)
            self.assertLessEqual(len(encode(value)), 24)

        long_text = "# " + "nagłówek " * 100 + "\n" + "treść " * 50
        parts = list(chunks(long_text, tokenizer, 24))
        for part in parts:
            encoded = embedding_text(part["text"], part["section"], tokenizer, 24)
            self.assertTrue(encoded.endswith(part["text"]))
            self.assertLess(len(encoded), len(part["section"]) + len(part["text"]))
            self.assertLessEqual(len(tokenizer.encode(encoded)), 24)

    def test_index_embeds_section_and_rebuilds_old_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            (root / "knowledge" / "note.md").write_text(
                "# Zwierzęta\n## Kot\nKot śpi.", encoding="utf-8"
            )
            fake = FakeModel()
            fake.encode = Mock(wraps=fake.encode)
            with patch("rag._shared.model", return_value=fake):
                index(root)
                encoded = fake.encode.call_args.args[0]
                self.assertEqual(
                    encoded,
                    [
                        "passage: Zwierzęta\n\n# Zwierzęta",
                        "passage: Zwierzęta > Kot\n\n## Kot\nKot śpi.",
                    ],
                )
                result = search(root, "kot")
                fake.encode.assert_called_with("query: kot", normalize_embeddings=True)
                self.assertIn("Sekcja: Zwierzęta > Kot", result)
                self.assertIn("## Kot\nKot śpi.", result)
                manifest_path = root / ".rag" / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["version"] = "old"
                manifest_path.write_text(json.dumps(manifest))
                fake.encode.reset_mock()
                self.assertIn("przebudowy", search(root, "kot"))
                fake.encode.assert_not_called()
                self.assertEqual(index(root), 2)
                fake.encode.assert_called_once()

    @patch("rag._shared.model", return_value=FakeModel())
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
            results = search(root, "kot").split("\n\n---\n\n")
            self.assertTrue(
                results[0].startswith(
                    "Źródło: knowledge/kot.md\nPodobieństwo: 1.0000\n"
                )
            )
            self.assertIn(
                "Źródło: knowledge/pies.md\nPodobieństwo: 0.0000\n", results[1]
            )
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

    @patch("rag._shared.model", return_value=FakeModel())
    def test_empty_markdown_file_can_be_indexed_and_updated(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "knowledge"
            source.mkdir()
            document = source / "empty.md"
            document.write_text("", encoding="utf-8")

            self.assertEqual(index(root), 0)

            document.write_text("Kot lubi mleko.", encoding="utf-8")
            self.assertEqual(index(root), 1)
            self.assertIn("Kot lubi mleko.", search(root, "kot"))

    def test_index_rejects_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "knowledge"
            source.mkdir()
            (source / "invalid.md").write_bytes(b"\xff")

            with self.assertRaises(UnicodeDecodeError):
                index(root)

    def test_index_reports_corrupted_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "knowledge"
            source.mkdir()
            (source / "note.md").write_text("Treść.", encoding="utf-8")
            manifest = root / ".rag" / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text("{niepoprawny json", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                index(root)

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

            with patch("rag._shared.model") as mocked:
                mocked.return_value.tokenizer = fake.tokenizer
                mocked.return_value.max_seq_length = fake.max_seq_length
                mocked.return_value.encode.side_effect = encode
                with self.assertRaises(RuntimeError):
                    index(root)
            self.assertEqual(calls[0], 64)
            with patch("rag._shared.model", return_value=fake):
                count = index(root)
                import lancedb

                table = lancedb.connect(str(root / ".rag" / "lancedb")).open_table(
                    "knowledge"
                )
                self.assertEqual(table.count_rows(), count)
                self.assertGreater(count, 64)


if __name__ == "__main__":
    unittest.main()
