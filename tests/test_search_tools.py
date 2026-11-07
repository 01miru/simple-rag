import asyncio
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp.server.fastmcp.exceptions import ToolError

from rag import create_server, index


class FakeModel:
    tokenizer = types.SimpleNamespace(
        encode=lambda text, **kwargs: re.findall(r"\w+", text)
    )
    max_seq_length = 128

    def encode(self, texts, normalize_embeddings=True):
        def vector(text):
            return types.SimpleNamespace(
                tolist=lambda: [1.0, 0.0] if "target" in text else [0.0, 1.0]
            )

        return (
            vector(texts)
            if isinstance(texts, str)
            else [vector(text) for text in texts]
        )


class SearchToolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        knowledge = self.root / "knowledge"
        for path, text in {
            "other.md": "target result outside",
            "project/a.md": "target result inside",
            "project/second.md": "target second inside",
            "project-neighbor/near.md": "target wrong segment",
            "percent%_name.md": "target special path",
        }.items():
            file = knowledge / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
        self.model_patch = patch("rag.model", return_value=FakeModel())
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        index(self.root)
        self.tool = create_server(self.root)
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()
        self.temp.cleanup()

    def call(self, query="target", **arguments):
        async def invoke():
            return await self.tool._tool_manager.call_tool(
                "search_knowledge", {"query": query, **arguments}
            )

        return self.loop.run_until_complete(invoke())

    @staticmethod
    def text(result):
        return result if isinstance(result, str) else result[0].text

    def test_limit_default_and_custom_and_validation(self):
        default = self.text(self.call())
        self.assertEqual(default.count("Źródło:"), 5)
        self.assertEqual(self.text(self.call(limit=1)).count("Źródło:"), 1)
        for limit in (0, -1, 51):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ToolError, "limit musi być z zakresu 1..50"),
            ):
                self.call(limit=limit)

    def test_prefix_filters_before_top_k_and_respects_segment_boundary(self):
        result = self.text(self.call(limit=1, path_prefix="knowledge/project/"))
        self.assertIn("knowledge/project/", result)
        self.assertNotIn("project-neighbor", result)
        self.assertNotIn("knowledge/other.md", result)
        self.assertIn(
            "knowledge/project/a.md",
            self.text(self.call(path_prefix="knowledge/project/a.md")),
        )
        self.assertIn(
            "Brak wyników", self.text(self.call(path_prefix="knowledge/missing"))
        )

    def test_empty_query_and_missing_index_do_not_load_model(self):
        with patch("rag.model", side_effect=AssertionError("model loaded")):
            self.assertIn("niepuste", self.text(self.call(query="")))
            missing = create_server(self.root / "missing")

            async def invoke_missing():
                return await missing._tool_manager.call_tool(
                    "search_knowledge", {"query": "target"}
                )

            self.assertIn(
                "indeks", self.text(self.loop.run_until_complete(invoke_missing()))
            )
