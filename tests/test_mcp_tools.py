import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from rag import create_server, get_document, list_documents


class DocumentToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.knowledge = self.root / "knowledge"
        self.knowledge.mkdir()
        self.document = self.knowledge / "folder" / "ą note.md"
        self.document.parent.mkdir()
        self.document.write_text("pierwsza\nśrodkowa\nostatnia", encoding="utf-8")
        self.manifest()

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        data = self.document.read_bytes()
        (self.root / ".rag").mkdir(exist_ok=True)
        (self.root / ".rag" / "manifest.json").write_text(
            json.dumps(
                {
                    "files": {
                        "knowledge/folder/ą note.md": hashlib.sha256(data).hexdigest(),
                        "knowledge/incomplete.md": None,
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_list_documents_filters_sorts_and_paginates_manifest(self):
        result = list_documents(self.root, "knowledge/folder/", limit=1)
        self.assertEqual(
            result,
            {
                "documents": [{"path": "knowledge/folder/ą note.md"}],
                "total": 1,
                "limit": 1,
                "offset": 0,
                "has_more": False,
            },
        )
        self.assertEqual(list_documents(self.root)["total"], 1)
        with self.assertRaises(ValueError):
            list_documents(self.root, limit=0)

    def test_get_document_reads_line_range_without_loading_model(self):
        with patch("rag._shared.model", side_effect=AssertionError("model loaded")):
            result = get_document(self.root, "knowledge/folder/ą note.md", 2, 1)
        self.assertEqual(result["content"], "środkowa")
        self.assertEqual(
            (result["start_line"], result["end_line"], result["total_lines"]), (2, 2, 3)
        )

    def test_get_document_rejects_traversal_symlink_and_stale_content(self):
        outside = self.root / "outside.md"
        outside.write_text("private", encoding="utf-8")
        (self.knowledge / "link.md").symlink_to(outside)
        with self.assertRaises(ValueError):
            get_document(self.root, "knowledge/link.md")
        self.document.write_text("zmienione", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ponowne indeksowanie"):
            get_document(self.root, "knowledge/folder/ą note.md")

    def test_server_registers_three_tools(self):
        server = create_server(self.root)
        names = {tool.name for tool in server._tool_manager.list_tools()}
        self.assertEqual(names, {"search_knowledge", "list_documents", "get_document"})


@pytest.mark.parametrize(
    "path", ["../AGENTS.md", ".rag/manifest.json", "/tmp/outside.md"]
)
def test_get_document_rejects_disallowed_paths(tmp_path, path):
    (tmp_path / "knowledge").mkdir()
    with pytest.raises(ValueError):
        get_document(tmp_path, path)
