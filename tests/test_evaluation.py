from pathlib import Path

import pytest

from rag.evaluate import evaluate


def test_evaluation_aggregates_categories_and_ranks(monkeypatch):
    def search(root, query, limit, mode):
        return [
            {"source": "knowledge/other.md", "content": "other"},
            {"source": "knowledge/test.md", "content": "needle"},
        ]

    monkeypatch.setattr("rag.evaluate.search_results", search)
    cases = [
        {
            "query": "private",
            "source": "knowledge/test.md",
            "category": "lexical",
            "content_contains": "needle",
        }
    ]
    report = evaluate(Path("."), cases)
    assert report["lexical"]["hybrid"]["mrr@3"] == 0.5
    assert report["all"]["semantic"]["recall@5"] == 1
    assert "private" not in str(report)


def test_evaluation_rejects_invalid_cases():
    with pytest.raises(ValueError):
        evaluate(Path("."), [{"query": "x"}])
