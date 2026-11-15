from pathlib import Path

import pytest

from rag.evaluate import evaluate


def test_evaluation_aggregates_categories_and_ranks(monkeypatch):
    def search(root, query, **kwargs):
        return [
            {"source": "knowledge/other.md", "fragments": [{"content": "other"}]},
            {"source": "knowledge/test.md", "fragments": [{"content": "needle"}]},
            {"source": "knowledge/third.md", "fragments": [{"content": "third"}]},
            {"source": "knowledge/fourth.md", "fragments": [{"content": "fourth"}]},
        ]

    monkeypatch.setattr("rag.evaluate.search_results", search)
    cases = [
        {
            "query": "private",
            "source": "knowledge/test.md",
            "category": "lexical",
            "content_contains": "needle",
        },
        {
            "query": "secret",
            "sources": [
                "knowledge/test.md",
                "knowledge/fourth.md",
                "knowledge/missing.md",
                "knowledge/absent.md",
            ],
            "category": "many",
        },
    ]
    report = evaluate(Path("."), cases)
    assert report["lexical"]["hybrid"]["mrr@3"] == 0.5
    assert report["lexical"]["semantic"]["recall@5"] == 1
    # One of four expected documents is in the top three, two in the top five.
    assert report["many"]["hybrid"]["recall@3"] == pytest.approx(1 / 3)
    assert report["many"]["hybrid"]["recall@5"] == 0.5
    assert report["many"]["hybrid"]["recall@10"] == 0.5
    assert report["many"]["hybrid"]["mrr@5"] == 0.5
    assert report["all"]["keyword"]["count"] == 2
    assert "private" not in str(report)
    assert "secret" not in str(report)


@pytest.mark.parametrize(
    "case",
    [
        {"query": "x"},
        {"query": "x", "category": "c", "source": "a.md", "sources": ["a.md"]},
        {"query": "x", "category": "c", "sources": []},
        {
            "query": "x",
            "category": "c",
            "sources": ["a.md", "b.md"],
            "content_contains": "y",
        },
    ],
)
def test_evaluation_rejects_invalid_cases(case):
    with pytest.raises(ValueError):
        evaluate(Path("."), [case])
