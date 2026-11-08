"""Compare FTS tokenizers on a small, reproducible Polish sample."""

import os

import lancedb
import pytest
from lancedb.index import FTS

CASES = [
    ("zwroty", "Zasady zwrotu opisano w regulaminie."),
    ("reklamacje", "Terminy reklamacji określa umowa."),
    ("wnioski", "Wniosek złożono w biurze."),
    ("pracownicy", "Pracownik podpisuje dokument."),
    ("podróże", "Koszt podróży rozlicza księgowość."),
    ("umowy", "Treść umowy ustala kierownik."),
    ("faktury", "Numer faktury wpisz w formularzu."),
    ("dostawy", "Termin dostawy wskazano w zleceniu."),
    ("rachunki", "Oryginał rachunku dołącz do wniosku."),
    ("urlopy", "Plan urlopu zatwierdza przełożony."),
    ("hasła", "Reset hasła wymaga potwierdzenia."),
    ("płatności", "Termin płatności podano na fakturze."),
]
DISTRACTORS = [
    "Zwrotka piosenki została nagrana w studiu.",
    "Reklama produktu pojawiła się w gazecie.",
    "Wnętrze biura wymaga remontu.",
    "Pracownia ceramiczna oferuje warsztaty.",
    "Podróżnik opisał swoją wyprawę.",
    "Umowny termin podano w kalendarzu.",
    "Fakty potwierdzono podczas kontroli.",
    "Dostawca podpisał protokół.",
    "Rachunkowość prowadzi dział finansów.",
    "Urlopowicz odpoczywa nad morzem.",
    "Hasło reklamowe pojawiło się na plakacie.",
    "Płatnik zgłasza dane do rejestru.",
]


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="opt-in benchmark"
)
def test_polish_tokenizers(tmp_path, capsys):
    db = lancedb.connect(str(tmp_path / "db"))
    rows = [{"id": i, "text": text} for i, (_, text) in enumerate(CASES)]
    rows += [{"id": i + len(CASES), "text": text} for i, text in enumerate(DISTRACTORS)]
    codes = ["AB", "VPN", "RX-104", "742"]
    rows += [
        {"id": len(rows) + i, "text": f"Identyfikator {code} oznacza zgłoszenie."}
        for i, code in enumerate(codes)
    ]
    metrics = {}
    for tokenizer in ("simple", "ngram"):
        table = db.create_table(tokenizer, data=rows)
        table.create_index(
            "text",
            config=FTS(
                base_tokenizer=tokenizer,
                stem=False,
                remove_stop_words=False,
                ngram_min_length=3,
                ngram_max_length=5,
            ),
        )
        ranks = []
        for expected, (query, _) in enumerate(CASES):
            result = table.search(query, query_type="fts").limit(5).to_list()
            ids = [row["id"] for row in result]
            ranks.append(ids.index(expected) + 1 if expected in ids else 0)
        code_ranks = []
        for i, code in enumerate(codes):
            result = table.search(code, query_type="fts").limit(5).to_list()
            ids = [row["id"] for row in result]
            expected = len(CASES) + len(DISTRACTORS) + i
            code_ranks.append(ids.index(expected) + 1 if expected in ids else 0)
        metrics[tokenizer] = {
            "code_ranks": code_ranks,
            "recall@3": sum(0 < r <= 3 for r in ranks) / len(ranks),
            "mrr@3": sum(1 / r for r in ranks if 0 < r <= 3) / len(ranks),
            "ranks": ranks,
        }
    with capsys.disabled():
        print("\nPolish tokenizers:", metrics)
    assert metrics["ngram"]["recall@3"] > metrics["simple"]["recall@3"]
    assert metrics["simple"]["code_ranks"][0] == 1
    assert metrics["ngram"]["code_ranks"][0] == 0


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="requires real model"
)
def test_combined_tokenizers_with_hybrid(tmp_path, capsys):
    from rag import index
    from rag.search import search_results

    codes = ["AB", "VPN", "RX-104", "742"]
    documents = [text for _, text in CASES] + DISTRACTORS
    documents += [f"Identyfikator {code} oznacza zgłoszenie." for code in codes]
    queries = [(query, i) for i, (query, _) in enumerate(CASES)]
    queries += [
        (code, len(CASES) + len(DISTRACTORS) + i) for i, code in enumerate(codes)
    ]
    source = tmp_path / "knowledge"
    source.mkdir()
    for i, document in enumerate(documents):
        (source / f"{i:02}.md").write_text(document, encoding="utf-8")
    index(tmp_path)
    db = lancedb.connect(str(tmp_path / "experiment"))
    table = db.create_table(
        "ngram",
        data=[
            {"path": f"knowledge/{i:02}.md", "text": document}
            for i, document in enumerate(documents)
        ],
    )
    table.create_index(
        "text",
        config=FTS(
            base_tokenizer="ngram",
            ngram_min_length=3,
            ngram_max_length=5,
            stem=False,
            remove_stop_words=False,
        ),
    )
    ranks = {key: [] for key in ("keyword", "combined", "hybrid", "combined_hybrid")}
    for query, expected in queries:
        simple = search_results(tmp_path, query, limit=30, mode="keyword")
        semantic = search_results(tmp_path, query, limit=30, mode="semantic")
        hybrid = search_results(tmp_path, query, limit=30, mode="hybrid")
        assert (
            isinstance(simple, list)
            and isinstance(semantic, list)
            and isinstance(hybrid, list)
        )
        ngram = table.search(query, query_type="fts").limit(30).to_list()
        simple_paths = [row["source"] for row in simple]
        semantic_paths = [row["source"] for row in semantic]
        ngram_paths = [row["path"] for row in ngram]
        # Equal RRF of both lexical rankings; then use the production hybrid weights.
        scores = {}
        for ranking in (simple_paths, ngram_paths):
            for rank, path in enumerate(ranking, 1):
                scores[path] = scores.get(path, 0.0) + 0.5 / (1 + rank)
        combined = sorted(scores, key=lambda path: -scores[path])
        hybrid_scores = {
            path: 0.8 / (1 + rank) for rank, path in enumerate(semantic_paths, 1)
        }
        for rank, path in enumerate(combined, 1):
            hybrid_scores[path] = hybrid_scores.get(path, 0.0) + 0.2 / (1 + rank)
        combined_hybrid = sorted(hybrid_scores, key=lambda path: -hybrid_scores[path])
        for key, paths in (
            ("keyword", simple_paths),
            ("combined", combined),
            ("hybrid", [row["source"] for row in hybrid]),
            ("combined_hybrid", combined_hybrid),
        ):
            target = f"knowledge/{expected:02}.md"
            ranks[key].append(paths.index(target) + 1 if target in paths else 0)
    metrics = {}
    for mode, values in ranks.items():
        metrics[mode] = {
            "recall@3": sum(0 < r <= 3 for r in values[:12]) / 12,
            "mrr@3": sum(1 / r for r in values[:12] if 0 < r <= 3) / 12,
            "code_ranks": values[12:],
        }
    with capsys.disabled():
        print("\nCombined tokenizer experiment:", metrics)
    assert all(0 < rank <= 3 for rank in metrics["combined"]["code_ranks"])
    assert metrics["combined"]["recall@3"] > metrics["keyword"]["recall@3"]
