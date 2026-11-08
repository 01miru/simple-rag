"""Small semantic retrieval benchmark, run explicitly with the real model."""

import os
import tempfile
import unittest
from pathlib import Path

import pytest

from rag import index, search

DOCUMENTS = {
    "urlopy.md": "Pracownicy składają wnioski o urlop wypoczynkowy w systemie kadrowym. Wniosek zatwierdza przełożony.",
    "hasla.md": "Zapomniane hasło do konta można zresetować przez link na stronie logowania. Link jest wysyłany na służbowy adres e-mail.",
    "zwroty.md": "Klient może zwrócić zakupiony produkt w ciągu 30 dni od dostawy. Zwrot pieniędzy następuje po otrzymaniu przesyłki.",
    "kopie.md": "Kopie zapasowe bazy danych wykonywane są codziennie w nocy i przechowywane przez 14 dni.",
    "podroze.md": "Koszty podróży służbowej, w tym bilety kolejowe i nocleg, rozlicza się po powrocie na podstawie rachunków.",
    "vpn.md": "Aby połączyć się z siecią firmową z domu, uruchom klienta VPN i zaloguj się kontem służbowym.",
}

# Each query has one relevant document; wording differs from the source text.
QUERIES = (
    ("Jak poprosić o wolne od pracy?", "urlopy.md"),
    ("Nie pamiętam danych logowania. Jak odzyskać dostęp?", "hasla.md"),
    ("Ile mam czasu na oddanie zamówienia?", "zwroty.md"),
    ("Jak długo przechowywany jest backup bazy?", "kopie.md"),
    ("Jak otrzymać zwrot za hotel podczas delegacji?", "podroze.md"),
    ("How can I access the company network remotely?", "vpn.md"),
)


@pytest.fixture(scope="module")
def long_document_index(tmp_path_factory):
    root = tmp_path_factory.mktemp("long_document")
    source = root / "knowledge"
    source.mkdir()
    document = Path(__file__).parent / "fixtures" / "podrecznik.md"
    (source / document.name).write_text(
        document.read_text(encoding="utf-8"), encoding="utf-8"
    )
    index(root)
    return root


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1",
    reason="Set RAG_QUALITY_TEST=1 to run the real-model benchmark",
)
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Po jakim czasie wygasa karta gościa?", "48 godzinach"),
        ("W jakiej temperaturze przechowywać odczynniki wzorcowe?", "2 do 8 stopni"),
        ("Pod jaki numer zadzwonić w razie awarii chłodni?", "numer 742"),
    ],
)
def test_long_document_fragments(long_document_index, query, expected):
    results = search(long_document_index, query, limit=3).split("\n\n---\n\n")
    assert any(expected in result for result in results), query


@unittest.skipUnless(
    os.environ.get("RAG_QUALITY_TEST") == "1",
    "Set RAG_QUALITY_TEST=1 to run the real-model benchmark",
)
class RetrievalQualityTest(unittest.TestCase):
    def test_recall_and_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "knowledge"
            source.mkdir()
            for name, text in DOCUMENTS.items():
                (source / name).write_text(text, encoding="utf-8")
            index(root)

            ranks = []
            for query, expected in QUERIES:
                results = search(root, query, limit=3).split("\n\n---\n\n")
                paths = [result.splitlines()[0] for result in results]
                target = f"Źródło: knowledge/{expected}"
                ranks.append(paths.index(target) + 1 if target in paths else 0)

        recall = sum(rank > 0 for rank in ranks) / len(ranks)
        mrr = sum(1 / rank for rank in ranks if rank) / len(ranks)
        details = ", ".join(
            f"{query}: {rank or 'brak'}"
            for (query, _), rank in zip(QUERIES, ranks, strict=True)
        )
        self.assertGreaterEqual(recall, 5 / 6, f"Recall@3={recall:.2f}; {details}")
        self.assertGreaterEqual(mrr, 0.65, f"MRR@3={mrr:.2f}; {details}")
