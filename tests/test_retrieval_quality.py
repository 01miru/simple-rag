"""Small semantic retrieval benchmark, run explicitly with the real model."""

import os
from pathlib import Path

import pytest

from rag import index, search
from rag.search import search_results

DOCUMENTS = {
    "urlopy.md": "Pracownicy składają wnioski o urlop wypoczynkowy w systemie kadrowym. Wniosek zatwierdza przełożony.",
    "hasla.md": "Zapomniane hasło do konta można zresetować przez link na stronie logowania. Link jest wysyłany na służbowy adres e-mail.",
    "zwroty.md": "Klient może zwrócić zakupiony produkt w ciągu 30 dni od dostawy. Zwrot pieniędzy następuje po otrzymaniu przesyłki.",
    "kopie.md": "Kopie zapasowe bazy danych wykonywane są codziennie w nocy i przechowywane przez 14 dni.",
    "podroze.md": "Koszty podróży służbowej, w tym bilety kolejowe i nocleg, rozlicza się po powrocie na podstawie rachunków.",
    "vpn.md": "Aby połączyć się z siecią firmową z domu, uruchom klienta VPN i zaloguj się kontem służbowym.",
}


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


@pytest.fixture(scope="module")
def sample_documents_index(tmp_path_factory):
    root = tmp_path_factory.mktemp("sample_documents")
    source = root / "knowledge"
    source.mkdir()
    for name, text in DOCUMENTS.items():
        (source / name).write_text(text, encoding="utf-8")
    index(root)
    return root


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1",
    reason="Set RAG_QUALITY_TEST=1 to run the real-model benchmark",
)
@pytest.mark.parametrize(
    ("queries", "limit", "min_recall", "min_mrr"),
    [
        (
            (
                ("Jak poprosić o wolne od pracy?", "urlopy.md"),
                ("Nie pamiętam danych logowania. Jak odzyskać dostęp?", "hasla.md"),
                ("Ile mam czasu na oddanie zamówienia?", "zwroty.md"),
                ("Jak długo przechowywany jest backup bazy?", "kopie.md"),
                ("Jak otrzymać zwrot za hotel podczas delegacji?", "podroze.md"),
                ("How can I access the company network remotely?", "vpn.md"),
            ),
            3,
            5 / 6,
            0.65,
        ),
        (
            (
                ("Gdzie złożyć wniosek o urlop?", "urlopy.md"),
                ("Kto zatwierdza urlop pracownika?", "urlopy.md"),
                ("Jak zgłosić urlop wypoczynkowy?", "urlopy.md"),
                ("W jakim systemie prosi się o wolne?", "urlopy.md"),
                ("Who approves an employee's vacation request?", "urlopy.md"),
                ("Jak zresetować zapomniane hasło?", "hasla.md"),
                ("Gdzie znajdę link do resetowania hasła?", "hasla.md"),
                ("Na jaki adres przychodzi link do zmiany hasła?", "hasla.md"),
                ("Nie pamiętam hasła do konta służbowego", "hasla.md"),
                ("How do I reset a forgotten password?", "hasla.md"),
                ("Jaki jest termin zwrotu produktu?", "zwroty.md"),
                ("Kiedy klient otrzyma zwrot pieniędzy?", "zwroty.md"),
                ("Czy można oddać produkt po dostawie?", "zwroty.md"),
                ("Ile dni przysługuje na zwrot zakupu?", "zwroty.md"),
                ("How long do customers have to return a product?", "zwroty.md"),
                ("Jak często wykonywane są kopie zapasowe bazy?", "kopie.md"),
                ("Kiedy powstaje kopia zapasowa bazy danych?", "kopie.md"),
                ("Po ilu dniach usuwa się backup bazy?", "kopie.md"),
                ("Jaki jest okres przechowywania kopii zapasowych?", "kopie.md"),
                ("How often is the database backed up?", "kopie.md"),
                ("Jak rozliczyć bilety kolejowe z delegacji?", "podroze.md"),
                ("Jakie dokumenty są potrzebne do rozliczenia podróży?", "podroze.md"),
                ("Kiedy rozlicza się koszty delegacji?", "podroze.md"),
                ("Czy nocleg można rozliczyć jako podróż służbową?", "podroze.md"),
                ("How do I claim travel expenses after a business trip?", "podroze.md"),
                ("Jak połączyć się z firmową siecią z domu?", "vpn.md"),
                ("Jakiego programu użyć do zdalnego dostępu do sieci?", "vpn.md"),
                ("Czym zalogować się do firmowego VPN?", "vpn.md"),
                ("Jak uzyskać dostęp do sieci firmowej poza biurem?", "vpn.md"),
                ("Which credentials should I use for the company VPN?", "vpn.md"),
            ),
            5,
            0.8,
            0.65,
        ),
    ],
    ids=["original", "paraphrases"],
)
def test_retrieval_quality(sample_documents_index, queries, limit, min_recall, min_mrr):
    ranks = []
    for query, expected in queries:
        results = search(sample_documents_index, query, limit=limit).split(
            "\n\n---\n\n"
        )
        paths = [result.splitlines()[0] for result in results]
        target = f"Źródło: knowledge/{expected}"
        ranks.append(paths.index(target) + 1 if target in paths else 0)

    recall = sum(rank > 0 for rank in ranks) / len(ranks)
    mrr = sum(1 / rank for rank in ranks if rank) / len(ranks)
    details = ", ".join(
        f"{query}: {rank or 'brak'}"
        for (query, _), rank in zip(queries, ranks, strict=True)
    )
    assert recall >= min_recall, f"Recall@{limit}={recall:.2f}; {details}"
    assert mrr >= min_mrr, f"MRR@{limit}={mrr:.2f}; {details}"


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="Set RAG_QUALITY_TEST=1"
)
def test_original_hybrid_comparison(sample_documents_index):
    queries = [
        ("Jak poprosić o wolne od pracy?", "urlopy.md"),
        ("Nie pamiętam danych logowania. Jak odzyskać dostęp?", "hasla.md"),
        ("Ile mam czasu na oddanie zamówienia?", "zwroty.md"),
        ("Jak długo przechowywany jest backup bazy?", "kopie.md"),
        ("Jak otrzymać zwrot za hotel podczas delegacji?", "podroze.md"),
        ("How can I access the company network remotely?", "vpn.md"),
    ]
    metrics = {}
    for mode in ("semantic", "hybrid"):
        ranks = []
        for query, expected in queries:
            results = search_results(sample_documents_index, query, limit=5, mode=mode)
            assert isinstance(results, list)
            paths = [result["source"] for result in results]
            target = f"knowledge/{expected}"
            ranks.append(paths.index(target) + 1 if target in paths else 0)
        metrics[mode] = {
            f"{metric}@{k}": sum(
                (1 if metric == "recall" else 1 / rank)
                for rank in ranks
                if 0 < rank <= k
            )
            / len(ranks)
            for k in (3, 5)
            for metric in ("recall", "mrr")
        }
    print("\nOriginal benchmark:", metrics)
    for metric in ("recall@3", "mrr@3"):
        assert metrics["hybrid"][metric] >= metrics["semantic"][metric]


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="Set RAG_QUALITY_TEST=1"
)
def test_independent_long_document_comparison(long_document_index):
    # These questions were not used to tune the hybrid ranking. The existing
    # three long-document queries are deliberately excluded.
    cases = [
        ("Co zrobić, gdy zginie identyfikator gościa?", "Utrata karty"),
        ("Kto towarzyszy odwiedzającym w laboratoriach?", "Osoba zapraszająca"),
        ("Jak oznaczać materiał biologiczny?", "inicjałami operatora"),
        ("Co zrobić z uszkodzonym przewodem przyrządu?", "pozostaje wyłączone"),
        ("Kiedy zdjąć rękawice ochronne?", "przed dotykaniem klamek"),
        ("Jak często zapisuje się temperaturę?", "dwa razy dziennie"),
        ("W jaki dzień przychodzą pojemniki na próbki?", "środa"),
        ("Jak zabezpieczyć pojemniki przed wysyłką?", "materiałem amortyzującym"),
        ("Kto przyznaje dostęp do katalogu zespołu?", "administrator"),
        ("Kto sprawdza raport przed jego publikacją?", "druga osoba"),
        ("Co przekazać następnej zmianie?", "Otwarte zadania"),
        ("Co sprawdzić po naprawieniu chłodni?", "zapis temperatury"),
    ]
    metrics = {}
    for mode in ("semantic", "hybrid"):
        ranks = []
        for query, expected in cases:
            results = search_results(long_document_index, query, limit=5, mode=mode)
            assert isinstance(results, list)
            rank = next(
                (i for i, row in enumerate(results, 1) if expected in row["content"]), 0
            )
            ranks.append(rank)
        metrics[mode] = {
            f"{metric}@{k}": sum(
                (1 if metric == "recall" else 1 / rank)
                for rank in ranks
                if 0 < rank <= k
            )
            / len(ranks)
            for k in (3, 5)
            for metric in ("recall", "mrr")
        }
        metrics[mode]["ranks"] = ranks
    print("\nIndependent long document:", metrics)
    assert metrics["hybrid"]["recall@3"] >= metrics["semantic"]["recall@3"]
    assert metrics["hybrid"]["mrr@3"] >= metrics["semantic"]["mrr@3"] - 0.03
