"""Opt-in comparison of lexical and semantic retrieval with the real model."""

import os
import time

import pytest

from rag import index
from rag.search import search_results

CASES = [
    ("ZXQ-739", "Procedura ZXQ-739 opisuje zasady konserwacji urządzenia."),
    ("KLM-482", "Formularz KLM-482 służy do zgłaszania uszkodzeń."),
    ("TRV-156", "Instrukcja TRV-156 reguluje postępowanie przy awarii."),
    ("AB-912", "Kod AB-912 oznacza zamówienie specjalne."),
    ("FZ-771", "Przypadek FZ-771 wymaga zgody kierownika."),
    ("QX-504", "Etykieta QX-504 identyfikuje partię materiału."),
    ("Pelagornis", "Projekt Pelagornis dotyczy archiwizacji danych."),
    ("Wierzbowice", "Oddział Wierzbowice prowadzi ewidencję przesyłek."),
    ("Zefirella", "Aplikacja Zefirella obsługuje rejestr dokumentów."),
    ("Orsinium", "System Orsinium przechowuje kopie dokumentacji."),
    ("Borowina", "Dział Borowina odpowiada za kontrolę dostaw."),
    ("Kwarcyn", "Program Kwarcyn zarządza obiegiem faktur."),
    ("zwroty kosztów", "Zasady zwrotu kosztów określono w regulaminie delegacji."),
    ("reklamacje przesyłek", "Procedura reklamacji przesyłek obowiązuje kurierów."),
    ("wnioski urlopowe", "Terminy wniosku urlopowego określa dział kadr."),
    ("hasła", "Zmiana hasła wymaga potwierdzenia tożsamości."),
    ("paragraf 47", "Paragraf 47 dotyczy okresu przechowywania akt."),
    ("742", "W razie awarii zadzwoń pod numer 742."),
]


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="requires real model"
)
def test_lexical_comparison(tmp_path, capsys):
    source = tmp_path / "knowledge"
    source.mkdir()
    for i, (_, document) in enumerate(CASES):
        (source / f"doc{i:02}.md").write_text(document, encoding="utf-8")
    index(tmp_path)
    metrics = {}
    for mode in ("semantic", "hybrid", "keyword"):
        ranks, durations = [], []
        for i, (query, _) in enumerate(CASES):
            start = time.perf_counter()
            results = search_results(tmp_path, query, limit=5, mode=mode)
            durations.append((time.perf_counter() - start) * 1000)
            assert isinstance(results, list)
            paths = [item["source"] for item in results]
            target = f"knowledge/doc{i:02}.md"
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
        metrics[mode]["mean_ms"] = sum(durations) / len(durations)
    with capsys.disabled():
        print("\nLexical benchmark:", metrics)
    assert metrics["hybrid"]["recall@3"] >= metrics["semantic"]["recall@3"]


# Similar records make the identifier, rather than the surrounding topic, decisive.
CODES = (
    "RX-104",
    "RX-109",
    "RX-114",
    "RX-119",
    "RX-124",
    "RX-129",
    "RX-134",
    "RX-139",
    "RX-144",
    "RX-149",
    "RX-154",
    "RX-159",
    "RX-164",
    "RX-169",
    "RX-174",
    "RX-179",
    "RX-184",
    "RX-189",
    "RX-194",
    "RX-199",
)


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="requires real model"
)
def test_similar_records_benchmark(tmp_path, capsys):
    source = tmp_path / "knowledge"
    source.mkdir()
    for i, code in enumerate(CODES):
        (source / f"record{i:02}.md").write_text(
            "# Rejestr serwisowy\n\n"
            + (
                "Pracownik sprawdza dokumentację, wykonuje przegląd urządzenia i zapisuje wynik kontroli. "
                * 18
            )
            + f"Numer zgłoszenia: {code}. "
            + (
                "Po zakończeniu serwisu kierownik oddziału zatwierdza raport oraz przekazuje go do archiwum. "
                * 12
            ),
            encoding="utf-8",
        )
    index(tmp_path)
    metrics = {}
    ranks_by_mode = {}
    for mode in ("semantic", "hybrid", "keyword"):
        ranks = []
        for i, code in enumerate(CODES):
            results = search_results(tmp_path, f"zgłoszenie {code}", limit=5, mode=mode)
            assert isinstance(results, list)
            paths = [row["source"] for row in results]
            expected = f"knowledge/record{i:02}.md"
            ranks.append(paths.index(expected) + 1 if expected in paths else 0)
        ranks_by_mode[mode] = ranks
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
    with capsys.disabled():
        print("\nSimilar records:", metrics)
        print("Ranks:", ranks_by_mode)
    assert metrics["hybrid"]["recall@3"] >= metrics["semantic"]["recall@3"]
