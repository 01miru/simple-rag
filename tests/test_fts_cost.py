"""Opt-in synthetic cost comparison for one and two lexical indexes."""

import os
import statistics
import time

import lancedb
import pytest
from lancedb.index import FTS


@pytest.mark.skipif(
    os.environ.get("RAG_COST_TEST") != "1", reason="opt-in cost benchmark"
)
def test_fts_cost(tmp_path, capsys):
    count = 2000
    rows = []
    for i in range(count):
        text = (
            f"Zgłoszenie serwisowe RX-{i:05} dotyczy przeglądu wyposażenia. "
            f"Oddział {i % 40} rozlicza koszty podróży i przechowuje dokumentację. "
            "Pracownik sprawdza procedury, potwierdza termin i składa wniosek o zwrot kosztów. "
            "Kierownik zatwierdza dokumenty, a dział administracji zapisuje wynik kontroli."
        )
        rows.append(
            {
                "id": i,
                "text": text,
                "ngram_text": text,
                "vector": [float((i + j) % 31) / 31 for j in range(16)],
            }
        )

    measurements = {}
    for dual in (False, True):
        name = "dual" if dual else "simple"
        path = tmp_path / name
        db = lancedb.connect(str(path))
        data = (
            rows
            if dual
            else [
                {key: value for key, value in row.items() if key != "ngram_text"}
                for row in rows
            ]
        )
        table = db.create_table("docs", data=data)
        start = time.perf_counter()
        table.create_index("text", config=FTS(stem=False, remove_stop_words=False))
        if dual:
            table.create_index(
                "ngram_text",
                config=FTS(
                    base_tokenizer="ngram",
                    ngram_min_length=3,
                    ngram_max_length=5,
                    stem=False,
                    remove_stop_words=False,
                ),
            )
        index_seconds = time.perf_counter() - start
        size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

        def run(query, table=table, dual=dual):
            vector = table.search([0.5] * 16).limit(30).to_list()
            exact = (
                table.search(query, query_type="fts", fts_columns="text")
                .limit(30)
                .to_list()
            )
            rankings = [exact]
            if dual:
                rankings.append(
                    table.search(query, query_type="fts", fts_columns="ngram_text")
                    .limit(30)
                    .to_list()
                )
            lexical = {}
            for ranking in rankings:
                for rank, item in enumerate(ranking, 1):
                    lexical[item["id"]] = lexical.get(item["id"], 0.0) + 1 / len(
                        rankings
                    ) / (1 + rank)
            fused = {
                item["id"]: 0.8 / (1 + rank) for rank, item in enumerate(vector, 1)
            }
            for rank, item in enumerate(
                sorted(lexical, key=lambda item: -lexical[item]), 1
            ):
                fused[item] = fused.get(item, 0.0) + 0.2 / (1 + rank)
            return sorted(fused, key=lambda item: -fused[item])[:5]

        queries = ["zwroty kosztów", "RX-00125", "wnioski", "podróże", "AB"]
        for query in queries:
            run(query)
        durations = []
        for _ in range(4):
            for query in queries:
                start = time.perf_counter()
                run(query)
                durations.append((time.perf_counter() - start) * 1000)
        measurements[name] = {
            "index_seconds": index_seconds,
            "total_bytes": size,
            "median_ms": statistics.median(durations),
            "mean_ms": statistics.mean(durations),
        }
    with capsys.disabled():
        print("\nFTS cost (2000 synthetic rows; embedding excluded):", measurements)
    assert all(result["total_bytes"] > 0 for result in measurements.values())


@pytest.mark.skipif(
    os.environ.get("RAG_COST_TEST") != "1", reason="requires real model"
)
def test_full_query_cost(tmp_path, capsys):
    import json

    from rag import _shared
    from rag.search import search_results

    # Vectors are encoded by the real model, then reused across records to keep
    # the fixture affordable; this benchmark measures queries, not indexing.
    model = _shared.model()
    templates = [
        f"Instrukcja działu {i}: zgłoszenie przeglądu, zwrot kosztów i archiwizacja dokumentów po zakończeniu pracy."
        for i in range(40)
    ]
    vectors = model.encode(
        [f"passage: {text}" for text in templates], normalize_embeddings=True
    )
    rows = []
    for i in range(2000):
        text = f"{templates[i % len(templates)]} Numer zgłoszenia RX-{i:05}."
        rows.append(
            {
                "path": f"knowledge/{i:05}.md",
                "section": "",
                "text": text,
                "fts_text": text,
                "chunk_id": str(i),
                "vector": vectors[i % len(vectors)].tolist(),
            }
        )
    rag_dir = tmp_path / ".rag"
    db = lancedb.connect(str(rag_dir / "lancedb"))
    table = db.create_table(_shared.TABLE, data=rows)
    table.create_index("fts_text", config=FTS(stem=False, remove_stop_words=False))
    (rag_dir / "manifest.json").write_text(
        json.dumps({"version": _shared.INDEX_VERSION, "files": {}}), encoding="utf-8"
    )
    queries = [
        "zwrot kosztów",
        "RX-00125",
        "archiwizacja dokumentów",
        "przegląd urządzenia",
        "numer zgłoszenia",
    ]
    assert isinstance(search_results(tmp_path, queries[0], mode="hybrid"), list)
    durations = {"semantic": [], "hybrid": []}
    # Alternate mode order to reduce systematic warm-cache bias.
    for i in range(4):
        for query in queries:
            for mode in (
                ("semantic", "hybrid") if i % 2 == 0 else ("hybrid", "semantic")
            ):
                start = time.perf_counter()
                result = search_results(tmp_path, query, mode=mode)
                durations[mode].append((time.perf_counter() - start) * 1000)
                assert isinstance(result, list) and result
    summary = {
        mode: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
        }
        for mode, values in durations.items()
    }
    summary["paired_delta"] = {
        "median_ms": statistics.median(
            h - s
            for h, s in zip(durations["hybrid"], durations["semantic"], strict=True)
        ),
        "mean_ms": statistics.mean(
            h - s
            for h, s in zip(durations["hybrid"], durations["semantic"], strict=True)
        ),
    }
    with capsys.disabled():
        print("\nFull query cost (2000 synthetic rows, real model):", summary)
