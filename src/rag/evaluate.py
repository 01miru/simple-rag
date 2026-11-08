"""Evaluate retrieval against a locally supplied JSON set without printing queries."""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from rag.search import search_results


def evaluate(root: Path, cases: list[dict]) -> dict:
    """Return aggregate metrics by category and mode."""
    if not cases:
        raise ValueError("Zestaw pytań jest pusty.")
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for case in cases:
        if not all(
            isinstance(case.get(key), str) and case[key].strip()
            for key in ("query", "source", "category")
        ):
            raise ValueError(
                "Każdy przypadek wymaga niepustych pól query, source i category."
            )
        if "content_contains" in case and (
            not isinstance(case["content_contains"], str)
            or not case["content_contains"]
        ):
            raise ValueError("content_contains musi być niepustym tekstem.")
    # Warm up the model and database before measuring; omit warm-up from metrics.
    warmup = search_results(root, cases[0]["query"], limit=5, mode="semantic")
    if isinstance(warmup, str):
        raise RuntimeError(warmup)  # noqa: TRY004
    for case in cases:
        for mode in ("semantic", "hybrid", "keyword"):
            start = time.perf_counter()
            results = search_results(root, case["query"], limit=5, mode=mode)
            elapsed = (time.perf_counter() - start) * 1000
            if isinstance(results, str):
                raise RuntimeError(results)  # noqa: TRY004
            rank = next(
                (
                    i
                    for i, row in enumerate(results, 1)
                    if row["source"] == case["source"]
                    and (
                        "content_contains" not in case
                        or case["content_contains"] in row["content"]
                    )
                ),
                0,
            )
            grouped[case["category"]][mode].append((rank, elapsed))
            grouped["all"][mode].append((rank, elapsed))
    report: dict = {}
    for category, modes in grouped.items():
        report[category] = {}
        for mode, observations in modes.items():
            n = len(observations)
            metrics = {
                f"{metric}@{k}": sum(
                    (1 if metric == "recall" else 1 / rank)
                    for rank, _ in observations
                    if 0 < rank <= k
                )
                / n
                for k in (3, 5)
                for metric in ("recall", "mrr")
            }
            metrics.update(
                count=n,
                median_ms=statistics.median(elapsed for _, elapsed in observations),
            )
            report[category][mode] = metrics
    return report


def main() -> None:
    """Read the evaluation set and print aggregate metrics as JSON."""
    parser = argparse.ArgumentParser(
        description="Ocena lokalnego indeksu bez wypisywania pytań i dokumentów"
    )
    parser.add_argument(
        "directory", type=Path, help="katalog z knowledge/ i aktualnym indeksem .rag/"
    )
    parser.add_argument("cases", type=Path, help="plik JSON z listą przypadków")
    args = parser.parse_args()
    try:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        if not isinstance(cases, list) or not all(
            isinstance(case, dict) for case in cases
        ):
            raise ValueError("Plik musi zawierać listę obiektów.")
        if any(case.get("category") == "all" for case in cases):
            raise ValueError("Kategoria 'all' jest zarezerwowana.")
        report = evaluate(args.directory.resolve(), cases)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Błąd: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
