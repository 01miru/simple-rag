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
    grouped: dict[str, dict[str, list[tuple[list[int], int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    expected: list[set[str]] = []
    for case in cases:
        sources = [case["source"]] if "source" in case else case.get("sources")
        if (
            ("source" in case and "sources" in case)
            or not isinstance(sources, list)
            or not sources
            or not all(
                isinstance(value, str) and value.strip()
                for value in (case.get("query"), case.get("category"), *sources)
            )
        ):
            raise ValueError(
                "Każdy przypadek wymaga niepustych pól query i category oraz "
                "jednego z pól source lub sources."
            )
        if "content_contains" in case and (
            not isinstance(case["content_contains"], str)
            or not case["content_contains"]
            or len(sources) != 1
        ):
            raise ValueError(
                "content_contains musi być niepustym tekstem i wymaga jednego źródła."
            )
        expected.append(set(sources))
    # Warm up the model and database before measuring; omit warm-up from metrics.
    warmup = search_results(root, cases[0]["query"], mode="semantic")
    if isinstance(warmup, str):
        raise RuntimeError(warmup)  # noqa: TRY004
    for case, wanted in zip(cases, expected, strict=True):
        for mode in ("semantic", "hybrid", "keyword"):
            start = time.perf_counter()
            results = search_results(
                root, case["query"], limit=10, mode=mode, fragments_per_document=5
            )
            elapsed = (time.perf_counter() - start) * 1000
            if isinstance(results, str):
                raise RuntimeError(results)  # noqa: TRY004
            # Ranks of the returned documents that the case expects.
            ranks = [
                i
                for i, row in enumerate(results, 1)
                if row["source"] in wanted
                and (
                    "content_contains" not in case
                    or any(
                        case["content_contains"] in part["content"]
                        for part in row["fragments"]
                    )
                )
            ]
            observation = (ranks, len(wanted), elapsed)
            grouped[case["category"]][mode].append(observation)
            grouped["all"][mode].append(observation)
    report: dict = {}
    for category, modes in grouped.items():
        report[category] = {}
        for mode, observations in modes.items():
            n = len(observations)
            metrics = {}
            for k in (3, 5, 10):
                # Recall is the share of expected documents found in the top k,
                # MRR the reciprocal rank of the first of them.
                metrics[f"recall@{k}"] = (
                    sum(
                        sum(rank <= k for rank in ranks) / min(wanted, k)
                        for ranks, wanted, _ in observations
                    )
                    / n
                )
                metrics[f"mrr@{k}"] = (
                    sum(
                        1 / ranks[0]
                        for ranks, _, _ in observations
                        if ranks and ranks[0] <= k
                    )
                    / n
                )
            metrics.update(
                count=n,
                median_ms=statistics.median(elapsed for _, _, elapsed in observations),
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
