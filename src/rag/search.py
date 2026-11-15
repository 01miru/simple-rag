"""Search and retrieve indexed documents."""

import hashlib
import json
from pathlib import Path
from typing import cast

from . import _shared
from ._shared import TABLE

# Weighted Reciprocal Rank Fusion of the semantic and keyword document rankings.
WEIGHTS = {"semantic": 0.8, "keyword": 0.2}
RRF_K = 1
CANDIDATES = 50
COLUMNS = ["path", "section", "text", "chunk_id", "start_line", "end_line"]


def search(
    root: Path,
    query: str,
    limit: int = 5,
    path_prefix: str | None = None,
    mode: str = "hybrid",
    fragments_per_document: int = 2,
) -> dict:
    """Search the index and return structured results or an error."""
    result = search_results(
        root,
        query,
        limit,
        path_prefix,
        mode=mode,
        fragments_per_document=fragments_per_document,
    )
    return {"error": result} if isinstance(result, str) else {"results": result}


def search_results(
    root: Path,
    query: str,
    limit: int = 5,
    path_prefix: str | None = None,
    *,
    mode: str = "hybrid",
    fragments_per_document: int = 2,
) -> list[dict] | str:
    """Rank documents and return each with its best-matching fragments.

    Args:
        root: Project directory containing the index.
        query: Question or phrase to look for.
        limit: Maximum number of documents to return.
        path_prefix: Optional file or directory path to search within.
        mode: Ranking method: hybrid, semantic, or keyword.
        fragments_per_document: Maximum number of fragments per document.

    Returns:
        Documents ordered by relevance, or an error message.

    Raises:
        ValueError: If a numeric argument or the mode is out of range.
    """
    import lancedb
    from lancedb.query import LanceQueryBuilder, LanceVectorQueryBuilder

    if not 1 <= limit <= 50:
        raise ValueError("limit musi być z zakresu 1..50.")
    if not 1 <= fragments_per_document <= 10:
        raise ValueError("fragments_per_document musi być z zakresu 1..10.")
    if mode not in ("hybrid", "semantic", "keyword"):
        raise ValueError("mode musi być jednym z: hybrid, semantic, keyword.")

    if not query.strip():
        return "Podaj niepuste zapytanie."

    db_path = root / ".rag" / "lancedb"
    if not db_path.exists():
        return "Brak indeksu. Uruchom `rag index`."

    db = lancedb.connect(str(db_path))
    if TABLE not in db.list_tables():
        return "Brak dokumentów w indeksie. Dodaj pliki .md i uruchom `rag index`."
    table = db.open_table(TABLE)
    if table.count_rows() == 0:
        return "Brak dokumentów w indeksie. Dodaj pliki .md i uruchom `rag index`."

    try:
        manifest = json.loads(
            (root / ".rag" / "manifest.json").read_text(encoding="utf-8")
        )
        compatible = manifest.get(
            "version"
        ) == _shared.INDEX_VERSION and not manifest.get("fts_dirty")
    except (OSError, ValueError, AttributeError):
        compatible = False
    if not compatible:
        return "Indeks wymaga przebudowy. Uruchom `rag index`."

    predicate = None
    if path_prefix:
        prefix = path_prefix.rstrip("/")
        literal = prefix.replace("'", "''")
        escaped = literal.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        predicate = f"path = '{literal}' OR path LIKE '{escaped}/%' ESCAPE '\\'"

    vector = None
    if mode in ("hybrid", "semantic"):
        vector = (
            _shared.model()
            .encode(f"query: {query}", normalize_embeddings=True)
            .tolist()
        )
    batch_size = max(limit * 10, CANDIDATES)

    def ranking(method: str) -> list[dict]:
        """Return fragments in relevance order, covering `limit` documents."""
        score = "_distance" if method == "semantic" else "_score"
        rows: list[dict] = []
        paths: set[str] = set()
        while True:
            # A long document can fill a whole batch with its own fragments, so
            # ask again without the documents already found until enough appear.
            request: LanceQueryBuilder
            if method == "semantic":
                request = cast(
                    LanceVectorQueryBuilder, table.search(vector)
                ).distance_type("cosine")
            else:
                request = table.search(query, query_type="fts", fts_columns="fts_text")
            quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in sorted(paths))
            conditions = [
                f"({condition})"
                for condition in (predicate, quoted and f"path NOT IN ({quoted})")
                if condition
            ]
            if conditions:
                request = request.where(" AND ".join(conditions), prefilter=True)
            batch = request.select([*COLUMNS, score]).limit(batch_size).to_list()
            rows += batch
            paths.update(row["path"] for row in batch)
            if len(paths) >= limit or len(batch) < batch_size:
                return rows

    documents: dict[str, dict] = {}
    for method in ("semantic", "keyword"):
        if mode not in ("hybrid", method):
            continue
        counts: dict[str, int] = {}
        for row in ranking(method):
            path = row["path"]
            value = 1 - row["_distance"] if method == "semantic" else row["_score"]
            document = documents.setdefault(
                path, {"source": path, "score": 0.0, "matched_by": [], "fragments": {}}
            )
            # Ranks count documents, and fragments within their own document.
            rank = counts[path] = counts.get(path, 0) + 1
            if rank == 1:
                document["matched_by"].append(method)
                document["score"] += (
                    WEIGHTS[method] / (RRF_K + len(counts))
                    if mode == "hybrid"
                    else value
                )
            order, fragment = document["fragments"].setdefault(
                row["chunk_id"],
                (
                    [0.0],
                    {
                        "section": row["section"],
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                        "content": row["text"],
                        "semantic_similarity": None,
                        "matched_by": [],
                    },
                ),
            )
            fragment["matched_by"].append(method)
            if method == "semantic":
                fragment["semantic_similarity"] = value
            order[0] += WEIGHTS[method] / (RRF_K + rank) if mode == "hybrid" else value

    results = sorted(documents.values(), key=lambda item: -item["score"])[:limit]
    for document in results:
        best = sorted(document["fragments"].values(), key=lambda pair: -pair[0][0])
        document["fragments"] = [
            fragment for _, fragment in best[:fragments_per_document]
        ]
    return results


def _indexed(root: Path) -> list[str]:
    manifest = root / ".rag" / "manifest.json"
    if not manifest.is_file():
        return []
    try:
        files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
        return sorted(path for path, digest in files.items() if digest is not None)
    except (OSError, ValueError):
        return []


def _matches(path: str, prefix: str | None) -> bool:
    if prefix is None:
        return True
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def list_documents(
    root: Path, path_prefix: str | None = None, limit: int = 100, offset: int = 0
) -> dict:
    """List indexed documents with optional prefix filtering and pagination."""
    if not 1 <= limit <= 500 or offset < 0:
        raise ValueError("limit musi być z zakresu 1..500, a offset nieujemny.")
    paths = [p for p in _indexed(root) if _matches(p, path_prefix)]
    page = paths[offset : offset + limit]
    return {
        "documents": [{"path": p} for p in page],
        "total": len(paths),
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(page) < len(paths),
    }


def get_document(
    root: Path, path: str, start_line: int = 1, max_lines: int = 200
) -> dict:
    """Read lines from an unchanged indexed Markdown document.

    Raises:
        ValueError: If the path, line range, or indexed content is invalid.
    """
    if start_line < 1 or not 1 <= max_lines <= 500:
        raise ValueError(
            "start_line musi być >=1, max_lines musi być z zakresu 1..500."
        )
    source = (root / "knowledge").resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(source) or target.suffix.lower() != ".md":
        raise ValueError("Dozwolone są wyłącznie pliki .md wewnątrz knowledge/.")
    relative = target.relative_to(root.resolve()).as_posix()
    if relative not in _indexed(root):
        raise ValueError("Dokument nie jest poprawnie zaindeksowany.")
    data = target.read_bytes()
    if (
        hashlib.sha256(data).hexdigest()
        != json.loads((root / ".rag/manifest.json").read_text(encoding="utf-8"))[
            "files"
        ][relative]
    ):
        raise ValueError(
            "Dokument zmienił się od indeksowania; wymagane ponowne indeksowanie."
        )
    lines = data.decode("utf-8").splitlines()
    selected = lines[start_line - 1 : start_line - 1 + max_lines]
    return {
        "path": relative,
        "start_line": start_line,
        "end_line": start_line + len(selected) - 1,
        "total_lines": len(lines),
        "content": "\n".join(selected),
        "has_more": start_line - 1 + len(selected) < len(lines),
    }
