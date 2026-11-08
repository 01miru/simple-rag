"""Search and retrieve indexed documents."""

import hashlib
import json
from pathlib import Path
from typing import cast

from . import _shared
from ._shared import TABLE


def search(
    root: Path, query: str, limit: int = 5, path_prefix: str | None = None
) -> str:
    """Search the index and return formatted matching fragments."""
    return _search(root, query, limit, path_prefix)


def _search(
    root: Path, query: str, limit: int, path_prefix: str | None, *, mcp: bool = False
) -> str:
    result = search_results(root, query, limit, path_prefix, mcp=mcp, mode="semantic")
    if isinstance(result, str):
        return result
    return format_results(result) or "Brak wyników."


def search_results(
    root: Path,
    query: str,
    limit: int = 5,
    path_prefix: str | None = None,
    *,
    mcp: bool = False,
    mode: str = "hybrid",
) -> list[dict] | str:
    """Search with semantic and/or full-text ranking."""
    import lancedb
    from lancedb.query import LanceVectorQueryBuilder

    if mode not in ("hybrid", "semantic", "keyword"):
        raise ValueError("mode musi być jednym z: hybrid, semantic, keyword.")

    if not query.strip():
        return "Podaj niepuste zapytanie."

    db_path = root / ".rag" / "lancedb"
    if not db_path.exists():
        return "Brak indeksu. Uruchom `rag index`."

    db = lancedb.connect(str(db_path))
    if TABLE not in db.list_tables():
        if mcp:
            return "Brak indeksu. Uruchom `rag index`."
        return "Brak dokumentów w indeksie. Dodaj pliki .md i uruchom `rag index`."
    table = db.open_table(TABLE)
    if not mcp and table.count_rows() == 0:
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

    rankings = []
    candidate_k = max(limit * 5, 30) if mode == "hybrid" else limit
    if mode in ("hybrid", "semantic"):
        vector = (
            _shared.model()
            .encode(f"query: {query}", normalize_embeddings=True)
            .tolist()
        )
        request = cast(LanceVectorQueryBuilder, table.search(vector)).distance_type(
            "cosine"
        )
        if predicate:
            request = request.where(predicate, prefilter=True)
        rankings.append(("semantic", request.limit(candidate_k).to_list()))
    if mode in ("hybrid", "keyword"):
        fts_request = table.search(query, query_type="fts", fts_columns="fts_text")
        if predicate:
            fts_request = fts_request.where(predicate, prefilter=True)
        rankings.append(("keyword", fts_request.limit(candidate_k).to_list()))

    merged: dict[str, dict] = {}
    for method, rows in rankings:
        seen = set()
        for rank, row in enumerate(rows, 1):
            key = row["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(
                key,
                {
                    "source": row["path"],
                    "section": row["section"],
                    "content": row["text"],
                    "score": 0.0,
                    "semantic_similarity": None,
                    "matched_by": [],
                },
            )
            item["matched_by"].append(method)
            if method == "semantic":
                item["semantic_similarity"] = 1 - row["_distance"]
            item["score"] += (
                (0.8 if method == "semantic" else 0.2) / (1 + rank)
                if mode == "hybrid"
                else (1 - row["_distance"] if method == "semantic" else row["_score"])
            )
    return sorted(merged.values(), key=lambda item: -item["score"])[:limit]


def format_results(rows: list[dict]) -> str:
    """Format semantic results as text."""
    return "\n\n---\n\n".join(
        f"Źródło: {row['source']}\n"
        f"{f'Sekcja: {row["section"]}\n' if row.get('section') else ''}"
        f"Podobieństwo: {row['score']:.4f}\n{row['content']}"
        for row in rows
    )


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
