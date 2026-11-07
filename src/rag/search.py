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
    import lancedb
    from lancedb.query import LanceVectorQueryBuilder

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

    vector = _shared.model().encode(query, normalize_embeddings=True).tolist()
    request = cast(LanceVectorQueryBuilder, table.search(vector)).distance_type(
        "cosine"
    )
    if path_prefix:
        prefix = path_prefix.rstrip("/")
        literal = prefix.replace("'", "''")
        escaped = literal.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        predicate = f"path = '{literal}' OR path LIKE '{escaped}/%' ESCAPE '\\'"
        request = request.where(predicate, prefilter=True)
    results = request.limit(limit).to_list()

    if not results and path_prefix:
        return "Brak wyników dla podanej ścieżki."
    return format_results(results) or "Brak wyników."


def format_results(rows: list[dict]) -> str:
    """Format search results with their sources and similarity scores."""
    return "\n\n---\n\n".join(
        f"Źródło: {row['path']}\n"
        f"{f'Sekcja: {row["section"]}\n' if row.get('section') else ''}"
        f"Podobieństwo: {1 - row['_distance']:.4f}\n{row['text']}"
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
