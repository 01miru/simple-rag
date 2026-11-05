"""Local Markdown search exposed through MCP."""

import argparse
import hashlib
import json
from functools import lru_cache
from itertools import islice
from pathlib import Path

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TABLE = "knowledge"
BATCH_SIZE = 64
INDEX_VERSION = f"{MODEL}:chunks-1000-150-v1"


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL, device="cpu")


def chunks(text: str, size: int = 1000, overlap: int = 150):
    """Split text into bounded, overlapping fragments, preferably at line breaks."""
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        fragment = text[start:end].strip()
        if fragment:
            yield fragment
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def index(root: Path) -> int:
    import lancedb

    source = root / "knowlage"
    if not source.is_dir():
        raise ValueError(f"Brak katalogu {source}")
    db = lancedb.connect(str(root / ".rag" / "lancedb"))
    manifest_path = root / ".rag" / "manifest.json"
    exists = TABLE in db.table_names()
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    if manifest.get("version") != INDEX_VERSION or (
        manifest.get("files") and not exists
    ):
        if exists:
            db.drop_table(TABLE)
            exists = False
        manifest = {"version": INDEX_VERSION, "files": {}}
    files = manifest["files"]

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        temporary.replace(manifest_path)

    def delete(path):
        if exists:
            escaped = path.replace("'", "''")
            db.open_table(TABLE).delete(f"path = '{escaped}'")

    paths = {path.relative_to(root).as_posix(): path for path in source.rglob("*.md")}
    for name in files.keys() - paths.keys():
        files[name] = None
        save()
        delete(name)
        del files[name]
        save()

    count = 0
    for name, path in sorted(paths.items()):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if files.get(name) == digest:
            continue
        text = data.decode("utf-8")
        files[name] = None
        save()
        delete(name)
        fragments = iter(chunks(text))
        while batch := list(islice(fragments, BATCH_SIZE)):
            vectors = model().encode(batch, normalize_embeddings=True)
            records = [
                {"path": name, "text": fragment, "vector": vector.tolist()}
                for fragment, vector in zip(batch, vectors, strict=True)
            ]
            if exists:
                db.open_table(TABLE).add(records)
            else:
                db.create_table(TABLE, data=records)
                exists = True
            count += len(records)
        files[name] = digest
        save()
    return count


def search(root: Path, query: str, limit: int = 5) -> str:
    import lancedb

    if not query.strip():
        return "Podaj niepuste zapytanie."
    db_path = root / ".rag" / "lancedb"
    if not db_path.exists():
        return "Brak indeksu. Uruchom `rag index`."
    db = lancedb.connect(str(db_path))
    if TABLE not in db.table_names():
        return "Brak dokumentów w indeksie. Dodaj pliki .md i uruchom `rag index`."
    table = db.open_table(TABLE)
    if table.count_rows() == 0:
        return "Brak dokumentów w indeksie. Dodaj pliki .md i uruchom `rag index`."
    vector = model().encode(query, normalize_embeddings=True).tolist()
    results = table.search(vector).limit(limit).to_list()
    return (
        "\n\n---\n\n".join(f"Źródło: {row['path']}\n{row['text']}" for row in results)
        or "Brak wyników."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lokalny RAG dla dokumentów Markdown")
    parser.add_argument("command", choices=("index", "serve"))
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="katalog projektu zawierający knowlage",
    )
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    if args.command == "index":
        try:
            count = index(root)
        except (OSError, UnicodeError, ValueError) as exc:
            parser.exit(1, f"Błąd indeksowania: {exc}\n")
        print(f"Przetworzono {count} nowych lub zmienionych fragmentów.")
    else:
        from mcp.server.fastmcp import FastMCP

        server = FastMCP("Lokalna wiedza")

        @server.tool()
        def search_knowledge(query: str) -> str:
            """Wyszukaj informacje w lokalnych dokumentach Markdown i zwróć źródła."""
            return search(root, query)

        server.run(transport="stdio")
