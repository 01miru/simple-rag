"""Local Markdown search exposed through MCP."""

import argparse
import hashlib
import json
import re
from functools import lru_cache
from itertools import islice
from pathlib import Path

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TABLE = "knowledge"
BATCH_SIZE = 64
INDEX_VERSION = f"{MODEL}:markdown-tokens-v1"


@lru_cache(maxsize=1)
def model():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging

    logging.disable_progress_bar()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL, device=device)


def chunks(text: str, tokenizer, limit: int):
    """Split Markdown into sections and blocks within the model's token limit."""

    def fits(value: str) -> bool:
        return (
            len(tokenizer.encode(value, add_special_tokens=True, truncation=False))
            <= limit
        )

    headings = []
    section = ""
    fragment = ""
    block = []
    fence = None

    def add_block(value):
        nonlocal fragment
        value = value.strip()
        if not value:
            return
        combined = f"{fragment}\n\n{value}" if fragment else value
        if fits(combined):
            fragment = combined
            return
        if fragment:
            yield {"text": fragment, "section": section}
            fragment = ""
        while value and not fits(value):
            # Oversized blocks must be split; find the longest prefix that fits.
            low, high = 1, len(value)
            while low < high:
                mid = (low + high + 1) // 2
                if fits(value[:mid]):
                    low = mid
                else:
                    high = mid - 1
            if not fits(value[:low]):
                raise ValueError("Limit tokenów jest zbyt mały dla tekstu.")
            yield {"text": value[:low], "section": section}
            value = value[low:]
        fragment = value

    lines = text.splitlines()
    for i, line in enumerate(lines):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        heading = (
            re.match(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$", line) if not fence else None
        )
        underline = (
            re.fullmatch(r" {0,3}(=+|-+)[ \t]*", lines[i + 1])
            if not fence and not block and line.strip() and i + 1 < len(lines)
            else None
        )
        if heading or underline:
            yield from add_block("\n".join(block))
            block = []
            if fragment:
                yield {"text": fragment, "section": section}
                fragment = ""
            level = (
                len(heading.group(1))
                if heading
                else (1 if underline.group(1)[0] == "=" else 2)
            )
            headings = [(depth, title) for depth, title in headings if depth < level]
            title = (
                re.sub(r"\s+#+\s*$", "", heading.group(2)).strip()
                if heading
                else line.strip()
            )
            headings.append((level, title))
            section = " > ".join(title for _, title in headings)
        if marker:
            mark = marker.group(1)
            if fence is None:
                fence = mark
            elif (
                mark[0] == fence[0]
                and len(mark) >= len(fence)
                and not line[marker.end() :].strip()
            ):
                fence = None
        if not line.strip() and not fence:
            yield from add_block("\n".join(block))
            block = []
        else:
            block.append(line)
    yield from add_block("\n".join(block))
    if fragment:
        yield {"text": fragment, "section": section}


def index(root: Path) -> int:
    import lancedb

    source = root / "knowledge"
    if not source.is_dir():
        raise ValueError(f"Brak katalogu {source}")

    db = lancedb.connect(str(root / ".rag" / "lancedb"))
    manifest_path = root / ".rag" / "manifest.json"
    exists = TABLE in db.list_tables()

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

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        temporary.replace(manifest_path)

    def delete_chunks(path: str) -> None:
        if exists:
            escaped = path.replace("'", "''")
            db.open_table(TABLE).delete(f"path = '{escaped}'")

    markdown_files = {
        path.relative_to(root).as_posix(): path for path in source.rglob("*.md")
    }

    for relative_path in files.keys() - markdown_files.keys():
        files[relative_path] = None
        save_manifest()
        delete_chunks(relative_path)
        del files[relative_path]
        save_manifest()

    indexed_chunks = 0

    for relative_path, file_path in sorted(markdown_files.items()):
        data = file_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if files.get(relative_path) == digest:
            continue

        text = data.decode("utf-8")
        files[relative_path] = None
        save_manifest()
        delete_chunks(relative_path)

        if not text.strip():
            files[relative_path] = digest
            save_manifest()
            continue
        encoder = model()
        fragments = iter(chunks(text, encoder.tokenizer, encoder.max_seq_length))
        while batch := list(islice(fragments, BATCH_SIZE)):
            vectors = encoder.encode(
                [part["text"] for part in batch], normalize_embeddings=True
            )
            records = [
                {
                    "path": relative_path,
                    **fragment,
                    "vector": vector.tolist(),
                }
                for fragment, vector in zip(batch, vectors, strict=True)
            ]
            if exists:
                db.open_table(TABLE).add(records)
            else:
                db.create_table(TABLE, data=records)
                exists = True
            indexed_chunks += len(records)

        files[relative_path] = digest
        save_manifest()

    return indexed_chunks


def search(root: Path, query: str, limit: int = 5) -> str:
    import lancedb

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

    vector = model().encode(query, normalize_embeddings=True).tolist()
    results = table.search(vector).distance_type("cosine").limit(limit).to_list()

    return (
        "\n\n---\n\n".join(
            f"Źródło: {row['path']}\n"
            f"{f'Sekcja: {row["section"]}\n' if row.get('section') else ''}"
            f"Podobieństwo: {1 - row['_distance']:.4f}\n{row['text']}"
            for row in results
        )
        or "Brak wyników."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lokalny RAG dla dokumentów Markdown")
    parser.add_argument("command", choices=("index", "serve"))
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="katalog projektu zawierający knowledge",
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
            """Zwróć fragmenty dokumentów, identyfikatory źródeł i podobieństwo."""
            return search(root, query)

        server.run(transport="stdio")
