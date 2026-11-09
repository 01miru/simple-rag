"""Command-line entry point and MCP server."""

import argparse
from pathlib import Path

from .indexing import index
from .search import get_document as read_document
from .search import list_documents as find_documents
from .search import search


def main() -> None:
    """Run the indexer or MCP server from command-line arguments."""
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
        server = create_server(root)

        server.run(transport="stdio")


def create_server(root: Path):
    """Create an MCP server for the given project directory."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("Lokalna wiedza")

    @server.tool()
    def search_knowledge(
        query: str, limit: int = 5, path_prefix: str | None = None, mode: str = "hybrid"
    ) -> dict:
        """Wyszukaj fragmenty: hybrid (RRF), semantic (cosinus), keyword (BM25); limit 1..50."""
        return search(root, query, limit, path_prefix, mode)

    @server.tool()
    def list_documents(
        path_prefix: str | None = None, limit: int = 100, offset: int = 0
    ) -> dict:
        """Wylistuj zaindeksowane źródła; używaj do przeglądania i paginacji bazy."""
        return find_documents(root, path_prefix, limit, offset)

    @server.tool()
    def get_document(path: str, start_line: int = 1, max_lines: int = 200) -> dict:
        """Pobierz ograniczony zakres źródła; używaj do weryfikacji i porównania."""
        return read_document(root, path, start_line, max_lines)

    return server
