# Repository Guidelines

## Project Structure

- `src/rag/__init__.py` contains the package, CLI, indexing logic, and MCP server.
- `tests/test_rag.py` covers chunking, model selection, indexing, recovery, and search.
- `README.md` documents installation and use. `.github/workflows/tests.yml` defines CI.
- Put Markdown knowledge files under `knowledge/` (including subdirectories); `.rag/` holds the generated local index. Both are ignored by Git.

## Build, Test, and Development

- `uv sync` installs the project and development dependencies from `uv.lock`.
- `uv run rag index` indexes or updates Markdown files in the current directory.
- `uv run rag serve --directory /path/to/project` starts the MCP server for that project.
- `uv run pytest` runs the test suite; CI uses `uv sync --locked` followed by this command.
- `uv run ruff check .` checks lint; `uv run ruff format --check .` checks formatting.

Use Python 3.13 for parity with CI. Indexing and search may download the embedding model on first use.

## Style and Naming

Follow Ruff formatting and lint rules, with four-space Python indentation and type annotations where useful. Use `snake_case` for functions and variables, `PascalCase` for classes, and descriptive `test_...` test names. Keep runtime imports lazy where they avoid loading heavy ML dependencies for unrelated commands.

## Testing

Tests use `unittest` assertions and mocking, collected by pytest. Add or update focused tests in `tests/test_rag.py` for behavior changes, especially index updates, deleted files, empty inputs, and interrupted indexing. There is no stated coverage threshold.

## Commits and Pull Requests

Recent commits use short, lowercase summaries (for example, `reformat` and `added ruff`); no stricter convention is established. Keep commit subjects concise and action-oriented. Pull requests should explain the user-visible or operational change, link related issues when applicable, and include test results. Include screenshots only when a change adds a visual interface.

## Configuration and Data

Do not commit `.rag/`, `knowledge/`, credentials, or downloaded model files. Use UTF-8 for Markdown input. The index is local and can be regenerated with `uv run rag index`.
