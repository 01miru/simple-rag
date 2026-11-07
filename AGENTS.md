# Repository Guidelines

## Project Structure & Module Organization

`src/rag/` contains the Python package: `indexing.py` chunks and indexes Markdown, `search.py` handles retrieval, `server.py` provides the CLI and MCP server, and `_shared.py` holds shared resources. Tests live in `tests/test_*.py`. Put UTF-8 Markdown sources in `knowledge/`, including subdirectories. The generated LanceDB index and manifest live in `.rag/`; they are local runtime data.

## Build, Test, and Development Commands

Use Python 3.13 and `uv`:

- `uv sync --locked` installs the locked dependencies, including development tools.
- `uv run rag index` indexes documents in `knowledge/`; rerun it after adding, changing, or deleting files.
- `uv run rag serve` starts the MCP server over stdio. Both CLI commands accept `--directory PATH`.
- `uv run pytest` runs the standard test suite; `uv run mypy .` runs the type checker, as in CI.
- `uv run pre-commit run --all-files` runs Ruff checks, formatting, and mypy. Ruff hooks may modify files.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` functions and modules, and `PascalCase` classes. Add type annotations where practical. Ruff enforces lint rules and Google-style docstrings for source code; tests are exempt from docstring rules. Use `uv run ruff check .` and `uv run ruff format .` to check and format code.

## Testing Guidelines

Tests use pytest, with `unittest` and mocks in parts of the suite. Name files `test_*.py` and test methods `test_*`; use temporary directories and fake models for fast, isolated tests. No general coverage threshold is configured. For retrieval changes, optionally run `RAG_QUALITY_TEST=1 uv run pytest -q tests/test_retrieval_quality.py`; this uses the real embedding model and checks Recall@3 and MRR@3 thresholds.

## Commit & Pull Request Guidelines

Recent commits use short, informal, lowercase descriptions such as `updated chunk function`; keep messages concise and describe the change. Pull requests should explain the behavior changed, note relevant issues, and report checks run. Include examples when changing CLI or MCP behavior, and update the README when usage changes.

## Configuration & Data

The first real embedding-model use may download the model from Hugging Face. Avoid committing local indexes or private knowledge documents; review `knowledge/` content before adding it to Git.
