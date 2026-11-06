# Lokalny RAG dla aplikacji obsługujących MCP

[![Testy](https://github.com/01miru/simple-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/01miru/simple-rag/actions/workflows/tests.yml)

Projekt wyszukuje semantycznie informacje w polskich i wielojęzycznych plikach Markdown. Tworzy embeddingi i indeks lokalnie. Przez serwer MCP można korzystać z wyszukiwania w Codexie oraz innych aplikacjach i agentach obsługujących MCP.

Polecenia `rag index` i `rag serve` uruchamia się w terminalu. Do indeksowania potrzebny jest katalog `knowledge/`. Serwer MCP udostępnia narzędzie `search_knowledge`, które zwraca fragmenty dokumentów wraz ze ścieżkami. Sposób konfiguracji klienta MCP opisuje dokumentacja używanej aplikacji.

## Jak to działa

```mermaid
flowchart LR
    A[Pliki Markdown w knowledge/] -->|rag index| B[Podział na fragmenty]
    B --> C[Embeddingi z sentence-transformers]
    C --> D[(LanceDB w .rag/lancedb/)]
    E[Aplikacja obsługująca MCP] -->|zapytanie przez search_knowledge| F[Serwer MCP]
    F --> G[Embedding zapytania]
    G --> D
    D -->|najbardziej podobne fragmenty i ścieżki| F
    F --> E
```

## Pakiety i baza wektorowa

- LanceDB przechowuje wektory i fragmenty dokumentów w `.rag/lancedb/`, a stan indeksowania plików w `.rag/manifest.json`.
- sentence-transformers tworzy embeddingi za pomocą modelu `paraphrase-multilingual-MiniLM-L12-v2` z Hugging Face.
- PyTorch uruchamia model na CPU albo na karcie NVIDIA obsługującej CUDA.
- FastMCP udostępnia wyszukiwanie przez MCP, korzystając z transportu `stdio`.

Pakiety zainstalujesz poleceniem `uv sync`. Przy pierwszym użyciu model zostanie pobrany z Hugging Face, co wymaga dostępu do internetu i wolnego miejsca na dysku. PyTorch wybiera kartę NVIDIA z CUDA, jeśli jest dostępna; w przeciwnym razie model działa na CPU.

## Instalacja i indeksowanie

Projekt wymaga Python 3.13 i [uv](https://docs.astral.sh/uv/). W katalogu projektu uruchom:

```sh
uv sync
# Umieść dokumenty .md w knowledge/ (również w podkatalogach).
uv run rag index
```

Po dodaniu, zmianie lub usunięciu dokumentów uruchom ponownie `uv run rag index`. Polecenie przetwarza treść partiami, aktualizuje w indeksie `.rag/` zmienione pliki i usuwa z niego te, których już nie ma. Dokumenty muszą być zapisane w UTF-8. Po aktualizacji ze starszej wersji pierwsze uruchomienie przebuduje indeks.

## Testy jakości wyszukiwania

Szybkie testy uruchom przez `uv run pytest`. Test jakości korzysta z rzeczywistego modelu: indeksuje sześć przykładowych dokumentów i sprawdza zapytania po polsku i angielsku. Uruchomisz go poleceniem:

```sh
RAG_QUALITY_TEST=1 uv run pytest -q tests/test_retrieval_quality.py
```

Test wymaga co najmniej 5 trafień na 6 zapytań w pierwszych trzech wynikach (Recall@3). Średnia odwrotność pozycji właściwego wyniku (MRR@3) musi wynosić co najmniej 0,65. Przy pierwszym uruchomieniu model może zostać pobrany. Zestaw jest niewielki i służy do wykrywania regresji; nie mierzy jakości wyszukiwania we własnych dokumentach.

## Podłączenie do Codexa

Podłącz Codexa do serwera MCP poniższym poleceniem. Zastąp przykładowe ścieżki bezwzględną ścieżką do projektu:

```sh
codex mcp add lokalna-wiedza -- /ABSOLUTNA/SCIEZKA/DO/rag/.venv/bin/rag serve --directory /ABSOLUTNA/SCIEZKA/DO/rag
```

Sprawdź konfigurację poleceniem `codex mcp list`, a potem uruchom ponownie sesję Codexa. Możesz poprosić na przykład: „Wyszukaj w lokalnej wiedzy informacje o … i podaj źródła”. Codex korzysta wtedy z fragmentów dokumentów i ścieżek zwróconych przez `search_knowledge`. Więcej informacji o konfiguracji MCP znajdziesz w [dokumentacji OpenAI](https://developers.openai.com/codex/mcp/).

Do indeksu i modelu nie potrzebujesz klucza API. Korzystanie z Codexa podlega ustawieniom wybranej usługi Codex.
