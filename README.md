# Lokalny RAG dla Codexa

[![Testy](https://github.com/01miru/simple-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/01miru/simple-rag/actions/workflows/tests.yml)

Semantyczne wyszukiwanie w polskich i wielojęzycznych plikach Markdown. Embeddingi i indeks powstają lokalnie. Przy pierwszym użyciu program pobiera model z Hugging Face, więc potrzebny jest dostęp do internetu i wolne miejsce na dysku.

Jeśli PyTorch wykryje kartę obsługującą CUDA, model użyje jej automatycznie. W przeciwnym razie obliczenia będą wykonywane na CPU.

## Instalacja i indeksowanie

Projekt wymaga Python 3.13 i [uv](https://docs.astral.sh/uv/). W katalogu projektu uruchom:

```sh
uv sync
# Umieść dokumenty .md w knowledge/ (również w podkatalogach).
uv run rag index
```

Po dodaniu, zmianie lub usunięciu dokumentów uruchom ponownie `uv run rag index`. Polecenie aktualizuje w indeksie `.rag/` tylko zmienione pliki i usuwa dokumenty, których już nie ma. Przetwarza treść partiami. Dokumenty muszą być zapisane w UTF-8. Po aktualizacji ze starszej wersji pierwsze uruchomienie przebuduje indeks.

## Podłączenie do Codexa

Dodaj serwer. W poleceniu zastąp przykładowe ścieżki bezwzględną ścieżką do projektu:

```sh
codex mcp add lokalna-wiedza -- /ABSOLUTNA/SCIEZKA/DO/rag/.venv/bin/rag serve --directory /ABSOLUTNA/SCIEZKA/DO/rag
```

Sprawdź konfigurację poleceniem `codex mcp list`, a potem uruchom ponownie sesję Codexa. Możesz poprosić na przykład: „Wyszukaj w lokalnej wiedzy informacje o … i podaj źródła”. Narzędzie `search_knowledge` zwraca fragmenty dokumentów i ich ścieżki, z których Codex korzysta przy tworzeniu odpowiedzi. Więcej informacji o konfiguracji MCP znajdziesz w [dokumentacji OpenAI](https://developers.openai.com/codex/mcp/).

Do indeksu i modelu nie potrzebujesz klucza API. Korzystanie z Codexa podlega ustawieniom wybranej usługi Codex.
