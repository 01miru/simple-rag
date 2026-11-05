# Lokalny RAG dla Codexa

Wyszukiwanie semantyczne po polskich i wielojęzycznych plikach Markdown. Embeddingi i indeks są tworzone lokalnie. Przy pierwszym użyciu model jest pobierany z Hugging Face; wymaga to dostępu do internetu i miejsca na dysku.

## Instalacja i indeksowanie

Projekt korzysta z Python 3.13. Wymagane są [uv](https://docs.astral.sh/uv/) oraz Python 3.13. W katalogu projektu uruchom:

```sh
uv sync
# Umieść dokumenty .md w knowlage/ (również w podkatalogach).
uv run rag index
```

Po dodaniu, zmianie lub usunięciu dokumentów ponownie uruchom `uv run rag index`. Polecenie aktualizuje tylko zmienione pliki w indeksie `.rag/` i usuwa z niego skasowane dokumenty. Fragmenty są przetwarzane partiami. Dokumenty muszą być zakodowane w UTF-8. Pierwsze uruchomienie po aktualizacji ze starej wersji przebuduje indeks.

## Podłączenie do Codexa

Dodaj serwer, podstawiając bezwzględną ścieżkę do tego projektu:

```sh
codex mcp add lokalna-wiedza -- /ABSOLUTNA/SCIEZKA/DO/rag/.venv/bin/rag serve --directory /ABSOLUTNA/SCIEZKA/DO/rag
```

Sprawdź `codex mcp list` i uruchom ponownie sesję Codexa. Poproś np. „Wyszukaj w lokalnej wiedzy informacje o … i podaj źródła”. Narzędzie `search_knowledge` zwraca fragmenty i ścieżki; Codex wykorzystuje je do przygotowania odpowiedzi. Wskazówki dotyczące konfiguracji MCP znajdują się w [dokumentacji OpenAI](https://developers.openai.com/codex/mcp/).

Indeks i model nie wymagają klucza API. Sama praca Codexa podlega ustawieniom używanej usługi Codex.
