# Pomiary wyszukiwania

8 listopada 2026, lokalny model multilingual-e5-large, małe syntetyczne bazy testowe, limit 5, ciepły proces. Tryb semantic w obecnym kodzie odwzorowuje wcześniejsze wyszukiwanie wektorowe; pomiar wykonano po wdrożeniu FTS, a nie na osobnym checkoutcie starej wersji.

| Zestaw | Tryb | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 6 pierwotnych pytań | semantic | 1,000 | 0,917 | 1,000 | 0,917 |
| 6 pierwotnych pytań | hybrid, równe wagi, k=60 | 1,000 | 0,806 | 1,000 | 0,806 |
| 6 pierwotnych pytań | hybrid, wagi 0,8/0,2, k=1 | 1,000 | 0,917 | 1,000 | 0,917 |
| 18 leksykalnych | semantic | 1,000 | 1,000 | 1,000 | 1,000 |
| 18 leksykalnych | hybrid | 1,000 | 1,000 | 1,000 | 1,000 |
| 18 leksykalnych | keyword | 0,944 | 0,944 | 0,944 | 0,944 |

Dla zestawu leksykalnego średni czas semantic wyniósł 130,3 ms, hybrid 134,8 ms, keyword 3,0 ms. To pomiar na 18 krótkich dokumentach, bez wnioskowania o typowych większych bazach. Kolejność trybów i rozgrzewanie mogą wpływać na czas.

Pierwotna konfiguracja obniżała MRR@3 o 0,111; wagi 0,8/0,2 i k=1 przywróciły wynik na tych sześciu pytaniach. To mała próbka, a zmiana była do niej dostrajana. Zestaw leksykalny jest za łatwy, by wykazać poprawę o 10 p.p. Wagi RRF 0,6/0,4 i 0,9/0,1 nie usunęły regresji; następnie sprawdzono wagi 0,8/0,2 z k=1. Polska fleksja nie jest gwarantowana przez aktualną tokenizację. Potrzebny jest trudniejszy, reprezentatywny korpus oraz porównanie tokenizacji przed stwierdzeniem poprawy jakości.

Uruchomienie: `RAG_QUALITY_TEST=1 uv run pytest -s tests/test_retrieval_quality.py tests/test_lexical_quality.py`. Test porównawczy sprawdza brak regresji dla sześciu pierwotnych pytań.


## Podobne długie zgłoszenia

Niezależna syntetyczna próba obejmuje 20 podobnych dokumentów zawierających różne kody `RX-...`; zapytania wskazują konkretny kod. Semantyczny i hybrydowy uzyskały Recall@3 i Recall@5 1,000, a MRR@3 i MRR@5 0,975. Keyword uzyskał 1,000 dla wszystkich czterech metryk. Pomiar z 8 listopada 2026 wykonano modelem multilingual-e5-large, limit 5.

Ta próba też nie pozwala wykazać poprawy Recall@3 o 10 punktów: wynik semantyczny osiąga maksimum. Nie stanowi reprezentatywnej próbki rzeczywistych dokumentów; identyfikatory są syntetyczne, a treść rekordów powtarzalna. Test sprawdza brak regresji, nie uzyskanie wymaganego zysku.


## Polska fleksja: porównanie tokenizerów FTS

Próba 12 zapytań z odmienionymi formami i 12 podobnych tematycznie lub brzmieniowo dokumentów rozpraszających. Indeks simple (bez stemmingu) osiągnął Recall@3 i MRR@3 0,417. Indeks ngram 3–5 osiągnął Recall@3 1,000 i MRR@3 0,917; dwa trafienia były na pozycji 2.

W osobnej próbie kodów `AB`, `VPN`, `RX-104`, `742` tokenizer simple znalazł wszystkie na pierwszej pozycji; ngram nie znalazł dwuznakowego `AB`, pozostałe były pierwsze. Próba mierzy wyłącznie FTS, nie ranking hybrydowy, i jest zbyt mała, by uzasadnić zmianę domyślnego indeksu. Konieczne jest sprawdzenie rozmiaru i czasu indeksu oraz niezależnego korpusu; przed ewentualnym wdrożeniem trzeba zachować wyszukiwanie krótkich identyfikatorów.

Uruchomienie: `RAG_QUALITY_TEST=1 uv run pytest -s tests/test_polish_tokenization.py`.


## Połączenie tokenizerów i ranking hybrydowy

Eksperyment na tych samych 12 parach form, 12 dokumentach rozpraszających i czterech kodach. Dwa rankingi FTS połączono równym RRF (k=1); następnie wynik połączono z rankingiem semantycznym wagami 0,8/0,2 (k=1), zgodnie z obecną konfiguracją hybrydy. Użyto modelu multilingual-e5-large i do 30 kandydatów z każdej gałęzi.

| Metoda | Recall@3 fleksja | MRR@3 fleksja | Pozycje kodów AB, VPN, RX-104, 742 |
| --- | ---: | ---: | --- |
| Obecny keyword | 0,417 | 0,417 | 1, 1, 1, 1 |
| Połączone FTS | 1,000 | 0,917 | 1, 1, 1, 1 |
| Obecny hybrid | 1,000 | 0,958 | 1, 1, 1, 1 |
| Hybrid z połączonym FTS | 1,000 | 0,958 | 1, 1, 1, 1 |

Połączenie pomogło samemu wyszukiwaniu słownemu, ale nie poprawiło hybrydy na tej małej próbce. To eksperymentalna implementacja wyłącznie w teście: nie uwzględnia kosztu budowy dodatkowego indeksu ani reprezentatywnego korpusu. Produkcyjny indeks pozostawiono bez zmian.

Uruchomienie: `RAG_QUALITY_TEST=1 uv run pytest -s tests/test_polish_tokenization.py::test_combined_tokenizers_with_hybrid`.

## Koszt drugiego indeksu FTS

Pomiar z 8 listopada 2026: 2000 syntetycznych, krótkich, bardzo podobnych dokumentów; lokalna LanceDB, wektory testowe o 16 wymiarach, 30 kandydatów na gałąź, pięć zapytań po czterokrotnym powtórzeniu po rozgrzaniu. Jedna baza miała indeks simple, druga dodatkową kopię tekstu i indeks ngram 3–5. Czas budowy obejmuje tworzenie indeksów FTS, a czas zapytania obejmuje wyszukiwanie wektorowe, FTS i fuzję w Pythonie; nie obejmuje generowania embeddingów ani pełnej ścieżki aplikacji.

| Wariant | Budowa FTS | Rozmiar bazy | Mediana zapytania | Średnia zapytania |
| --- | ---: | ---: | ---: | ---: |
| simple | 0,015 s | 369 434 B | 3,43 ms | 3,31 ms |
| simple + ngram | 0,169 s | 710 448 B | 7,40 ms | 10,32 ms |

Na tej próbie dodatkowa gałąź zwiększyła rozmiar o 341 014 B, czas budowy FTS o 0,154 s i medianę części wyszukiwania o 3,97 ms. Dane są powtarzalne i dobrze się kompresują, a wektory są mniejsze niż rzeczywiste; wynik nie jest prognozą dla produkcyjnej bazy. Nie potwierdza też pełnego kryterium opóźnienia poniżej 50 ms. Próba kosztu nie mierzy jakości wyników.

Uruchomienie: `RAG_COST_TEST=1 uv run pytest -s tests/test_fts_cost.py`.

## Nowe pytania do podręcznika

Dwanaście pytań do istniejącego `tests/fixtures/podrecznik.md`, innych niż trzy wcześniej używane pytania do tego dokumentu; oczekiwany wynik sprawdzany jest po fragmencie treści, limit 5. Pytania nie służyły do dostrajania RRF, ale sam dokument był już częścią testów jakości, więc nie jest niezależnym korpusem. Pomiar wykonano lokalnym modelem multilingual-e5-large.

| Tryb | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| semantic | 0,917 | 0,917 | 1,000 | 0,938 |
| hybrid | 1,000 | 0,944 | 1,000 | 0,944 |

W 11 przypadkach oba tryby umieściły oczekiwany fragment na pierwszej pozycji; w pozostałym hybryda przesunęła go z czwartej na trzecią. Ten wynik nie zastępuje niezależnego, reprezentatywnego korpusu dokumentów ani dowodu poprawy o 10 p.p. dla zapytań leksykalnych.

Uruchomienie: `RAG_QUALITY_TEST=1 uv run pytest -s tests/test_retrieval_quality.py::test_independent_long_document_comparison`.

## Niezależny zestaw syntetyczny sprawdzony narzędziem oceny

Przygotowano 12 nowych krótkich dokumentów biurowych i 24 pytania: po 12 identyfikatorów lub nazw własnych oraz parafraz. Dokumenty i pytania nie były używane przy strojeniu RRF. Pomiar wykonano lokalnym modelem multilingual-e5-large za pomocą `rag.evaluate`; czas obejmuje pełną funkcję wyszukiwania po rozgrzaniu modelu.

| Pytania | Tryb | Recall@3 | MRR@3 | Recall@5 | MRR@5 | Mediana |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| leksykalne | semantic | 1,000 | 1,000 | 1,000 | 1,000 | 128,9 ms |
| leksykalne | hybrid | 1,000 | 1,000 | 1,000 | 1,000 | 131,0 ms |
| parafrazy | semantic | 1,000 | 1,000 | 1,000 | 1,000 | 132,6 ms |
| parafrazy | hybrid | 1,000 | 1,000 | 1,000 | 1,000 | 133,7 ms |
| parafrazy | keyword | 0,667 | 0,625 | 0,667 | 0,625 | 3,2 ms |

Zestaw nie pokazuje regresji hybrydy, ale wyniki semantyczne osiągają maksimum, więc nie da się na nim wykazać wymaganego zysku leksykalnego. Jest mały, syntetyczny i nie zastępuje pomiaru na rzeczywistej bazie. Czasy mają charakter orientacyjny i nie spełniają wymogu testu typowej dużej bazy.

Uruchomienie: `RAG_QUALITY_TEST=1 uv run pytest -s tests/test_independent_corpus.py`.

## Pełny czas zapytania na 2000 rekordów

Pomiar z 8 listopada 2026 obejmuje funkcję `search_results`, w tym kodowanie zapytania rzeczywistym modelem multilingual-e5-large, wyszukiwanie i fuzję. Zastosowano 2000 syntetycznych rekordów z wektorami pełnego wymiaru, pięć zapytań po cztery powtórzenia, rozgrzanie modelu i naprzemienną kolejność trybów.

| Tryb | Mediana | Średnia |
| --- | ---: | ---: |
| semantic | 139,59 ms | 139,90 ms |
| hybrid | 144,14 ms | 144,01 ms |
| Różnica par hybrid − semantic | 5,38 ms | 4,12 ms |

W tej próbie mediana różnic jest mniejsza od 50 ms. Wektory powstały z 40 rzeczywistych embeddingów tekstów i są wielokrotnie użyte; nie kodują indywidualnych identyfikatorów rekordów. Baza jest syntetyczna, nie służy do oceny jakości i nie dowodzi opóźnienia na reprezentatywnym korpusie. Pomiar nie obejmuje czasu budowy indeksu ani dodatkowej gałęzi n-gramowej.

Uruchomienie: `RAG_COST_TEST=1 uv run pytest -s tests/test_fts_cost.py::test_full_query_cost`.
