"""Opt-in benchmark: many similar documents, several of them relevant per query."""

import os

import pytest

from rag import index
from rag.evaluate import evaluate

# Five differently worded reports per topic; a topic query should find all five.
TOPICS = {
    "Które zgłoszenia dotyczą zbyt wysokiej temperatury pracy sprzętu?": [
        "Po około dwóch godzinach ciągłej pracy obudowa robi się gorąca, a drukarka wstrzymuje zadania do czasu ostygnięcia.",
        "Czujnik termiczny głowicy zgłasza przekroczenie dopuszczalnej temperatury już przy średnim obciążeniu.",
        "Operator zauważył zapach nagrzanego plastiku; wentylator pracuje stale na najwyższych obrotach.",
        "Urządzenie przegrzewa się w trakcie drukowania długich serii etykiet i przechodzi w tryb ochronny.",
        "Latem w hali produkcyjnej sprzęt nagrzewa się tak mocno, że wyświetla ostrzeżenie termiczne.",
    ],
    "Przypadki blokowania się nośnika w mechanizmie podawania": [
        "Taśma z etykietami klinuje się w podajniku co kilkadziesiąt wydruków i trzeba ją ręcznie wyciągać.",
        "Papier zawija się wokół wałka dociskowego, przez co mechanizm staje w połowie zadania.",
        "Podczas pobierania materiału rolka blokuje się, a urządzenie zgłasza zacięcie.",
        "Etykiety odklejają się od podkładu wewnątrz toru prowadzenia i zatrzymują przesuw.",
        "Operator zgłasza, że materiał regularnie zacina się przy obcinaku.",
    ],
    "Zgłoszenia o utracie łączności z siecią": [
        "Drukarka co pewien czas znika z sieci i zadania wysyłane z systemu magazynowego nie docierają.",
        "Po zmianie konfiguracji przełącznika urządzenie nie uzyskuje adresu IP.",
        "Połączenie bezprzewodowe zrywa się kilka razy dziennie; pomaga dopiero ponowne uruchomienie modułu Wi-Fi.",
        "Serwer wydruku zgłasza brak odpowiedzi urządzenia, mimo że kabel Ethernet jest podłączony.",
        "Zadania zatrzymują się w kolejce, ponieważ sprzęt traci kontakt z siecią lokalną.",
    ],
    "Sprzęt dostarczony z uszkodzeniami powstałymi podczas przewozu": [
        "Karton dotarł zgnieciony z jednej strony, a po rozpakowaniu stwierdzono pękniętą pokrywę.",
        "Kurier dostarczył przesyłkę z wyraźnymi śladami upadku; wyłamany został zawias klapy.",
        "Podczas odbioru zauważono wgniecenie obudowy i zbitą szybkę wyświetlacza.",
        "Urządzenie zostało uszkodzone w transporcie: odłamany jest uchwyt rolki.",
        "Paleta przyjechała przechylona, a opakowanie ochronne było rozerwane; panel boczny jest pęknięty.",
    ],
    "Reklamacje słabej jakości wydruku": [
        "Na etykietach pojawiają się pionowe białe pasy, przez które kody kreskowe nie dają się zeskanować.",
        "Wydruk jest blady i nierównomierny, szczególnie przy prawej krawędzi.",
        "Tekst na naklejkach rozmazuje się i ściera po dotknięciu palcem.",
        "Kody kreskowe są nieczytelne dla skanerów, bo linie zlewają się ze sobą.",
        "Klient reklamuje jakość druku: znaki są poszarpane, a część punktów w ogóle się nie drukuje.",
    ],
    "Urządzenie samo się wyłącza albo uruchamia ponownie": [
        "Sprzęt wyłącza się samoczynnie kilka razy na zmianę, bez żadnego komunikatu.",
        "Po podłączeniu do gniazda drukarka startuje w pętli i nie dochodzi do ekranu głównego.",
        "Zasilacz wydaje wyraźne buczenie, a dioda zasilania migocze.",
        "Urządzenie nie reaguje na przycisk włączania, dopóki nie odłączy się i nie podłączy przewodu.",
        "W trakcie pracy następują nagłe restarty, po których tracone są ustawienia.",
    ],
}
# Every model appears in five reports, one per wording and on different topics.
MODELS = ["KX-210", "KX-215", "KX-220", "KX-225", "KX-230", "KX-235"]


def write_corpus(knowledge) -> list[dict]:
    """Write the reports and return evaluation cases with their expected sources."""
    by_model: dict[str, list[str]] = {model: [] for model in MODELS}
    cases = []
    for topic, (query, descriptions) in enumerate(TOPICS.items()):
        sources = []
        for variant, description in enumerate(descriptions):
            number = topic * len(descriptions) + variant
            model = MODELS[(topic + variant) % len(MODELS)]
            name = f"protokol{number:02}.md"
            (knowledge / name).write_text(
                f"# Protokół zgłoszenia serwisowego nr {1000 + number}\n\n"
                "## Dane zgłoszenia\n"
                f"Model urządzenia: {model}. Zgłoszenie przyjęto telefonicznie "
                "i potwierdzono wiadomością e-mail.\n\n"
                f"## Opis problemu\n{description}\n\n"
                "## Przebieg obsługi\n"
                "Zgłoszenie zarejestrowano w systemie serwisowym i przypisano do "
                "technika dyżurnego. Technik skontaktował się z osobą zgłaszającą, "
                "potwierdził dane urządzenia oraz ustalił termin wizyty. Podczas "
                "wizyty wykonano standardową procedurę diagnostyczną i zapisano "
                "wynik w karcie serwisowej.\n\n"
                "## Zalecenia\n"
                "Użytkownik powinien stosować się do instrukcji obsługi, wykonywać "
                "okresowe przeglądy i zgłaszać kolejne nieprawidłowości przez "
                "formularz serwisowy. Kierownik oddziału zatwierdza protokół "
                "i przekazuje go do archiwum.\n",
                encoding="utf-8",
            )
            sources.append(f"knowledge/{name}")
            by_model[model].append(f"knowledge/{name}")
        cases.append({"query": query, "sources": sources, "category": "topic"})
    cases += [
        {
            "query": f"zgłoszenia dotyczące modelu {model}",
            "sources": sources,
            "category": "identifier",
        }
        for model, sources in by_model.items()
    ]
    return cases


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="requires real model"
)
def test_similar_documents(tmp_path, capsys):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    cases = write_corpus(knowledge)
    index(tmp_path)
    report = evaluate(tmp_path, cases)
    with capsys.disabled():
        print("\nSimilar documents:", report)
    # Thresholds sit below the measured values recorded in tests/baseline.md.
    assert report["topic"]["hybrid"]["recall@10"] >= 0.8
    assert report["identifier"]["hybrid"]["recall@5"] >= 0.9
