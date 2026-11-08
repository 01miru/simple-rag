"""Opt-in evaluation on a separate synthetic operational corpus."""

import os

import pytest

from rag import index
from rag.evaluate import evaluate

# Documents and questions are separate from the samples used to tune RRF.
RECORDS = [
    (
        "Dział IT wydaje służbowe laptopy na podstawie protokołu LT-731. Sprzęt należy zwrócić w ostatnim dniu pracy.",
        "LT-731",
        "Kiedy oddaje się komputer służbowy?",
    ),
    (
        "Zgłoszenia uszkodzonych monitorów przyjmuje serwis przez formularz MN-284. Należy podać numer seryjny urządzenia.",
        "MN-284",
        "Jakie dane są potrzebne do zgłoszenia monitora?",
    ),
    (
        "Sala konferencyjna Róża jest rezerwowana w kalendarzu. Zmianę rezerwacji należy zgłosić recepcji przed południem.",
        "Róża",
        "Komu zgłosić zmianę rezerwacji sali?",
    ),
    (
        "Archiwum umów papierowych mieści się w szafie C-17. Klucz wydaje dział administracji po wpisaniu nazwiska do rejestru.",
        "C-17",
        "Gdzie przechowywane są papierowe umowy?",
    ),
    (
        "Wniosek KS-842 służy do zamówienia karty dostępu do parkingu. Decyzję podejmuje kierownik obiektu.",
        "KS-842",
        "Kto decyduje o karcie parkingowej?",
    ),
    (
        "Instrukcja PR-650 opisuje przekazywanie przesyłek kurierskich. Nieodebrane paczki trafiają do magazynu po dwóch dniach.",
        "PR-650",
        "Po jakim czasie nieodebrane paczki trafiają do magazynu?",
    ),
    (
        "Za aktualizację listy telefonów alarmowych odpowiada koordynator obiektu. Dokument oznaczono jako AL-918.",
        "AL-918",
        "Kto aktualizuje numery alarmowe?",
    ),
    (
        "W biurze Stokrotka rośliny podlewa firma zewnętrzna w każdy wtorek. Usterki instalacji wodnej zgłasza się administracji.",
        "Stokrotka",
        "Kiedy w biurze podlewane są rośliny?",
    ),
    (
        "Numer faktury zakupu należy podać w zgłoszeniu FK-367. Dział finansów sprawdza dane przed wypłatą należności.",
        "FK-367",
        "Kto sprawdza fakturę przed wypłatą?",
    ),
    (
        "Instrukcja BK-491 dotyczy odbioru dokumentów księgowych. Upoważniony pracownik potwierdza odbiór podpisem.",
        "BK-491",
        "Jak potwierdza się odbiór dokumentów księgowych?",
    ),
    (
        "Do zamówienia materiałów biurowych używa się formularza MB-526. Zamówienia są zbierane do końca każdego miesiąca.",
        "MB-526",
        "Do kiedy zbiera się zamówienia materiałów biurowych?",
    ),
    (
        "W systemie SZ-803 rejestruje się szkolenia pracowników. Potwierdzenie ukończenia jest dostępne u przełożonego.",
        "SZ-803",
        "Gdzie otrzymać potwierdzenie zakończenia szkolenia?",
    ),
]


@pytest.mark.skipif(
    os.environ.get("RAG_QUALITY_TEST") != "1", reason="requires real model"
)
def test_independent_corpus(tmp_path, capsys):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    cases = []
    for i, (document, code, question) in enumerate(RECORDS):
        filename = f"record{i:02}.md"
        (knowledge / filename).write_text(document, encoding="utf-8")
        for category, query in (("lexical", code), ("paraphrase", question)):
            cases.append(
                {
                    "query": query,
                    "source": f"knowledge/{filename}",
                    "category": category,
                }
            )
    index(tmp_path)
    report = evaluate(tmp_path, cases)
    with capsys.disabled():
        print("\nIndependent synthetic corpus:", report)
    assert report["all"]["hybrid"]["recall@3"] >= report["all"]["semantic"]["recall@3"]
