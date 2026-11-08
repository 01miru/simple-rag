import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import chunks

DOCUMENT = Path(__file__).parent / "fixtures" / "podrecznik.md"
TOKENIZER = SimpleNamespace(
    encode=lambda text, **kwargs: [0, *re.findall(r"\w+|[^\w\s]", text), 1]
)


def test_long_document_chunks_preserve_content_and_sections():
    text = DOCUMENT.read_text(encoding="utf-8")
    parts = list(chunks(text, TOKENIZER, 80))
    assert len(parts) > 10
    assert all(len(TOKENIZER.encode(part["text"])) <= 80 for part in parts)
    # Whitespace between blocks may be normalized, but no non-whitespace is lost.
    assert re.sub(r"\s+", "", "".join(p["text"] for p in parts)) == re.sub(
        r"\s+", "", text
    )
    assert any("| Materiały biurowe | poniedziałek |" in p["text"] for p in parts)
    assert any("# To jest komentarz, nie nagłówek" in p["text"] for p in parts)


@pytest.mark.parametrize(
    ("phrase", "section"),
    [
        ("48 godzinach", "Dostęp i identyfikatory > Karty wejściowe"),
        ("2 do 8 stopni", "Magazyn i transport > Temperatura przechowywania"),
        ("numer 742", "Zakończenie pracy > Zgłoszenia awaryjne"),
    ],
)
def test_long_document_section(phrase, section):
    parts = chunks(DOCUMENT.read_text(encoding="utf-8"), TOKENIZER, 80)
    assert any(phrase in p["text"] and p["section"].endswith(section) for p in parts)


@pytest.mark.parametrize(
    "block",
    [
        "wyjątkowytekst " * 100,
        "```python\n" + "print('wynik')\n" * 100 + "```",
    ],
    ids=["paragraph", "code"],
)
def test_oversized_paragraph_and_code_are_split_without_losing_text(block):
    parts = list(chunks(block, TOKENIZER, 30))
    assert len(parts) > 1
    assert all(len(TOKENIZER.encode(part["text"])) <= 30 for part in parts)
    assert "".join(part["text"] for part in parts) == block.strip()
