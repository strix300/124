"""Проверяет отклонение страниц без текста и превышения лимита страниц."""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import ApplicationError


@pytest.mark.parametrize("pages,code", [(1, "page_without_text"), (101, "page_limit")])
def test_blank_or_oversized_pdf(tmp_path: Path, pages: int, code: str) -> None:
    pypdf = pytest.importorskip("pypdf", reason="Требуется parser из зависимостей проекта")
    from app.processing.parsing import parse_pdf

    path = tmp_path / "document.pdf"
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=300, height=300)
    with path.open("wb") as target:
        writer.write(target)
    with pytest.raises(ApplicationError) as error:
        parse_pdf(path, Settings())
    assert error.value.code == code


def test_hundred_text_pages_preserve_offsets(tmp_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    from app.processing.parsing import parse_pdf
    from scripts.create_example import create_example

    example = tmp_path / "example.pdf"
    create_example(example)
    writer = PdfWriter()
    reader = PdfReader(example)
    for _ in range(50):
        for page in reader.pages:
            writer.add_page(page)
    document = tmp_path / "hundred.pdf"
    writer.write(document)
    pages = parse_pdf(document, Settings())
    assert len(pages) == 100 and pages[-1].number == 100
    assert all(page.text.strip() for page in pages)
    assert pages[-1].char_start == sum(len(page.text) + 1 for page in pages[:-1])


def test_parser_timeout_is_an_application_error(tmp_path: Path) -> None:
    from app.processing.parsing import parse_pdf
    from scripts.create_example import create_example

    path = tmp_path / "document.pdf"
    create_example(path)
    with pytest.raises(ApplicationError) as error:
        parse_pdf(path, Settings(pdf_parse_timeout=0.000001))
    assert error.value.code == "pdf_timeout"
