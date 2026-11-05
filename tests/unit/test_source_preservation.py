"""Проверяет страницы, смещения и неоднозначный повторяющийся текст."""

from uuid import uuid4

from app.core.config import Settings
from app.core.constants import PAGE_SEPARATOR
from app.models import DocumentRecord, Page
from app.processing.chunking import split_pages


def test_page_boundaries_sections_and_overlapping_offsets() -> None:
    first = "4.2 Штрафные санкции\n" + "Неустойка 0,1% за день. " * 8
    second = "Продолжение раздела. " * 8
    full = first + PAGE_SEPARATOR + second
    pages = [
        Page(number=1, text=first, char_start=0),
        Page(number=2, text=second, char_start=len(first) + 1),
    ]
    document = DocumentRecord(
        doc_id=uuid4(),
        doc_title="Договор",
        content_hash="hash",
        task_id=uuid4(),
        pending_generation=uuid4(),
        index_profile="profile",
    )
    chunks = split_pages(pages, document, Settings(chunk_tokens=50, chunk_overlap_tokens=10), len)
    assert {chunk.page_from for chunk in chunks} == {1, 2}
    assert all(chunk.page_from == chunk.page_to for chunk in chunks)
    assert all(full[chunk.char_start : chunk.char_end] == chunk.text for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.section == "4.2 Штрафные санкции" for chunk in chunks)
    assert chunks[-1].char_end == len(full)


def test_repeated_text_does_not_shift_offsets_backwards() -> None:
    text = "Повторяемая строка. " * 20
    document = DocumentRecord(
        doc_id=uuid4(),
        doc_title="Договор",
        content_hash="hash",
        task_id=uuid4(),
        pending_generation=uuid4(),
        index_profile="profile",
    )
    chunks = split_pages(
        [Page(number=1, text=text, char_start=0)],
        document,
        Settings(chunk_tokens=50, chunk_overlap_tokens=10),
        len,
    )
    assert chunks[-1].char_end == len(text)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.char_end - current.char_start <= 10
        assert not text[previous.char_end : current.char_start].strip()
