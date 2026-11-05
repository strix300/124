"""Разбивает текст средствами LangChain, не смешивая страницы и разделы."""

import re
from collections.abc import Callable, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.models import Chunk, DocumentRecord, Page

SECTION_HEADING = re.compile(r"(?m)^(?:\d+(?:\.\d+)*\.?[ \t]+)[^\n]{2,100}$")


def _sections(text: str, previous: str | None) -> Iterator[tuple[int, str, str | None]]:
    start = 0
    section = previous
    for match in SECTION_HEADING.finditer(text):
        if match.start() > start:
            yield start, text[start : match.start()], section
        start = match.start()
        section = match.group().strip()
    if start < len(text):
        yield start, text[start:], section


def split_pages(
    pages: list[Page],
    document: DocumentRecord,
    settings: Settings,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    """Сохраняет точные срезы источника и документные символьные координаты."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=count_tokens,
        strip_whitespace=False,
        keep_separator=True,
    )
    chunks: list[Chunk] = []
    section: str | None = None
    for page in pages:
        for section_start, section_text, current_section in _sections(page.text, section):
            section = current_section
            previous_start = -1
            previous_end = 0
            for text in splitter.split_text(section_text):
                if not text.strip():
                    continue
                start = section_text.find(text, previous_start + 1)
                while start >= 0 and start < previous_end:
                    overlap = section_text[start:previous_end]
                    if count_tokens(overlap) <= settings.chunk_overlap_tokens:
                        break
                    start = section_text.find(text, start + 1)
                if start < 0 or count_tokens(text) > settings.chunk_tokens:
                    raise ApplicationError(
                        "invalid_chunk",
                        "Не удалось сохранить границы фрагмента.",
                        500,
                    )
                previous_start = start
                previous_end = start + len(text)
                char_start = page.char_start + section_start + start
                chunks.append(
                    Chunk(
                        doc_id=document.doc_id,
                        doc_title=document.doc_title,
                        generation_id=document.pending_generation,
                        index_profile=document.index_profile,
                        chunk_index=len(chunks),
                        page_from=page.number,
                        page_to=page.number,
                        section=section,
                        text=text,
                        char_start=char_start,
                        char_end=char_start + len(text),
                    )
                )
    if not chunks:
        raise ApplicationError("empty_document", "Документ не содержит доступного текста.")
    return chunks
