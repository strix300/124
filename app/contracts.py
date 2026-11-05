"""Публичные запросы, ответы и отдельный контракт диагностики."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, model_validator

from app.core.constants import MAX_DOCUMENT_PAGES, MAX_QUESTION_CHARS, MAX_QUOTE_CHARS
from app.models import Candidate, StrictModel, TaskStatus


class Submission(StrictModel):
    """Идентификаторы принятого документа и задания."""

    doc_id: UUID
    task_id: UUID
    status_url: str


class DocumentSummary(StrictModel):
    """Описывает загруженный документ независимо от срока жизни задания."""

    doc_id: UUID
    doc_title: str
    uploaded_at: datetime
    status: TaskStatus | Literal["ready", "status_expired"]
    task_id: UUID
    page_count: int | None
    chunk_count: int | None


class DocumentPage(StrictModel):
    """Возвращает ограниченную страницу каталога."""

    items: list[DocumentSummary]
    next_cursor: UUID | None


class QuestionRequest(StrictModel):
    """Содержит только текущий вопрос без истории и сессии."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Citation(StrictModel):
    """Содержит цитату и метаданные, восстановленные только из Qdrant payload."""

    fragment_id: UUID
    doc_id: UUID
    doc_title: str
    text: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)
    page: int = Field(ge=1, le=MAX_DOCUMENT_PAGES)
    section: str | None
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_span(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("Координаты цитаты не соответствуют её длине.")
        return self


class Diagnostics(StrictModel):
    """Отделяет измерения диагностических запросов от основного результата."""

    branch_measurement: Literal["separate_queries"] = "separate_queries"
    dense_candidates: int
    sparse_candidates: int
    fusion_candidates: int
    rerank_input_candidates: int
    rerank_output_candidates: int
    reranked_candidates: list[Candidate]
    context_fragments: list[Candidate]
    timings_ms: dict[str, float]


class AnswerResponse(StrictModel):
    """Предоставляет только проверенный ответ и исходные цитаты."""

    answer: str = Field(min_length=1)
    found: StrictBool
    citations: list[Citation]
    diagnostics: Diagnostics | None = None

    @model_validator(mode="after")
    def check_citations(self) -> Self:
        if self.found != bool(self.citations):
            raise ValueError("Ответ должен содержать цитаты только при found=true.")
        return self
