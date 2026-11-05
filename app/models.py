"""Типизированные данные между этапами индексации и формирования ответа."""

from datetime import UTC, datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool, model_validator

from app.core.constants import (
    MAX_ANSWER_CLAIMS,
    MAX_CLAIM_CHARS,
    MAX_CLAIM_EVIDENCE,
    MAX_DOCUMENT_PAGES,
    MAX_DOCUMENT_TITLE_CHARS,
    MAX_QUOTE_CHARS,
)

type TaskStatus = Literal["queued", "running", "retrying", "succeeded", "failed"]
type NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


class StrictModel(BaseModel):
    """Запрещает неописанные поля на границах приложения."""

    model_config = ConfigDict(extra="forbid")


class Page(StrictModel):
    """Сохраняет исходный текст страницы и его положение в документе."""

    number: int = Field(ge=1, le=MAX_DOCUMENT_PAGES)
    text: str
    char_start: int = Field(ge=0)


class Chunk(StrictModel):
    """Содержит неизменённый фрагмент и проверяемую привязку к источнику."""

    record_type: Literal["chunk"] = "chunk"
    doc_id: UUID
    doc_title: str = Field(min_length=1, max_length=MAX_DOCUMENT_TITLE_CHARS)
    generation_id: UUID
    index_profile: str
    chunk_index: int = Field(ge=0)
    page_from: int = Field(ge=1, le=MAX_DOCUMENT_PAGES)
    page_to: int = Field(ge=1, le=MAX_DOCUMENT_PAGES)
    section: str | None
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_span(self) -> Self:
        """Не допускает противоречивые координаты источника."""
        if self.page_from != self.page_to or self.char_end - self.char_start != len(self.text):
            raise ValueError("Некорректные границы фрагмента.")
        return self


class DocumentRecord(StrictModel):
    """Хранит каталог документа и ссылку на полностью опубликованное поколение."""

    record_type: Literal["document"] = "document"
    doc_id: UUID
    doc_title: str = Field(min_length=1, max_length=MAX_DOCUMENT_TITLE_CHARS)
    content_hash: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: UUID
    pending_generation: UUID
    active_generation: UUID | None = None
    index_profile: str
    active_profile: str | None = None
    page_count: int | None = Field(None, ge=1, le=MAX_DOCUMENT_PAGES)
    chunk_count: int | None = Field(None, ge=1)
    deleting: bool = False


class SparseVector(StrictModel):
    """Представляет разреженные веса без типов конкретного хранилища."""

    indices: list[NonNegativeInt]
    values: list[FiniteFloat]

    @model_validator(mode="after")
    def check_weights(self) -> Self:
        if len(self.indices) != len(self.values) or len(set(self.indices)) != len(self.indices):
            raise ValueError("Каждому уникальному индексу должен соответствовать один вес.")
        return self


class EmbeddedChunk(StrictModel):
    """Объединяет исходный фрагмент с двумя представлениями поиска."""

    chunk: Chunk
    dense: list[FiniteFloat] = Field(min_length=1)
    sparse: SparseVector


class Candidate(StrictModel):
    """Сохраняет источник и оценки после поиска и переранжирования."""

    fragment_id: UUID
    chunk: Chunk
    fusion_score: FiniteFloat
    rerank_score: FiniteFloat | None = None


class Evidence(StrictModel):
    """Ограничивает свидетельство идентификатором контекста и точной цитатой."""

    fragment_id: UUID
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)


class Claim(StrictModel):
    """Связывает каждое утверждение с конкретными свидетельствами."""

    text: str = Field(min_length=1, max_length=MAX_CLAIM_CHARS)
    evidence: list[Evidence] = Field(min_length=1, max_length=MAX_CLAIM_EVIDENCE)


class DraftAnswer(StrictModel):
    """Задаёт структурированный ответ генератора до проверки."""

    found: StrictBool
    claims: list[Claim] = Field(max_length=MAX_ANSWER_CLAIMS)

    @model_validator(mode="after")
    def check_found(self) -> Self:
        """Отделяет отсутствие сведений от ответа с утверждениями."""
        if self.found != bool(self.claims):
            raise ValueError("found должен соответствовать наличию утверждений.")
        return self


class ClaimVerdict(StrictModel):
    """Содержит семантическую оценку одного утверждения отдельным verifier."""

    claim_index: int = Field(ge=0, strict=True)
    supported: StrictBool


class GroundingVerdict(StrictModel):
    """Проверяет полноту ответа и поддержку каждого утверждения контекстом."""

    sufficient_evidence: StrictBool
    answers_question: StrictBool
    claims: list[ClaimVerdict]


class TaskError(StrictModel):
    """Предоставляет безопасное описание ошибки фонового задания."""

    code: str
    message: str


class TaskRecord(StrictModel):
    """Состояние задания, хранимое исключительно в Redis."""

    task_id: UUID
    doc_id: UUID
    generation_id: UUID
    status: TaskStatus = "queued"
    stage: str = "queued"
    attempt: NonNegativeInt = 0
    completed_chunks: NonNegativeInt = 0
    total_chunks: NonNegativeInt | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: TaskError | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
