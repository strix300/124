"""Выполняет независимый поиск, переранжирование и полную проверку ответа."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError

from app.contracts import AnswerResponse, Citation, Diagnostics
from app.core.config import Settings
from app.core.constants import NOT_FOUND_ANSWER
from app.core.errors import ApplicationError, InvalidGeneration
from app.core.timing import AnswerTrace, check_deadline, measure
from app.infrastructure.embeddings import LocalEmbeddings
from app.infrastructure.generation import AnswerGenerator
from app.infrastructure.reranking import LocalReranker
from app.infrastructure.task_state import TaskState
from app.models import Candidate, DraftAnswer
from app.services.verification import GroundingVerifier, validate_evidence

if TYPE_CHECKING:
    from app.infrastructure.qdrant import QdrantStore

logger = logging.getLogger(__name__)


def build_response(draft: DraftAnswer, context: list[Candidate]) -> AnswerResponse:
    """Восстанавливает метаданные без доверия к номерам страниц, придуманным LLM."""
    if not draft.found:
        return AnswerResponse(answer=NOT_FOUND_ANSWER, found=False, citations=[])
    fragments = {candidate.fragment_id: candidate.chunk for candidate in context}
    citations: list[Citation] = []
    seen: set[tuple[UUID, str]] = set()
    for claim in draft.claims:
        for evidence in claim.evidence:
            key = evidence.fragment_id, evidence.quote
            if key in seen:
                continue
            seen.add(key)
            source = fragments[evidence.fragment_id]
            start = source.char_start + source.text.index(evidence.quote)
            citations.append(
                Citation(
                    fragment_id=evidence.fragment_id,
                    doc_id=source.doc_id,
                    doc_title=source.doc_title,
                    text=evidence.quote,
                    page=source.page_from,
                    section=source.section,
                    char_start=start,
                    char_end=start + len(evidence.quote),
                )
            )
    return AnswerResponse(
        answer=" ".join(claim.text for claim in draft.claims),
        found=True,
        citations=citations,
    )


def generate_verified_answer(
    generator: AnswerGenerator,
    verifier: GroundingVerifier,
    question: str,
    context: list[Candidate],
    deadline: float,
    timings: dict[str, float],
    *,
    trace: AnswerTrace | None = None,
) -> AnswerResponse:
    """Разрешает ровно одну коррекцию, повторяя после неё весь цикл проверок."""
    correction: str | dict[str, str] | None = None
    trace = trace if trace is not None else AnswerTrace(timings)
    for attempt in range(2):
        trace.attempt = attempt + 1
        trace.failed_stage = None
        check_deadline(deadline)
        phase = "generation" if attempt == 0 else "correction"
        try:
            with measure(timings, phase), trace.stage(phase):
                generated = generator.generate(question, context, deadline, correction, trace=trace)
            with measure(timings, "validation"):
                with trace.stage("validation.draft"):
                    draft = DraftAnswer.model_validate(generated.model_dump())
                validate_evidence(draft, context, trace=trace)
                with trace.stage("validation.metadata"):
                    response = build_response(draft, context)
            with measure(timings, "grounding_verification"), trace.stage("grounding_verification"):
                verifier.verify(question, draft, context, deadline, trace=trace)
            check_deadline(deadline)
            return response
        except (ValidationError, InvalidGeneration) as exc:
            reason = (
                exc.reason
                if isinstance(exc, InvalidGeneration)
                else json.dumps(
                    exc.errors(include_input=False, include_context=False, include_url=False),
                    ensure_ascii=False,
                )
            )
            logger.warning(
                "answer_rejected request_id=%s attempt=%s stage=%s reason=%s",
                trace.request_id,
                trace.attempt,
                trace.failed_stage,
                reason,
            )
            if attempt == 1:
                raise InvalidGeneration(
                    "Ответ не прошёл проверку после единственной коррекции.",
                ) from exc
            correction = (
                (exc.correction_feedback or exc.reason)
                if isinstance(exc, InvalidGeneration)
                else "Исправь структуру JSON по заданной схеме."
            )
    raise RuntimeError("Проверка завершилась без результата.")


class AnsweringService:
    """Оркестрирует pipeline без истории запросов и дополнительных источников."""

    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        state: TaskState,
        generator: AnswerGenerator,
        verifier: GroundingVerifier,
    ) -> None:
        self.settings = settings
        self.store = store
        self.state = state
        self.generator = generator
        self.verifier = verifier

    def answer(self, doc_id: UUID, question: str, diagnostics: bool) -> AnswerResponse:
        """Отправляет генератору только верхние фрагменты обязательного реранкера."""
        started = perf_counter()
        deadline = started + self.settings.answer_deadline_seconds
        timings: dict[str, float] = {}
        trace = AnswerTrace(timings)
        try:
            if not question.strip() or len(question) > self.settings.question_max_chars:
                raise ApplicationError(
                    "invalid_question",
                    "Вопрос пуст или превышает допустимый размер.",
                )
            document = self.store.get_document(doc_id)
            if document is None or document.deleting:
                raise ApplicationError("document_not_found", "Документ не найден.", 404)
            if document.active_generation is None:
                raise ApplicationError(
                    "document_not_ready",
                    "Индексация документа ещё не завершена.",
                    409,
                )
            if document.active_profile != self.settings.index_profile():
                raise ApplicationError(
                    "index_profile_changed",
                    "Требуется контролируемая переиндексация.",
                    409,
                )
            with self.state.inference_slot() as inference:
                with measure(timings, "query_embedding"):
                    with LocalEmbeddings(self.settings) as embeddings:
                        dense, sparse = embeddings.embed_question(question)
                check_deadline(deadline)
                inference.ensure_owned()
                with measure(timings, "retrieval"):
                    candidates, counts = self.store.retrieve(
                        doc_id,
                        document.active_generation,
                        dense,
                        sparse,
                        diagnostics,
                    )
                with measure(timings, "reranking"):
                    inference.ensure_owned()
                    ranked = LocalReranker(self.settings).rerank(question, candidates)
                selected = ranked[: self.settings.reranker_output_limit]
                context: list[Candidate] = []
                for candidate in selected:
                    if self.generator.context_fits(question, [*context, candidate]):
                        context.append(candidate)
                if candidates and not context:
                    raise ApplicationError(
                        "context_budget",
                        "Ни один фрагмент не помещается в контекст.",
                        422,
                    )
                inference.ensure_owned()
                response = generate_verified_answer(
                    self.generator,
                    self.verifier,
                    question,
                    context,
                    deadline,
                    timings,
                    trace=trace,
                )
                inference.ensure_owned()
            timings["total"] = (perf_counter() - started) * 1000
            if diagnostics:
                response.diagnostics = Diagnostics(
                    dense_candidates=counts["dense_candidates"],
                    sparse_candidates=counts["sparse_candidates"],
                    fusion_candidates=counts["fusion_candidates"],
                    rerank_input_candidates=len(ranked),
                    rerank_output_candidates=len(selected),
                    reranked_candidates=ranked,
                    context_fragments=context,
                    timings_ms=timings,
                )
            return response
        finally:
            timings["total"] = (perf_counter() - started) * 1000
            logger.info(
                "answer_timing request_id=%s doc_id=%s timings_ms=%s",
                trace.request_id,
                doc_id,
                json.dumps(timings),
            )
