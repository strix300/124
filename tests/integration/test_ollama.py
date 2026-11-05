"""Проверяет генерацию и отдельный verifier на настоящем локальном Ollama."""

import os
from time import perf_counter

import pytest

from app.core.config import Settings
from app.infrastructure.generation import AnswerGenerator
from app.infrastructure.local_chat import LocalChat
from app.models import Candidate, Claim, DraftAnswer, Evidence
from app.services.answering import generate_verified_answer
from app.services.verification import GroundingVerifier, validate_evidence

pytestmark = [
    pytest.mark.integration,
    pytest.mark.models,
    pytest.mark.skipif(
        os.getenv("RAG_RUN_MODEL_TESTS") != "1",
        reason="Требуется запущенный Ollama с локальными моделями и RAG_RUN_MODEL_TESTS=1",
    ),
]


def test_generation_passes_full_grounding_pipeline(context: list[Candidate]) -> None:
    settings = Settings()
    chat = LocalChat(settings)
    timings: dict[str, float] = {}

    answer = generate_verified_answer(
        AnswerGenerator(settings, chat),
        GroundingVerifier(settings, chat),
        "Какова неустойка за каждый день просрочки?",
        context,
        perf_counter() + settings.answer_deadline_seconds,
        timings,
    )

    assert answer.found is True
    assert answer.citations
    for citation in answer.citations:
        assert citation.fragment_id == context[0].fragment_id
        assert citation.text in context[0].chunk.text
        assert citation.page == context[0].chunk.page_from
        assert citation.doc_id == context[0].chunk.doc_id
    assert timings["generation"] > 0
    assert timings["grounding_verification"] > 0


def test_grounding_verifier_independently_accepts_supported_claim(
    context: list[Candidate],
) -> None:
    settings = Settings()
    statement = "Неустойка составляет 0,1% за каждый день просрочки."
    draft = DraftAnswer(
        found=True,
        claims=[
            Claim(
                text=statement,
                evidence=[Evidence(fragment_id=context[0].fragment_id, quote=statement)],
            )
        ],
    )
    validate_evidence(draft, context)

    GroundingVerifier(settings, LocalChat(settings)).verify(
        "Какова неустойка за каждый день просрочки?",
        draft,
        context,
        perf_counter() + settings.answer_deadline_seconds,
    )
