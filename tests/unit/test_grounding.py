"""Проверяет важные отказы, цитаты и обязательную отдельную семантическую проверку."""

from time import perf_counter
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import InvalidGeneration, ServiceUnavailable
from app.core.timing import AnswerTrace
from app.models import Candidate, Claim, DraftAnswer, Evidence, GroundingVerdict
from app.services.answering import generate_verified_answer
from app.services.verification import GroundingVerifier, validate_evidence, verification_payload


def draft_for(context: list[Candidate]) -> DraftAnswer:
    """Создаёт один проверяемый тезис без выдуманных метаданных."""
    return DraftAnswer(
        found=True,
        claims=[
            Claim(
                text="Неустойка — 0,1% за день просрочки.",
                evidence=[
                    Evidence(
                        fragment_id=context[0].fragment_id,
                        quote="Неустойка составляет 0,1% за каждый день просрочки.",
                    )
                ],
            )
        ],
    )


def test_quote_and_context_membership(context: list[Candidate]) -> None:
    draft = draft_for(context)
    validate_evidence(draft, context)
    draft.claims[0].evidence[0].quote = "Неустойка составляет 10%."
    with pytest.raises(InvalidGeneration):
        validate_evidence(draft, context)
    draft = draft_for(context)
    draft.claims[0].evidence[0].fragment_id = uuid4()
    with pytest.raises(InvalidGeneration):
        validate_evidence(draft, context)


def test_trailing_newline_is_rejected_and_only_suggested_for_correction(
    context: list[Candidate],
) -> None:
    draft = draft_for(context)
    quote = draft.claims[0].evidence[0].quote
    draft.claims[0].evidence[0].quote += "\n"
    with pytest.raises(InvalidGeneration) as rejected:
        validate_evidence(draft, context)
    assert draft.claims[0].evidence[0].quote == quote + "\n"
    feedback = rejected.value.correction_feedback
    assert feedback is not None
    assert feedback["suggested_quote"] == quote
    assert quote in context[0].chunk.text


def test_correction_revalidates_and_uses_trusted_metadata(context: list[Candidate]) -> None:
    invalid = draft_for(context)
    invalid.claims[0].evidence[0].quote = "Выдуманная цитата"
    generator = Mock()
    generator.generate.side_effect = [invalid, draft_for(context)]
    verifier = Mock()
    timings: dict[str, float] = {}
    result = generate_verified_answer(
        generator,
        verifier,
        "Какая неустойка?",
        context,
        perf_counter() + 30,
        timings,
    )
    assert generator.generate.call_count == 2
    assert verifier.verify.call_count == 1
    assert result.citations[0].page == 12
    assert result.citations[0].section == context[0].chunk.section
    assert result.citations[0].doc_id == context[0].chunk.doc_id
    citation = result.citations[0]
    assert citation.char_start == 100 + context[0].chunk.text.index(citation.text)
    first_context = generator.generate.call_args_list[0].args[1]
    corrected_context = generator.generate.call_args_list[1].args[1]
    assert first_context is corrected_context
    feedback = generator.generate.call_args_list[1].args[3]
    assert feedback["rejected_quote"] == "Выдуманная цитата"
    assert feedback["source_text"] == context[0].chunk.text
    assert timings["generation"] > 0 and timings["correction"] > 0
    assert timings["grounding_verification"] > 0
    assert "attempt_1.validation.exact_quotes" in timings
    assert "attempt_1.grounding_verification" not in timings
    assert "attempt_2.validation.metadata" in timings
    assert "attempt_2.grounding_verification" in timings


def test_pdf_newlines_remain_mandatory_after_correction(
    context: list[Candidate],
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = context[0].chunk
    source.text = "Неустойка составляет\n0,1% за каждый день просрочки."
    source.char_end = source.char_start + len(source.text)
    generator = Mock()
    generator.generate.return_value = draft_for(context)
    verifier = Mock()
    trace = AnswerTrace()

    with pytest.raises(InvalidGeneration):
        generate_verified_answer(
            generator,
            verifier,
            "Неустойка?",
            context,
            perf_counter() + 30,
            trace.timings,
            trace=trace,
        )

    assert generator.generate.call_count == 2
    verifier.verify.assert_not_called()
    assert "grounding_verification" not in trace.timings
    assert trace.failed_stage == "validation.exact_quotes"
    assert f"request_id={trace.request_id} attempt=2 stage=validation.exact_quotes" in caplog.text
    assert "0,1%" not in caplog.text

    draft = draft_for(context)
    draft.claims[0].evidence[0].quote = source.text
    validate_evidence(draft, context)


def test_semantic_failure_has_exactly_one_correction(context: list[Candidate]) -> None:
    generator = Mock()
    generator.generate.return_value = draft_for(context)
    verifier = Mock()
    verifier.verify.side_effect = InvalidGeneration("Неверная сумма.")
    with pytest.raises(InvalidGeneration):
        generate_verified_answer(generator, verifier, "Сумма?", context, perf_counter() + 30, {})
    assert generator.generate.call_count == 2
    assert verifier.verify.call_count == 2


def test_unavailable_verifier_is_not_a_not_found_answer(context: list[Candidate]) -> None:
    generator = Mock()
    generator.generate.return_value = draft_for(context)
    verifier = Mock()
    verifier.verify.side_effect = ServiceUnavailable("Ollama")
    with pytest.raises(ServiceUnavailable):
        generate_verified_answer(generator, verifier, "Вопрос", context, perf_counter() + 30, {})
    assert generator.generate.call_count == 1


def test_not_found_also_requires_semantic_verification(context: list[Candidate]) -> None:
    generator = Mock()
    generator.generate.return_value = DraftAnswer(found=False, claims=[])
    verifier = Mock()
    result = generate_verified_answer(
        generator,
        verifier,
        "Погода?",
        context,
        perf_counter() + 30,
        {},
    )
    assert result.found is False and result.citations == []
    verifier.verify.assert_called_once()


@pytest.mark.parametrize(
    "verdict",
    [
        {"sufficient_evidence": True, "answers_question": True, "claims": []},
        {
            "sufficient_evidence": True,
            "answers_question": True,
            "claims": [{"claim_index": 0, "supported": False}],
        },
        {
            "sufficient_evidence": True,
            "answers_question": False,
            "claims": [{"claim_index": 0, "supported": True}],
        },
    ],
)
def test_verifier_checks_every_claim(context: list[Candidate], verdict: dict[str, object]) -> None:
    chat = Mock()
    chat.complete.return_value = GroundingVerdict.model_validate(verdict)
    verifier = GroundingVerifier(Settings(), chat)
    with pytest.raises(InvalidGeneration):
        verifier.verify("Вопрос", draft_for(context), context, perf_counter() + 30)


def test_invalid_json_after_correction_is_an_error(context: list[Candidate]) -> None:
    generator = Mock()
    try:
        DraftAnswer.model_validate_json('{"found":true,"claims":[]}')
    except ValidationError as exc:
        generator.generate.side_effect = exc
    with pytest.raises(InvalidGeneration):
        generate_verified_answer(generator, Mock(), "Вопрос", context, perf_counter() + 30, {})
    assert generator.generate.call_count == 2


def test_verification_preserves_overlapping_quotes_without_repeating_source(
    context: list[Candidate],
) -> None:
    draft = draft_for(context)
    fragment = context[0]
    draft.claims.append(
        Claim(
            text="Повторное утверждение",
            evidence=[
                Evidence(fragment_id=fragment.fragment_id, quote=fragment.chunk.text),
                Evidence(fragment_id=fragment.fragment_id, quote="0,1% за каждый день"),
            ],
        )
    )
    payload = verification_payload(draft, context)
    segments = payload["context"][0]["segments"]
    assert "".join(segment["text"] for segment in segments) == fragment.chunk.text
    texts = {segment["id"]: segment["text"] for segment in segments}
    for claim, encoded in zip(draft.claims, payload["draft"]["claims"], strict=True):
        for evidence, identifiers in zip(claim.evidence, encoded["evidence"], strict=True):
            assert "".join(texts[identifier] for identifier in identifiers) == evidence.quote


def test_repeated_evidence_fits_verification_context(context: list[Candidate]) -> None:
    import json

    fragment = context[0]
    fragment.chunk.text = "Исходные условия документа. " * 140
    fragment.chunk.char_end = fragment.chunk.char_start + len(fragment.chunk.text)
    evidence = Evidence(fragment_id=fragment.fragment_id, quote=fragment.chunk.text)
    draft = DraftAnswer(
        found=True,
        claims=[
            Claim(text="Условия приведены в документе.", evidence=[evidence] * 8) for _ in range(8)
        ],
    )
    payload = json.dumps(verification_payload(draft, context), ensure_ascii=False)
    assert len(payload.encode("utf-8")) < 10000
