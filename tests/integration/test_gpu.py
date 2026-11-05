"""Проверяет фактическое устройство локальных моделей и полный цикл коррекции."""

import json
import os
from collections.abc import Callable
from time import perf_counter

import httpx
import pytest

from app.core.config import Settings
from app.infrastructure.embeddings import LocalEmbeddings
from app.infrastructure.generation import AnswerGenerator
from app.infrastructure.local_chat import LocalChat
from app.infrastructure.reranking import LocalReranker
from app.models import Candidate
from app.services.answering import generate_verified_answer
from app.services.verification import GroundingVerifier

pytestmark = [
    pytest.mark.integration,
    pytest.mark.models,
    pytest.mark.skipif(
        os.getenv("RAG_RUN_GPU_TESTS") != "1",
        reason="Требуются CUDA, локальные веса и RAG_RUN_GPU_TESTS=1",
    ),
]


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_dense_and_reranker_execute_on_selected_device(
    context: list[Candidate],
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    import torch
    from sentence_transformers import CrossEncoder

    settings = Settings(model_device=device)
    observed_devices: list[str] = []
    original_predict = CrossEncoder.predict

    def predict(self: CrossEncoder, *args: object, **kwargs: object) -> object:
        observed_devices.append(next(self.model.parameters()).device.type)
        return original_predict(self, *args, **kwargs)

    monkeypatch.setattr(CrossEncoder, "predict", predict)
    if device == "cuda":
        assert torch.cuda.is_available()
    with LocalEmbeddings(settings) as embeddings:
        assert next(embeddings.dense._client.parameters()).device.type == device
        dense, sparse = embeddings.embed_question("Какова неустойка?")
        assert len(dense) == settings.dense_dimensions
        assert sparse.indices
    ranked = LocalReranker(settings).rerank("Какова неустойка?", context)
    assert ranked[0].rerank_score is not None
    assert observed_devices == [device]


def test_real_llm_correction_repeats_all_checks(
    context: list[Candidate],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Вносит одну ошибку цитаты после реального inference и требует реальную коррекцию."""
    from app.models import DraftAnswer

    settings = Settings()
    chat = LocalChat(settings)
    generator = AnswerGenerator(settings, chat)
    original_generate = generator.generate
    attempts = 0

    def generate(*args: object, **kwargs: object) -> DraftAnswer:
        nonlocal attempts
        draft = original_generate(*args, **kwargs)
        attempts += 1
        if attempts == 1:
            assert draft.found and draft.claims
            draft.claims[0].evidence[0].quote = "Несуществующая цитата для проверки коррекции."
        return draft

    monkeypatch.setattr(generator, "generate", generate)
    timings: dict[str, float] = {}
    answer = generate_verified_answer(
        generator,
        GroundingVerifier(settings, chat),
        "Какова неустойка за каждый день?",
        context,
        perf_counter() + settings.answer_deadline_seconds,
        timings,
    )
    assert attempts == 2
    assert answer.found and answer.citations
    assert "attempt_1.validation.exact_quotes" in timings
    assert "attempt_1.grounding_verification" not in timings
    for stage in (
        "validation.draft",
        "validation.evidence_ids",
        "validation.exact_quotes",
        "validation.metadata",
        "validation.grounding_verdict",
        "grounding_verification.pydantic",
        "grounding_verification.llm",
    ):
        assert f"attempt_2.{stage}" in timings
    assert all(citation.text in context[0].chunk.text for citation in answer.citations)
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(f"{settings.ollama_url}/api/ps")
        response.raise_for_status()
        loaded = next(
            item
            for item in response.json()["models"]
            if item["name"] == settings.verification_model
        )
    assert loaded["size_vram"] > 0
    assert loaded["size_vram"] == loaded["size"]
    assert loaded["context_length"] == settings.ollama_context_tokens
    record_property("timings_ms", json.dumps(timings))
