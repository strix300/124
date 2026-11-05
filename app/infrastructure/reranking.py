"""Применяет отдельную локальную cross-encoder модель к fusion-кандидатам."""

import math

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.infrastructure.embeddings import model_directory, release_memory
from app.models import Candidate


class LocalReranker:
    """Переупорядочивает кандидатов, сохраняя источник и обе оценки."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rerank(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        """Не допускает неявного усечения текста при оценке релевантности."""
        if not candidates:
            return []
        import torch
        from sentence_transformers import CrossEncoder
        from sentence_transformers.base.modality_types import PairInput

        directory = model_directory(
            self.settings,
            "reranker",
            self.settings.reranker_model,
            self.settings.reranker_revision,
        )
        torch.set_num_threads(self.settings.inference_threads)
        model: CrossEncoder | None = None
        try:
            model = CrossEncoder(
                str(directory),
                device=self.settings.model_device,
                max_length=self.settings.reranker_max_tokens,
                local_files_only=True,
            )
            if self.settings.reranker_max_tokens > model.tokenizer.model_max_length:
                raise ApplicationError(
                    "reranker_token_limit",
                    "Лимит токенов превышает возможности реранкера.",
                    503,
                )
            selected = candidates[: self.settings.reranker_input_limit]
            pairs: list[PairInput] = []
            for candidate in selected:
                text = candidate.chunk.text
                if len(model.tokenizer.encode(question, text)) > self.settings.reranker_max_tokens:
                    raise ApplicationError(
                        "reranker_input_limit",
                        "Вопрос с фрагментом превышает лимит реранкера.",
                    )
                pairs.append((question, text))
            scores = model.predict(pairs, batch_size=self.settings.reranker_batch_size)
            scored_candidates = [
                (float(score), candidate) for candidate, score in zip(selected, scores, strict=True)
            ]
            if any(not math.isfinite(score) for score, _ in scored_candidates):
                raise ApplicationError(
                    "invalid_rerank_score", "Реранкер вернул неверную оценку.", 503
                )
            scored_candidates.sort(key=lambda scored: scored[0], reverse=True)
            return [
                candidate.model_copy(update={"rerank_score": score})
                for score, candidate in scored_candidates
            ]
        except (OSError, ValueError, RuntimeError) as exc:
            raise ApplicationError(
                "reranker_model_error",
                "Не удалось выполнить локальное переранжирование.",
                503,
            ) from exc
        finally:
            del model
            release_memory(self.settings.model_device)
