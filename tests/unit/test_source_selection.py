"""Проверяет точные цитаты с переносами строк на предоставленном PDF."""

import json
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock

from pydantic import BaseModel

from app.core.config import Settings
from app.infrastructure.generation import AnswerGenerator
from app.models import Candidate
from app.processing.parsing import parse_pdf
from app.services.answering import build_response
from app.services.verification import validate_evidence


def test_handbook_selection_preserves_original_quotes_and_page_offsets(
    handbook_pdf: Path,
    handbook_cases: list[tuple[int, str, str, int | None]],
    context: list[Candidate],
) -> None:
    settings = Settings()
    pages = parse_pdf(handbook_pdf, settings)
    for _, question, _, page_number in handbook_cases[:4]:
        assert page_number is not None
        page = pages[page_number - 1]
        candidate = context[0].model_copy(deep=True)
        candidate.chunk.text = page.text
        candidate.chunk.page_from = candidate.chunk.page_to = page.number
        candidate.chunk.char_start = page.char_start
        candidate.chunk.char_end = page.char_start + len(page.text)
        chat = Mock()

        def select_quote(
            model: str,
            schema: type[BaseModel],
            system: str,
            prompt: str,
            deadline: float,
            **kwargs: object,
        ) -> BaseModel:
            quote = next(
                item
                for item in json.loads(prompt)["context"][0]["quotes"]
                if "\n" in item["text"] and len(item["text"]) > 100
            )
            return schema.model_validate(
                {
                    "found": True,
                    "claims": [
                        {
                            "text": quote["text"],
                            "evidence": [quote["quote_id"]],
                        }
                    ],
                }
            )

        chat.complete.side_effect = select_quote
        draft = AnswerGenerator(settings, chat).generate(
            question,
            [candidate],
            perf_counter() + 30,
        )
        validate_evidence(draft, [candidate])
        response = build_response(draft, [candidate])

        citation = response.citations[0]
        assert citation.page == page_number
        assert "\n" in citation.text
        start = citation.char_start - page.char_start
        end = citation.char_end - page.char_start
        assert page.text[start:end] == citation.text
