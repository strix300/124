"""Независимые вопросы по опубликованному поколению документа."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_resources
from app.bootstrap import Resources
from app.contracts import AnswerResponse, QuestionRequest

router = APIRouter(prefix="/documents", tags=["questions"])


@router.post("/{doc_id}/questions", response_model=AnswerResponse)
def ask_question(
    doc_id: UUID,
    body: QuestionRequest,
    resources: Annotated[Resources, Depends(get_resources)],
    diagnostics: bool = False,
) -> JSONResponse:
    """Возвращает ответ только после структурной и семантической проверки."""
    answer = resources.answers.answer(doc_id, body.question, diagnostics)
    excluded = {"diagnostics"} if answer.diagnostics is None else set()
    return JSONResponse(content=answer.model_dump(mode="json", exclude=excluded))
