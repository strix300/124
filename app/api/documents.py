"""Загрузка, список и явное удаление документов."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Form, Header, Query, Response, UploadFile

from app.api.dependencies import get_resources
from app.bootstrap import Resources
from app.contracts import DocumentPage, Submission

router = APIRouter(prefix="/documents", tags=["documents"])
Dependencies = Annotated[Resources, Depends(get_resources)]


@router.post("", status_code=202, response_model=Submission)
def upload_document(
    file: UploadFile,
    resources: Dependencies,
    title: Annotated[str | None, Form()] = None,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
) -> Submission:
    """Сохраняет PDF и запускает фоновую индексацию."""
    document_title = title if title is not None else file.filename or "Документ"
    return resources.documents.submit(file.file, document_title, idempotency_key)


@router.get("", response_model=DocumentPage)
def list_documents(
    resources: Dependencies,
    limit: Annotated[int, Query(ge=1, le=1000)] = 20,
    cursor: UUID | None = None,
) -> DocumentPage:
    """Возвращает страницу каталога, включая ещё не проиндексированные документы."""
    return resources.documents.list(limit, cursor)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: UUID, resources: Dependencies) -> Response:
    """Явно удаляет исходный файл и все поколения; повторное удаление безопасно."""
    resources.documents.delete(doc_id)
    return Response(status_code=204)
