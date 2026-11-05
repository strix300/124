"""Чтение зарегистрированных фоновых заданий."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import get_resources
from app.bootstrap import Resources
from app.core.errors import ApplicationError
from app.models import TaskRecord

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskRecord)
def task_status(
    task_id: UUID,
    resources: Annotated[Resources, Depends(get_resources)],
) -> TaskRecord:
    """Не подменяет неизвестный или истёкший ID фиктивным статусом PENDING."""
    task = resources.state.get(task_id)
    if task is None:
        raise ApplicationError("task_not_found", "Задание не найдено или срок хранения истёк.", 404)
    return task
