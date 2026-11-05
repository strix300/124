"""Собирает конкретные зависимости приложения без скрытого service locator."""

from dataclasses import dataclass

from app.core.config import Settings
from app.infrastructure.files import FileStore
from app.infrastructure.generation import AnswerGenerator
from app.infrastructure.local_chat import LocalChat
from app.infrastructure.qdrant import QdrantStore
from app.infrastructure.task_state import TaskState
from app.services.answering import AnsweringService
from app.services.documents import DocumentService
from app.services.verification import GroundingVerifier
from app.workers.publishing import enqueue_indexing


@dataclass
class Resources:
    """Владеет соединениями в пределах одного процесса API."""

    settings: Settings
    store: QdrantStore
    state: TaskState
    documents: DocumentService
    answers: AnsweringService

    def close(self) -> None:
        """Закрывает ресурсы при остановке приложения."""
        self.store.close()
        self.state.close()


def build_resources(settings: Settings) -> Resources:
    """Передаёт сервисам явные зависимости; веса загружаются позже по необходимости."""
    store = QdrantStore(settings)
    state = TaskState(settings)
    chat = LocalChat(settings)
    return Resources(
        settings=settings,
        store=store,
        state=state,
        documents=DocumentService(settings, store, state, FileStore(settings), enqueue_indexing),
        answers=AnsweringService(
            settings,
            store,
            state,
            AnswerGenerator(settings, chat),
            GroundingVerifier(settings, chat),
        ),
    )
