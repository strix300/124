"""Измеряет фактическое время этапов без предположений о производительности."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from app.core.errors import ApplicationError, InvalidGeneration


@dataclass
class AnswerTrace:
    """Связывает измерения и ошибки этапов с одним запросом и номером попытки."""

    timings: dict[str, float] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    attempt: int = 1
    failed_stage: str | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Измеряет этап, запоминая наиболее точное место отказа валидации."""
        with measure(self.timings, f"attempt_{self.attempt}.{name}"):
            try:
                yield
            except (ValidationError, InvalidGeneration):
                if self.failed_stage is None:
                    self.failed_stage = name
                raise


@contextmanager
def measure(timings: dict[str, float], stage: str) -> Iterator[None]:
    """Добавляет длительность этапа в миллисекундах, включая повторные попытки."""
    started = perf_counter()
    try:
        yield
    finally:
        timings[stage] = timings.get(stage, 0) + (perf_counter() - started) * 1000


def check_deadline(deadline: float) -> None:
    """Прерывает обработку между этапами при исчерпании общего бюджета."""
    if perf_counter() >= deadline:
        raise ApplicationError("answer_timeout", "Превышено время обработки вопроса.", 504)
