"""Хранит задания в Redis и ограничивает конкурентную работу локальных моделей."""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from filelock import FileLock, Timeout
from redis import Redis
from redis.exceptions import LockError, RedisError

from app.core.config import Settings
from app.core.constants import TERMINAL_STATUSES
from app.core.errors import ApplicationError, ServiceUnavailable
from app.models import TaskRecord

SAVE_TASK = """
local old = redis.call('GET', KEYS[1])
if old then
    local previous = cjson.decode(old)
    if previous.status == 'succeeded' or previous.status == 'failed' then return 0 end
    if previous.attempt > tonumber(ARGV[4]) then return 0 end
end
if ARGV[3] == '1' then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[5])
    redis.call('ZREM', KEYS[2], ARGV[6])
else
    redis.call('SET', KEYS[1], ARGV[1])
    redis.call('ZADD', KEYS[2], ARGV[2], ARGV[6])
end
return 1
"""

REGISTER_TASK = """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('ZADD', KEYS[2], ARGV[2], ARGV[3])
return 1
"""


class Lease:
    """Продлевает блокировку и явно отклоняет работу после потери владения."""

    def __init__(self, redis: Redis, key: str, seconds: int, wait: float = 0) -> None:
        self.lock = redis.lock(key, timeout=seconds, blocking_timeout=wait, thread_local=False)
        self.seconds = seconds
        self.stopped = threading.Event()
        self.lost = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "Lease":
        try:
            if not self.lock.acquire():
                raise ServiceUnavailable("занятая очередь inference или документ")
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc
        self.thread = threading.Thread(target=self._renew, daemon=True)
        self.thread.start()
        return self

    def _renew(self) -> None:
        while not self.stopped.wait(self.seconds / 3):
            try:
                self.lock.extend(self.seconds, replace_ttl=True)
            except (RedisError, LockError):
                self.lost.set()
                return

    def ensure_owned(self) -> None:
        """Не позволяет публиковать результат при потере Redis или блокировки."""
        try:
            if self.lost.is_set() or not self.lock.owned():
                raise ServiceUnavailable("блокировка операции")
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=self.seconds)
        try:
            if self.lock.owned():
                self.lock.release()
            elif exc is None:
                raise ServiceUnavailable("блокировка операции")
        except RedisError as error:
            if exc is None:
                raise ServiceUnavailable("Redis") from error


class TaskState:
    """Предоставляет известные задания; активные записи не имеют TTL."""

    def __init__(self, settings: Settings, redis: Redis | None = None) -> None:
        self.settings = settings
        self.redis = redis or Redis.from_url(
            settings.state_redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_timeout,
            socket_connect_timeout=settings.redis_timeout,
        )
        self.prefix = settings.redis_prefix

    def close(self) -> None:
        """Освобождает соединения Redis."""
        self.redis.close()

    def _key(self, task_id: UUID) -> str:
        return f"{self.prefix}:task:{task_id}"

    def register(self, task: TaskRecord) -> None:
        """Регистрирует задание до отправки в брокер, сохраняя существующее состояние."""
        try:
            self.redis.eval(
                REGISTER_TASK,
                2,
                self._key(task.task_id),
                f"{self.prefix}:active",
                task.model_dump_json(),
                str(task.updated_at.timestamp()),
                str(task.task_id),
            )
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc

    def save(self, task: TaskRecord) -> None:
        """Атомарно обновляет статус и включает TTL только для терминальных состояний."""
        task.updated_at = datetime.now(UTC)
        try:
            self.redis.eval(
                SAVE_TASK,
                2,
                self._key(task.task_id),
                f"{self.prefix}:active",
                task.model_dump_json(),
                str(task.updated_at.timestamp()),
                str(int(task.status in TERMINAL_STATUSES)),
                str(task.attempt),
                str(self.settings.terminal_task_ttl),
                str(task.task_id),
            )
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc

    def get(self, task_id: UUID) -> TaskRecord | None:
        """Отличает отсутствующее задание от реального ожидающего задания."""
        try:
            record = cast(str | None, self.redis.get(self._key(task_id)))
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc
        return TaskRecord.model_validate_json(record) if record is not None else None

    def get_many(self, task_ids: list[UUID]) -> list[TaskRecord | None]:
        """Получает состояния страницы каталога одним обращением к Redis."""
        if not task_ids:
            return []
        try:
            records = cast(
                list[str | None],
                self.redis.mget([self._key(task_id) for task_id in task_ids]),
            )
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc
        return [TaskRecord.model_validate_json(record) if record else None for record in records]

    def stale_tasks(self) -> list[TaskRecord]:
        """Выбирает ограниченную порцию заданий для восстановления после сбоев."""
        try:
            identifiers = self.redis.zrangebyscore(
                f"{self.prefix}:active",
                "-inf",
                time.time() - self.settings.stale_task_seconds,
                start=0,
                num=self.settings.reconciliation_batch_size,
            )
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc
        records = self.get_many([UUID(value) for value in cast(list[str], identifiers)])
        return [record for record in records if record]

    def document_lease(self, doc_id: UUID, wait: float = 0) -> Lease:
        """Сериализует изменение и удаление одного документа."""
        return Lease(
            self.redis,
            f"{self.prefix}:document:{doc_id}",
            self.settings.lock_lease_seconds,
            wait,
        )

    @contextmanager
    def inference_slot(self, wait_seconds: float | None = None) -> Iterator[Lease]:
        """Ограничивает суммарную память inference во всех процессах приложения."""
        deadline = time.monotonic() + (
            self.settings.inference_wait_seconds if wait_seconds is None else wait_seconds
        )
        while True:
            for index in range(self.settings.inference_slots):
                lease = Lease(
                    self.redis,
                    f"{self.prefix}:inference:{index}",
                    self.settings.lock_lease_seconds,
                )
                stack = ExitStack()
                try:
                    directory = self.settings.storage_dir / ".locks"
                    directory.mkdir(parents=True, exist_ok=True)
                    stack.enter_context(FileLock(directory / f"inference-{index}.lock", timeout=0))
                    stack.enter_context(lease)
                except (ServiceUnavailable, Timeout):
                    stack.close()
                    continue
                with stack:
                    yield lease
                    lease.ensure_owned()
                return
            if time.monotonic() >= deadline:
                raise ServiceUnavailable("очередь локальных моделей")
            time.sleep(self.settings.lock_poll_seconds)

    def bind_request(
        self,
        key: str,
        signature: str,
        doc_id: UUID,
        task_id: UUID,
    ) -> tuple[UUID, UUID]:
        """Связывает HTTP-ключ с запросом независимо от дедупликации документов."""
        record = json.dumps(
            {
                "signature": signature,
                "doc_id": str(doc_id),
                "task_id": str(task_id),
            }
        )
        redis_key = f"{self.prefix}:request:{key}"
        try:
            self.redis.set(redis_key, record, ex=self.settings.idempotency_ttl, nx=True)
            existing = cast(str | None, self.redis.get(redis_key))
        except RedisError as exc:
            raise ServiceUnavailable("Redis") from exc
        if existing is None:
            raise ServiceUnavailable("Redis")
        binding = json.loads(existing)
        if binding["signature"] != signature:
            raise ApplicationError(
                "idempotency_conflict",
                "Ключ уже использован для другого запроса.",
                409,
            )
        return UUID(binding["doc_id"]), UUID(binding["task_id"])
