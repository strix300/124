"""Создаёт FastAPI-приложение и управляет ресурсами через lifespan."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from app.api import documents, questions, tasks
from app.api.access import AccessMiddleware
from app.api.upload_limit import UploadLimitMiddleware
from app.bootstrap import build_resources
from app.core.config import Settings
from app.core.errors import ApplicationError


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собирает API, оставляя блокирующие операции в рабочих потоках FastAPI."""
    config = settings or Settings()
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if not config.api_key.get_secret_value():
            raise RuntimeError("Настройте RAG_API_KEY: выполните rag init-api-key.")
        resources = await run_in_threadpool(build_resources, config)
        application.state.resources = resources
        try:
            await run_in_threadpool(resources.store.initialize)
            await run_in_threadpool(resources.store.verify_catalog_support)
            yield
        finally:
            await run_in_threadpool(resources.close)

    application = FastAPI(title="Document RAG", version="0.1.0", lifespan=lifespan)
    application.add_middleware(
        UploadLimitMiddleware,
        max_bytes=config.max_upload_bytes + config.multipart_overhead_bytes,
        json_max_bytes=config.max_json_bytes,
        body_timeout=config.request_body_timeout,
    )
    application.add_middleware(AccessMiddleware, settings=config)

    @application.exception_handler(ApplicationError)
    async def application_error(request: Request, exc: ApplicationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).error("unexpected_api_failure", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Внутренняя ошибка сервиса.",
                }
            },
        )

    @application.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        """Проверяет работоспособность процесса без обращения к моделям."""
        return {"status": "ok"}

    @application.get("/health/ready", tags=["health"])
    def ready(request: Request) -> dict[str, str]:
        """Проверяет хранилища; наличие весов проверяется отдельной командой doctor."""
        resources = request.app.state.resources
        try:
            resources.state.redis.ping()
        except RedisError as exc:
            raise ApplicationError("redis_unavailable", "Redis недоступен.", 503) from exc
        resources.store.list_documents(1, None)
        return {"status": "ok"}

    authentication = [Depends(HTTPBearer(auto_error=False))]
    application.include_router(documents.router, dependencies=authentication)
    application.include_router(tasks.router, dependencies=authentication)
    application.include_router(questions.router, dependencies=authentication)
    return application


app = create_app()
