"""Проверяет доступ и ограничивает нагрузку до чтения тела и занятия рабочего потока."""

import secrets

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import Settings


class AccessMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.key = settings.api_key.get_secret_value().encode("utf-8")
        self.request_limit = settings.max_active_requests
        self.question_limit = settings.max_active_questions
        self.active_requests = 0
        self.active_questions = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/health/live":
            await self.app(scope, receive, send)
            return
        is_public = scope["path"] in {
            "/health/ready",
            "/docs",
            "/docs/oauth2-redirect",
            "/openapi.json",
            "/redoc",
        }
        headers = dict(scope["headers"])
        authorization = headers.get(b"authorization", b"")
        scheme, _, token = authorization.partition(b" ")
        if not is_public and not self.key:
            await self._reject(
                scope, receive, send, 503, "api_key_missing", "API-ключ не настроен."
            )
            return
        if not is_public and (
            scheme.lower() != b"bearer" or not secrets.compare_digest(token, self.key)
        ):
            await self._reject(scope, receive, send, 401, "unauthorized", "Требуется API-ключ.")
            return
        is_question = scope["method"] == "POST" and scope["path"].rstrip("/").endswith("/questions")
        if self.active_requests >= self.request_limit or (
            is_question and self.active_questions >= self.question_limit
        ):
            await self._reject(
                scope, receive, send, 429, "server_busy", "Сервис занят. Повторите запрос."
            )
            return
        self.active_requests += 1
        self.active_questions += int(is_question)
        try:
            await self.app(scope, receive, send)
        finally:
            self.active_requests -= 1
            self.active_questions -= int(is_question)

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        status: int,
        code: str,
        message: str,
    ) -> None:
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else {}
        if status == 429:
            headers["Retry-After"] = "1"
        await JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": message}},
            headers=headers,
        )(scope, receive, send)
