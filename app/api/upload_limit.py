"""Ограничивает размер и время чтения тела, освобождая multipart spool при отказе."""

import asyncio
from time import monotonic

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class UploadLimitMiddleware:
    """Проверяет размер тела, включая запросы без Content-Length."""

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        json_max_bytes: int = 128 * 1024,
        body_timeout: float = 60,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.json_max_bytes = json_max_bytes
        self.body_timeout = body_timeout

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Не допускает неограниченное заполнение временного каталога загрузкой."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_upload = scope["method"] == "POST" and scope["path"].rstrip("/") == "/documents"
        limit = self.max_bytes if is_upload else self.json_max_bytes
        deadline = monotonic() + self.body_timeout
        received = 0
        rejected = False
        response = JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "upload_too_large",
                    "message": "Превышен размер загрузки.",
                }
            },
        )
        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared < 0 or declared > limit:
                await response(scope, receive, send)
                return

        async def limited_receive() -> Message:
            nonlocal received, rejected, response
            try:
                async with asyncio.timeout(max(0, deadline - monotonic())):
                    message = await receive()
            except TimeoutError as exc:
                rejected = True
                response = JSONResponse(
                    status_code=408,
                    content={
                        "error": {
                            "code": "body_timeout",
                            "message": "Истекло время чтения запроса.",
                        }
                    },
                )
                raise MultiPartException("Истекло время чтения запроса.") from exc
            received += len(message.get("body", b""))
            if received > limit:
                rejected = True
                raise MultiPartException("Превышен размер загрузки.")
            return message

        async def limited_send(message: Message) -> None:
            if not rejected:
                await send(message)
            elif message["type"] == "http.response.start":
                await response(scope, receive, send)

        try:
            await self.app(scope, limited_receive, limited_send)
        except MultiPartException:
            if not rejected:
                raise
            await response(scope, receive, send)
