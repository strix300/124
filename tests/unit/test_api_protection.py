"""Проверяет отказ до чтения тела и доступность служебных маршрутов под нагрузкой."""

import asyncio
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request

from app.api.access import AccessMiddleware
from app.api.upload_limit import UploadLimitMiddleware
from app.core.config import Settings

API_KEY = "test-key-" + "a" * 32
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def test_access_body_limit_and_admission() -> None:
    application = FastAPI()
    application.add_middleware(UploadLimitMiddleware, max_bytes=1024, json_max_bytes=32)
    application.add_middleware(
        AccessMiddleware,
        settings=Settings(api_key=API_KEY, max_active_questions=1),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    @application.post("/documents/test/questions")
    async def question(request: Request) -> dict[str, bool]:
        await request.json()
        started.set()
        await release.wait()
        return {"ok": True}

    @application.get("/health/live")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    async def forbidden_body() -> AsyncIterator[bytes]:
        raise AssertionError("Неавторизованное тело не должно читаться")
        yield b""

    async def oversized_body() -> AsyncIterator[bytes]:
        yield b'{"question":"'
        yield b"x" * 64
        yield b'"}'

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            path = "/documents/test/questions"
            assert (await client.post(path, content=forbidden_body())).status_code == 401
            assert (
                await client.post(path, content=oversized_body(), headers=HEADERS)
            ).status_code == 413
            first = asyncio.create_task(
                client.post(path, json={"question": "test"}, headers=HEADERS)
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            try:
                assert (await client.post(path, json={}, headers=HEADERS)).status_code == 429
                assert (await client.get("/health/live")).status_code == 200
            finally:
                release.set()
                assert (await first).status_code == 200
            assert (await client.post(path, json={}, headers=HEADERS)).status_code == 200

    asyncio.run(scenario())
