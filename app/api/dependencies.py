"""Передаёт lifespan-ресурсы HTTP-маршрутам."""

from typing import cast

from fastapi import Request

from app.bootstrap import Resources


def get_resources(request: Request) -> Resources:
    """Возвращает зависимости текущего экземпляра приложения."""
    return cast(Resources, request.app.state.resources)
