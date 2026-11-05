"""Запрещает cloud aliases до отправки контекста локальному inference runtime."""

import httpx
import pytest

from app.core.errors import ApplicationError
from app.infrastructure.local_chat import validate_local_model


@pytest.mark.parametrize("remote", [False, True])
def test_model_alias_must_resolve_to_local_weights(remote: bool) -> None:
    def show(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        metadata: dict[str, object] = {"model_info": {"general.architecture": "qwen3"}}
        if remote:
            metadata["remote_host"] = "https://ollama.com"
            metadata["remote_model"] = "remote-model"
        return httpx.Response(200, json=metadata)

    with httpx.Client(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(show),
    ) as client:
        if remote:
            with pytest.raises(ApplicationError) as error:
                validate_local_model(client, "local-looking-alias")
            assert error.value.code == "local_model_required"
        else:
            validate_local_model(client, "qwen3:4b")
