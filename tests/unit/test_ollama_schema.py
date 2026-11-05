"""Проверяет совместимость грамматики Ollama без ослабления выходных моделей."""

import json
import sys
from copy import deepcopy
from time import perf_counter
from types import ModuleType
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import BaseModel, JsonValue, ValidationError

from app.core.config import Settings
from app.core.errors import InvalidGeneration
from app.core.timing import AnswerTrace
from app.infrastructure.generation import AnswerGenerator
from app.infrastructure.local_chat import LocalChat
from app.models import Candidate, DraftAnswer, GroundingVerdict
from app.services.verification import GroundingVerifier


class StubResponseError(Exception):
    """Заменяет только исключение отсутствующего SDK в изолированном тесте."""

    status_code: int = 500


@pytest.fixture
def chat_constructor(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Перехватывает параметры LangChain, не загружая SDK и не выполняя inference."""
    constructor = Mock()
    constructor.return_value.invoke.return_value.response_metadata = {}
    langchain_module = ModuleType("langchain_ollama")
    langchain_module.ChatOllama = constructor
    ollama_module = ModuleType("ollama")
    ollama_module.ResponseError = StubResponseError
    monkeypatch.setitem(sys.modules, "langchain_ollama", langchain_module)
    monkeypatch.setitem(sys.modules, "ollama", ollama_module)
    monkeypatch.setattr("app.infrastructure.local_chat.validate_local_model", Mock())
    return constructor


@pytest.mark.parametrize(
    "schema,payload",
    [
        (DraftAnswer, {"found": False, "claims": []}),
        (
            GroundingVerdict,
            {
                "sufficient_evidence": False,
                "answers_question": False,
                "claims": [],
            },
        ),
    ],
)
def test_sent_schema_only_removes_string_bounds(
    chat_constructor: Mock,
    schema: type[BaseModel],
    payload: dict[str, JsonValue],
) -> None:
    original = schema.model_json_schema()
    expected = deepcopy(original)
    if schema is DraftAnswer:
        for definition, field in (("Claim", "text"), ("Evidence", "quote")):
            del expected["$defs"][definition]["properties"][field]["minLength"]
            del expected["$defs"][definition]["properties"][field]["maxLength"]
    chat_constructor.return_value.invoke.return_value.content = json.dumps(payload)
    settings = Settings()

    result = LocalChat(settings).complete(
        settings.generation_model,
        schema,
        "Инструкция",
        "Вопрос",
        perf_counter() + 30,
    )

    sent_schema = chat_constructor.call_args.kwargs["format"]
    assert sent_schema == expected
    assert '"minLength"' not in json.dumps(sent_schema)
    assert '"maxLength"' not in json.dumps(sent_schema)
    assert schema.model_json_schema() == original
    assert result == schema.model_validate(payload)


@pytest.mark.parametrize(
    "field,length",
    [
        ("text", 0),
        ("text", 1501),
        ("quote", 0),
        ("quote", 4001),
    ],
)
def test_original_string_limits_reject_model_output(
    chat_constructor: Mock,
    field: str,
    length: int,
) -> None:
    evidence = {"fragment_id": str(uuid4()), "quote": "Точная цитата."}
    claim: dict[str, JsonValue] = {"text": "Утверждение.", "evidence": [evidence]}
    if field == "text":
        claim["text"] = "я" * length
    else:
        evidence["quote"] = "я" * length
    chat_constructor.return_value.invoke.return_value.content = json.dumps(
        {
            "found": True,
            "claims": [claim],
        }
    )
    settings = Settings()

    with pytest.raises(ValidationError) as error:
        LocalChat(settings).complete(
            settings.generation_model,
            DraftAnswer,
            "Инструкция",
            "Вопрос",
            perf_counter() + 30,
        )

    failure = error.value.errors()[0]
    assert failure["type"] == ("string_too_short" if length == 0 else "string_too_long")
    assert failure["loc"][-1] == field


def test_original_verifier_constraints_reject_model_output(chat_constructor: Mock) -> None:
    chat_constructor.return_value.invoke.return_value.content = json.dumps(
        {
            "sufficient_evidence": True,
            "answers_question": True,
            "claims": [{"claim_index": -1, "supported": True}],
        }
    )
    settings = Settings()

    with pytest.raises(ValidationError) as error:
        LocalChat(settings).complete(
            settings.verification_model,
            GroundingVerdict,
            "Инструкция",
            "Вопрос",
            perf_counter() + 30,
        )

    assert error.value.errors()[0]["loc"] == ("claims", 0, "claim_index")


@pytest.mark.parametrize("correction", [None, "Исправь идентификатор."])
def test_generation_constrains_ids_without_changing_validation(
    chat_constructor: Mock,
    context: list[Candidate],
    correction: str | None,
) -> None:
    settings = Settings()
    original = DraftAnswer.model_json_schema()
    chat_constructor.return_value.invoke.return_value.content = '{"found":false,"claims":[]}'
    generator = AnswerGenerator(settings, LocalChat(settings))

    generator.generate("Вопрос", context, perf_counter() + 30, correction)

    sent = chat_constructor.call_args.kwargs["format"]
    messages = chat_constructor.return_value.invoke.call_args.args[0]
    quote_context = json.loads(messages[1].content)["context"]
    assert sent["$defs"]["Claim"]["properties"]["evidence"]["items"]["enum"] == [
        quote["quote_id"] for fragment in quote_context for quote in fragment["quotes"]
    ]
    assert sent["properties"]["claims"]["maxItems"] == original["properties"]["claims"]["maxItems"]
    original_evidence = original["$defs"]["Claim"]["properties"]["evidence"]
    sent_evidence = sent["$defs"]["Claim"]["properties"]["evidence"]
    assert sent_evidence["minItems"] == original_evidence["minItems"]
    assert sent_evidence["maxItems"] == original_evidence["maxItems"]
    assert quote_context[0]["fragment_id"] == str(context[0].fragment_id)
    assert DraftAnswer.model_json_schema() == original
    assert '"maxLength"' not in json.dumps(sent)

    generator.generate("Вопрос", [], perf_counter() + 30)
    empty_schema = chat_constructor.call_args.kwargs["format"]
    assert empty_schema["properties"]["found"]["const"] is False
    assert empty_schema["properties"]["claims"]["maxItems"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        "unknown_id",
        "empty_text",
        "long_text",
        "no_evidence",
        "too_many_evidence",
        "too_many_claims",
        "false_with_claims",
    ],
)
def test_selection_still_enforces_original_answer_constraints(
    chat_constructor: Mock,
    context: list[Candidate],
    failure: str,
) -> None:
    claim = {"text": "Утверждение", "evidence": ["q0_0"]}
    payload = {"found": True, "claims": [claim]}
    if failure == "unknown_id":
        claim["evidence"] = ["foreign_quote"]
    elif failure == "empty_text":
        claim["text"] = ""
    elif failure == "long_text":
        claim["text"] = "я" * 1501
    elif failure == "no_evidence":
        claim["evidence"] = []
    elif failure == "too_many_evidence":
        claim["evidence"] *= 9
    elif failure == "too_many_claims":
        payload["claims"] *= 9
    else:
        payload["found"] = False
    chat_constructor.return_value.invoke.return_value.content = json.dumps(payload)
    settings = Settings()

    with pytest.raises(InvalidGeneration if failure == "unknown_id" else ValidationError):
        AnswerGenerator(settings, LocalChat(settings)).generate(
            "Вопрос",
            context,
            perf_counter() + 30,
        )


@pytest.mark.parametrize("log_outputs", [False, True])
def test_llm_observability_keeps_outputs_opt_in(
    chat_constructor: Mock,
    caplog: pytest.LogCaptureFixture,
    log_outputs: bool,
) -> None:
    caplog.set_level("INFO")
    output = '{"sufficient_evidence":false,"answers_question":false,"claims":[]}'
    chat_constructor.return_value.invoke.return_value.content = output
    chat_constructor.return_value.invoke.return_value.response_metadata = {
        "eval_count": 21,
        "done_reason": "stop",
    }
    settings = Settings(llm_log_outputs=log_outputs)
    trace = AnswerTrace(attempt=2)

    LocalChat(settings).complete(
        settings.verification_model,
        GroundingVerdict,
        "Инструкция",
        "Вопрос",
        perf_counter() + 30,
        trace=trace,
        phase="grounding_verification",
    )

    assert '"eval_count": 21' in caplog.text
    assert "attempt=2 phase=grounding_verification" in caplog.text
    assert ("ollama_output" in caplog.text) is log_outputs
    assert "attempt_2.grounding_verification.llm" in trace.timings
    assert "attempt_2.grounding_verification.pydantic" in trace.timings


@pytest.mark.parametrize("found", [False, True])
def test_verifier_schema_requires_exactly_the_actual_claims(
    chat_constructor: Mock,
    context: list[Candidate],
    found: bool,
) -> None:
    original = GroundingVerdict.model_json_schema()
    draft = DraftAnswer.model_validate(
        {
            "found": found,
            "claims": [
                {
                    "text": "Неустойка составляет 0,1%.",
                    "evidence": [
                        {
                            "fragment_id": str(context[0].fragment_id),
                            "quote": context[0].chunk.text,
                        }
                    ],
                }
            ]
            if found
            else [],
        }
    )
    chat_constructor.return_value.invoke.return_value.content = json.dumps(
        {
            "sufficient_evidence": found,
            "answers_question": found,
            "claims": [{"claim_index": 0, "supported": True}] if found else [],
        }
    )
    settings = Settings()

    GroundingVerifier(settings, LocalChat(settings)).verify(
        "Неустойка?",
        draft,
        context,
        perf_counter() + 30,
    )

    schema = chat_constructor.call_args.kwargs["format"]
    assert schema["properties"]["claims"]["minItems"] == len(draft.claims)
    assert schema["properties"]["claims"]["maxItems"] == len(draft.claims)
    if found:
        assert schema["$defs"]["ClaimVerdict"]["properties"]["claim_index"]["enum"] == [0]
    assert GroundingVerdict.model_json_schema() == original
