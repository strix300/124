"""Записывает реальные результаты и измерения полного локального pipeline."""

import argparse
import json
import time
from pathlib import Path

import httpx
from pydantic import JsonValue

from app.contracts import AnswerResponse, Submission
from app.core.config import Settings
from app.models import TaskRecord


def smoke(pdf: Path, api_url: str, output: Path, timeout: float, warm: bool = False) -> None:
    """Проверяет точное обозначение, отсутствие ответа и реальные ссылки на страницы."""
    results: dict[str, JsonValue] = {}
    answers: dict[str, AnswerResponse] = {}
    started = time.perf_counter()
    with httpx.Client(
        base_url=api_url,
        timeout=timeout,
        trust_env=False,
        headers={"Authorization": f"Bearer {Settings().api_key.get_secret_value()}"},
    ) as client:
        with pdf.open("rb") as source:
            response = client.post(
                "/documents",
                files={"file": (pdf.name, source, "application/pdf")},
                data={"title": f"Smoke {time.time_ns()}"},
            )
        response.raise_for_status()
        submission = Submission.model_validate_json(response.content)
        results["submission"] = submission.model_dump(mode="json")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = client.get(submission.status_url)
            response.raise_for_status()
            status = TaskRecord.model_validate_json(response.content)
            if status.status in {"succeeded", "failed"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError("Индексация не завершилась в заданный срок.")
        results["indexing"] = status.model_dump(mode="json")
        results["indexing_wall_ms"] = (time.perf_counter() - started) * 1000
        if status.status != "succeeded":
            raise RuntimeError(f"Индексация завершилась ошибкой: {status.error}")
        questions = [
            ("exact", "Какова неустойка за просрочку поставки по пункту 4.2?"),
            ("absent", "Какой номер банковского счёта указан для оплаты?"),
        ]
        if warm:
            questions.append(("warm", questions[0][1]))
        for name, question in questions:
            request_started = time.perf_counter()
            response = client.post(
                f"/documents/{submission.doc_id}/questions",
                params={"diagnostics": True},
                json={"question": question},
            )
            results[name] = {
                "question": question,
                "status_code": response.status_code,
                "response": response.json(),
                "wall_ms": (time.perf_counter() - request_started) * 1000,
            }
            if response.status_code == 200:
                answers[name] = AnswerResponse.model_validate_json(response.content)
        output.parent.mkdir(parents=True, exist_ok=True)
        results["total_wall_ms"] = (time.perf_counter() - started) * 1000
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        exact = answers.get("exact")
        absent = answers.get("absent")
        if exact is None or not exact.found or not exact.citations:
            raise RuntimeError("Вопрос по точному обозначению не прошёл проверку; см. отчёт.")
        if "0,1%" not in exact.answer.replace("0.1%", "0,1%") or not any(
            citation.page == 2 and "0.1%" in citation.text for citation in exact.citations
        ):
            raise RuntimeError(
                "Ответ не подтверждает ставку 0,1% на странице 2 тестового договора."
            )
        if absent is None or absent.found or absent.citations:
            raise RuntimeError("Отказ при отсутствии сведений не прошёл проверку; см. отчёт.")
        if warm and ("warm" not in answers or not answers["warm"].found):
            raise RuntimeError("Повторный запрос не прошёл проверку; см. отчёт.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path(".data/smoke-results.json"))
    parser.add_argument("--timeout", type=float, default=4000)
    parser.add_argument("--warm", action="store_true")
    arguments = parser.parse_args()
    smoke(arguments.pdf, arguments.api_url, arguments.output, arguments.timeout, arguments.warm)
