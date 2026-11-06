"""Сохраняет реальные HTTP-ответы и проверяет их по исходному PDF."""

import argparse
import hashlib
import json
import platform
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx
from pydantic import JsonValue

from app.contracts import AnswerResponse, Submission
from app.core.config import Settings
from app.models import Page, TaskRecord
from app.processing.parsing import parse_pdf


def scenario_questions(scenario: str) -> list[tuple[str, str]]:
    """Выбирает вопросы регламента из существующего ключа, не загружая ключ в API."""
    if scenario == "contract":
        return [
            ("exact", "Какова неустойка за просрочку поставки по пункту 4.2?"),
            ("absent", "Какой номер банковского счёта указан для оплаты?"),
        ]
    key = Path(__file__).resolve().parents[1] / "tests/testdata/severny_kontur_qa.txt"
    questions = dict(re.findall(r"^Q(\d+)\. (.+)$", key.read_text(encoding="utf-8"), re.MULTILINE))
    return [("exact", questions["3"]), ("absent", questions["27"])]


def validate_answer(
    answer: AnswerResponse,
    name: str,
    scenario: str,
    pages: list[Page],
    submission: Submission,
    title: str,
) -> None:
    """Сверяет каждую цитату с оригиналом и проверяет условия выбранного сценария."""
    if answer.diagnostics is None:
        raise ValueError("Отсутствует запрошенная диагностика.")
    diagnostics = answer.diagnostics
    if diagnostics.timings_ms.get("grounding_verification", 0) <= 0:
        raise ValueError("Отсутствует измерение независимого verifier.")
    fragments = {item.fragment_id: item.chunk for item in diagnostics.context_fragments}
    for citation in answer.citations:
        chunk = fragments[citation.fragment_id]
        page = pages[citation.page - 1]
        start = citation.char_start - page.char_start
        end = citation.char_end - page.char_start
        if not (
            citation.doc_id == submission.doc_id == chunk.doc_id
            and citation.doc_title == title == chunk.doc_title
            and citation.page == chunk.page_from
            and citation.section == chunk.section
            and citation.text in chunk.text
            and 0 <= start < end <= len(page.text)
            and page.text[start:end] == citation.text
        ):
            raise ValueError("Цитата или её координаты не совпадают с исходным PDF.")
    if name == "absent":
        if answer.found or answer.citations:
            raise ValueError("Ожидался found=false с пустыми citations.")
        return
    if not answer.found or not answer.citations:
        raise ValueError("Ожидался точный ответ с цитатами.")
    if scenario == "contract":
        if "0,1%" not in answer.answer.replace("0.1%", "0,1%") or not any(
            citation.page == 2 and "0.1%" in citation.text for citation in answer.citations
        ):
            raise ValueError("Не подтверждена ставка 0,1% на странице 2 тестового договора.")
    else:
        expected = ("250000", "руководитель закупок", "операционный директор")
        answer_text = re.sub(r"(?<=\d)\s+(?=\d)", "", answer.answer.lower())
        quotes = " ".join(citation.text for citation in answer.citations if citation.page == 11)
        quoted_text = re.sub(r"(?<=\d)\s+(?=\d)", "", " ".join(quotes.lower().split()))
        if not all(value in answer_text and value in quoted_text for value in expected):
            raise ValueError("Не подтверждены 250 000 рублей и оба согласующих на странице 11.")


def smoke(
    pdf: Path,
    api_url: str,
    output: Path,
    timeout: float,
    warm: bool = False,
    scenario: str = "contract",
) -> None:
    """Записывает отчёт даже при сбое; успешный код выхода означает проверку обоих вопросов.

    Args:
        pdf: Оригинал регламента либо PDF, созданный create_example.py.
        api_url: Адрес запущенного API с теми же настройками, что в локальном .env.
        output: Путь отчёта; API-ключ в него не записывается.
        timeout: Таймаут HTTP-запроса и отдельный срок ожидания индексации, секунды.
        warm: Повторить точный вопрос после основных сценариев.
        scenario: handbook для регламента, contract для синтетического договора.
    """
    settings = Settings()
    results: dict[str, JsonValue] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "scenario": scenario,
        "source": {"path": str(pdf), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()},
        "client_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                package: version(package)
                for package in ("torch", "pypdf", "qdrant-client", "sentence-transformers")
            },
            "model_device": settings.model_device,
            "generation_model": settings.generation_model,
            "verification_model": settings.verification_model,
            "index_profile": settings.index_profile(),
        },
        "passed": False,
    }
    started = time.perf_counter()
    title = "Reviewer: Северный контур" if scenario == "handbook" else f"Smoke {time.time_ns()}"
    try:
        pages = parse_pdf(pdf, settings)
        results["source_pages"] = len(pages)
        with httpx.Client(
            base_url=api_url,
            timeout=timeout,
            trust_env=False,
            headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
        ) as client:

            def capture(name: str, response: httpx.Response) -> None:
                request = response.request
                headers = {"Authorization": "Bearer <RAG_API_KEY>"}
                if "content-type" in request.headers:
                    headers["Content-Type"] = request.headers["content-type"]
                body: JsonValue = None
                if headers.get("Content-Type") == "application/json":
                    body = json.loads(request.content)
                elif name == "submission":
                    body = {"file": str(pdf), "title": title}
                try:
                    payload = response.json()
                except ValueError:
                    payload = response.text
                results[name] = {
                    "request": {
                        "method": request.method,
                        "url": str(request.url),
                        "headers": headers,
                        "body": body,
                    },
                    "status_code": response.status_code,
                    "response": payload,
                }

            for name in ("live", "ready"):
                response = client.get(f"/health/{name}")
                capture(name, response)
                response.raise_for_status()
            with pdf.open("rb") as source:
                response = client.post(
                    "/documents",
                    files={"file": (pdf.name, source, "application/pdf")},
                    data={"title": title},
                )
            capture("submission", response)
            if response.status_code != 202:
                raise RuntimeError(f"Загрузка: HTTP {response.status_code}; см. отчёт.")
            submission = Submission.model_validate_json(response.content)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                response = client.get(submission.status_url)
                capture("indexing", response)
                response.raise_for_status()
                status = TaskRecord.model_validate_json(response.content)
                if status.status in {"succeeded", "failed"}:
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Индексация не завершилась в заданный срок.")
            results["indexing_wall_ms"] = (time.perf_counter() - started) * 1000
            if status.status != "succeeded":
                raise RuntimeError(f"Индексация завершилась ошибкой: {status.error}")
            response = client.get("/documents", params={"limit": 20})
            capture("catalog", response)
            response.raise_for_status()
            questions = scenario_questions(scenario)
            if warm:
                questions.append(("warm", questions[0][1]))
            failures: list[str] = []
            for name, question in questions:
                response = client.post(
                    f"/documents/{submission.doc_id}/questions",
                    params={"diagnostics": True},
                    json={"question": question},
                )
                capture(name, response)
                try:
                    response.raise_for_status()
                    answer = AnswerResponse.model_validate_json(response.content)
                    validate_answer(answer, name, scenario, pages, submission, title)
                except (ValueError, KeyError, IndexError, httpx.HTTPStatusError) as exc:
                    failures.append(f"{name}: {exc}")
            results["validation_errors"] = failures
            if failures:
                raise RuntimeError("Проверка ответов не пройдена; см. validation_errors в отчёте.")
            results["passed"] = True
    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        results["total_wall_ms"] = (time.perf_counter() - started) * 1000
        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path(".data/smoke-results.json"))
    parser.add_argument("--timeout", type=float, default=4000)
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--scenario", choices=("contract", "handbook"), default="contract")
    arguments = parser.parse_args()
    smoke(
        arguments.pdf,
        arguments.api_url,
        arguments.output,
        arguments.timeout,
        arguments.warm,
        arguments.scenario,
    )
