"""Небольшие исходные данные для проверки поведения без загрузки моделей."""

import re
from pathlib import Path
from uuid import uuid4

import pytest

from app.models import Candidate, Chunk


@pytest.fixture(scope="session")
def handbook_pdf() -> Path:
    """Использует предоставленный PDF без создания копии тестового документа."""
    path = Path(__file__).resolve().parents[1] / "severny_kontur_operational_handbook.pdf"
    if not path.is_file():
        pytest.fail(f"Отсутствует исходный PDF: {path}")
    return path


@pytest.fixture(scope="session")
def handbook_cases(handbook_pdf: Path) -> list[tuple[int, str, str, int | None]]:
    """Читает вопросы и эталон из файлов; при отсутствии списка использует вопросы ключа."""
    root = handbook_pdf.parent
    key = (root / "rag_test_answer_key.txt").read_text(encoding="utf-8-sig")
    entries = re.findall(
        r"^Q(\d+)\. (.+)\nОжидаемый ответ: (.+)\nОриентир: (.+)",
        key,
        re.MULTILINE,
    )
    assert entries, "Не удалось прочитать эталонные ответы."
    questions_path = root / "rag_test_questions.txt"
    questions = (
        dict(
            re.findall(
                r"^(?:Q)?(\d+)[.)]\s+(.+)",
                questions_path.read_text(encoding="utf-8-sig"),
                re.MULTILINE,
            )
        )
        if questions_path.is_file()
        else {number: question for number, question, _, _ in entries}
    )
    return [
        (
            int(number),
            questions[number],
            expected,
            int(page.removeprefix("стр. ")) if page.startswith("стр. ") else None,
        )
        for number, _, expected, page in entries
    ]


@pytest.fixture
def context() -> list[Candidate]:
    """Предоставляет неизменённый русский фрагмент с доверенными координатами."""
    text = "4.2 Штрафные санкции\nНеустойка составляет 0,1% за каждый день просрочки."
    return [
        Candidate(
            fragment_id=uuid4(),
            fusion_score=0.05,
            rerank_score=0.8,
            chunk=Chunk(
                doc_id=uuid4(),
                doc_title="Договор",
                generation_id=uuid4(),
                index_profile="profile",
                chunk_index=0,
                page_from=12,
                page_to=12,
                section="4.2 Штрафные санкции",
                text=text,
                char_start=100,
                char_end=100 + len(text),
            ),
        )
    ]
