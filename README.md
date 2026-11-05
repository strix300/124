# Document RAG

Backend на FastAPI для вопросов по PDF до 100 страниц. Celery индексирует документ
через Redis; Qdrant хранит dense и BM25 векторы в одной коллекции. Поиск объединяется
в Qdrant, затем отдельная модель переранжирует кандидатов. Ответ и дословные цитаты
проходят структурную проверку и отдельный вызов LLM-проверяющего. Истории диалога нет.

## Запуск

Штатный Compose рассчитан на Linux или WSL2 с NVIDIA GPU, драйвером для CUDA 12.6
и настроенным NVIDIA Container Toolkit. Нужны Docker Compose v2.24+ и интернет
для первоначальной загрузки зависимостей и моделей. При обработке запросов модели
работают локально. Ресурсы и скорость следует измерить на своей машине.

Из корня репозитория:

```sh
cp .env.example .env
docker compose --profile app --profile inference build
docker run --rm -v "${PWD}:/workspace" -w /workspace document-rag:cuda rag init-api-key
docker compose --profile inference up -d redis qdrant ollama
docker compose --profile app run --rm api rag download-models
docker compose --profile app run --rm api rag init
docker compose --profile app --profile inference up -d --no-build
docker compose exec -T api rag doctor
```

`rag init-api-key` сохраняет секрет в `.env`. Для команд `curl` ниже экспортируйте
`RAG_API_KEY` с этим значением. В PowerShell вместо `cp` используйте `Copy-Item`.
Не перезаписывайте существующий `.env` при обновлении.

API: `http://localhost:8000`, Swagger: `http://localhost:8000/docs`.

## API

Все операции с документами и задачами требуют `Authorization: Bearer <RAG_API_KEY>`.
Один ключ даёт доступ ко всем документам установки.

| Метод и путь | Назначение |
|---|---|
| `POST /documents` | Multipart: `file`, необязательный `title`; возвращает `202` и идентификаторы |
| `GET /tasks/{task_id}` | Состояние фоновой индексации |
| `GET /documents?limit=20&cursor=UUID` | Каталог: `items`, `next_cursor` |
| `POST /documents/{doc_id}/questions?diagnostics=false` | JSON с текущим `question` |
| `DELETE /documents/{doc_id}` | Явное удаление документа и всех поколений, `204` |
| `GET /health/live` | Проверка процесса без ключа |
| `GET /health/ready` | Проверка Redis и Qdrant без ключа |

### Загрузка и статус

```sh
curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H 'Idempotency-Key: contract-001' \
  -F 'title=Договор поставки' -F 'file=@contract.pdf;type=application/pdf'
```

Пример контракта `202`:

```json
{
  "doc_id": "ad46dc3b-334b-4cb1-9807-a38cd2802812",
  "task_id": "04d57902-b97c-4e25-a06d-5ebdd0a0769f",
  "status_url": "/tasks/04d57902-b97c-4e25-a06d-5ebdd0a0769f"
}
```

```sh
curl -H "Authorization: Bearer $RAG_API_KEY" http://localhost:8000/tasks/TASK_ID
curl -H "Authorization: Bearer $RAG_API_KEY" 'http://localhost:8000/documents?limit=20'
```

Статусы задачи: `queued`, `running`, `retrying`, `succeeded`, `failed`. В ответе есть
этап, попытка, число записанных фрагментов и ошибка. У готового документа в каталоге
статус `ready`; у неготового без записи задачи — `status_expired`. Неизвестная или
истёкшая задача возвращает `404`. Терминальные записи по умолчанию живут 7 дней.

Одинаковые байты и точный заголовок переиспользуют документ. `Idempotency-Key`
связывает запрос с его идентификаторами; повтор ключа с другим содержимым даёт `409`.

### Вопрос с точным обозначением

UUID замените значением из загрузки. JSON ниже иллюстрирует контракт для условного
договора; это **не сохранённые результаты реального inference**.

```sh
curl -X POST 'http://localhost:8000/documents/DOC_ID/questions?diagnostics=true' \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Какова неустойка по пункту 4.2?"}'
```

Пример основного ответа; объект `diagnostics` для краткости опущен:

```json
{
  "answer": "Неустойка составляет 0,1% за каждый день просрочки.",
  "found": true,
  "citations": [{
    "fragment_id": "8de12178-010e-4fd5-a07c-f2a864679cb5",
    "doc_id": "ad46dc3b-334b-4cb1-9807-a38cd2802812",
    "doc_title": "Договор поставки",
    "text": "Неустойка составляет 0,1% за каждый день просрочки.",
    "page": 12,
    "section": "4.2 Штрафные санкции",
    "char_start": 2500,
    "char_end": 2551
  }]
}
```

Если сведений для вопроса «Каков банковский счёт поставщика?» в документе нет:

```json
{
  "answer": "В документе не найдено сведений, достаточных для ответа на вопрос.",
  "found": false,
  "citations": []
}
```

Диагностика добавляет числа dense/sparse/fusion кандидатов, количество кандидатов
до и после отбора реранкера, их оценки, фактический контекст и длительности этапов.
Основной поиск — один `query_points` с двумя `Prefetch` и `FusionQuery`.
Числа отдельных ветвей измеряются дополнительным `query_batch_points`:
`branch_measurement=separate_queries`. Это повторное измерение, а не счётчики
исходного prefetch; результаты измерения не формируют ответ.

Ошибки: `401` — ключ, `408` — чтение тела, `413` — размер загрузки, `415` — формат,
`422` — ввод/лимиты, `404` — ресурс, `409` — индекс не готов/конфликт,
`429` — сервис занят, `502` — ответ LLM не прошёл проверки, `503` — инфраструктура,
`504` — общий срок обработки вопроса истёк.

## Устройство и ограничения

`api` — HTTP и защита запросов; `services` — сценарии; `processing` — parsing
и chunking; `infrastructure` — хранилища и модели; `workers` — Celery.
Конфигурация находится в `app/core/config.py`, контракты — в `app/contracts.py`.

По умолчанию: dense `intfloat/multilingual-e5-small`, BM25 `Qdrant/bm25`,
реранкер `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, генератор и verifier
`qwen2.5:7b-instruct-q4_K_M`. Модели, лимиты, fusion `rrf`/`dbsf` и подключения
настраиваются переменными `RAG_*` из `.env.example`.

Чанки сохраняют страницы и точные символьные смещения извлечённого текста.
Поиск фильтруется по документу и опубликованному поколению. Частично записанный
индекс не публикуется; повтор задачи использует детерминированные ID.
Переиндексация запускается командой `rag reindex DOC_ID`.

OCR отсутствует. Проверка LLM вероятностная: дословная цитата
и второй вызов модели сами по себе не доказывают правильность ответа.
Исходники и старые поколения хранятся до явного удаления документа.
`docker compose down -v` удаляет volumes с данными.

Для production нужны воспроизводимая сборка, CI со стеком и моделями, оценка
качества на целевых документах, мониторинг, резервное копирование с проверкой
восстановления, дисковые квоты, TLS и разграничение доступа при нескольких клиентах.

## Проверки

Для нативной разработки нужен Python 3.12. После установки `pip install -e '.[dev]'`:

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy app
python -m pip check
```

Интеграционные тесты включаются флагами `RAG_RUN_INTEGRATION=1` (Qdrant),
`RAG_RUN_STACK_TESTS=1` (worker), `RAG_RUN_MODEL_TESTS=1` (Ollama),
`RAG_RUN_GPU_TESTS=1` (GPU) и `RAG_RUN_HANDBOOK_TESTS=1` (вопросы по регламенту).
По умолчанию они пропускаются. Для реального отчёта с ответами запустите работающий
стек и выполните:

```sh
python scripts/create_example.py .data/example.pdf
python scripts/smoke.py .data/example.pdf --output .data/smoke-results.json --warm
```

## Тестовые данные

Тестовые PDF-документы и соответствующие наборы вопросов с ответами находятся в `tests/testdata/`.

| Файл | Назначение |
|---|---|
| `computer_networks_part_1_pages_1-100.pdf` | Учебное пособие «Компьютерные сети» (страницы 1–100) |
| `computer_networks_qa_sequential.txt` | Вопросы и ответы для проверки по пособию «Компьютерные сети» |
| `severny_kontur_operational_handbook.pdf` | Операционный регламент «Северный контур» |
| `severny_kontur_qa.txt` | Вопросы и ответы для проверки по операционному регламенту |

Материалы предназначены для ручного и интеграционного тестирования загрузки документов, индексации, поиска релевантных фрагментов (retrieval) и формирования ответов с цитатами.
Для проверки загрузите соответствующий PDF через API, дождитесь завершения индексации и используйте файл с вопросами и ответами для проверки результатов.
Тестовые материалы не требуются для штатного запуска приложения.
