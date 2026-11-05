FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 LANGSMITH_TRACING=false TOKENIZERS_PARALLELISM=false
WORKDIR /service
ARG TORCH_REQUIREMENT="torch==2.14.1"
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN pip install --upgrade pip==26.3 setuptools==84.2.1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "${TORCH_REQUIREMENT}" --index-url "${TORCH_INDEX_URL}"
COPY pyproject.toml ./
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/pip pip install .
RUN useradd --create-home --uid 10001 rag && mkdir -p /data/documents /data/models \
    && chown -R rag:rag /data
USER rag
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
