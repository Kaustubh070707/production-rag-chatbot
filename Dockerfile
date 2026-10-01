FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
FROM python:3.12-slim
WORKDIR /app
# Patch OS CVEs Trivy flags (libpcre2, openssl family) ahead of the base
# rebuild: upgrade the exact affected packages, drop apt lists after.
RUN apt-get update \
 && apt-get install -y --no-install-recommends --only-upgrade libssl3t64 openssl openssl-provider-legacy libpcre2-8-0 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY docs/ ./docs/
COPY eval/questions.jsonl ./eval/questions.jsonl
COPY static/ ./static/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]