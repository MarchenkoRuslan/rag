FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app/ app/
COPY ui/ ui/

ARG INSTALL_EXTRAS=
RUN if [ -z "$INSTALL_EXTRAS" ]; then \
      pip install --no-cache-dir .; \
    else \
      pip install --no-cache-dir ".[$INSTALL_EXTRAS]"; \
    fi

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=base /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

RUN mkdir -p /app/data /app/storage /app/embeddings

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
