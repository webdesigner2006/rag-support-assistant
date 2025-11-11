FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md /app/
RUN pip install -U pip && pip install -e .[dev]

COPY src /app/src
ENV PYTHONPATH=/app/src

EXPOSE 8080
CMD ["uvicorn", "rag_support.main:app", "--host", "0.0.0.0", "--port", "8080"]
