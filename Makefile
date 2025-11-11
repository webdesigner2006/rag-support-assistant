.PHONY: setup fmt lint typecheck test run docker-build docker-up

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]
	pre-commit install

fmt:
	black src tests
	ruff --fix src tests

lint:
	ruff src tests

typecheck:
	mypy src

test:
	pytest -q

run:
	uvicorn rag_support.main:app --host $${HOST:-0.0.0.0} --port $${PORT:-8080} --reload

docker-build:
	docker build -t rag-support-assistant:latest .

docker-up:
	docker-compose up --build
