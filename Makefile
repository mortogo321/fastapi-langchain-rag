.PHONY: lint test typecheck check build up down

lint:
	ruff check .
	ruff format --check .

test:
	pytest -q

typecheck:
	mypy app

check: lint typecheck test

build:
	docker build -t fastapi-langchain-rag .

up:
	docker compose up --build

down:
	docker compose down
