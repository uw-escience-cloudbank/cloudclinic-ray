.PHONY: lint fix ruff format type

lint: ruff type

fix:
	uv run --all-extras ruff check --fix src

ruff:
	uv run --all-extras ruff check src

format:
	uv run --all-extras ruff format src

type:
	uv run --all-extras pyrefly check
