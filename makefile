.PHONY: lint test check

lint:
	ruff check --fix 
	ruff format

test:
	pytest

check: lint test