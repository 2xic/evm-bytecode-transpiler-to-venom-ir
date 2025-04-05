.PHONY: lint test check

lint:
	ruff check --fix 
	ruff format

test:
	python3.10 -m pytest

eval:
	python3.10 -m pip install -r requirements.txt
	python3.10 src/evals/eval.py 

check: lint test