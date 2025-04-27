.PHONY: lint test check

lint:
	ruff check --fix --unsafe-fixes
	ruff format 

test:
	python3.10 -m pytest

eval:
	python3.10 src/evals/eval.py 

test_v2:
	# Only test the new engine design
	pytest -k "v2"

coverage_v2:
	# Only test coverage the new engine design
	pytest -k "v2" --cov=v2 --cov-report=html:coverage_report
	cd coverage_report && python3 -m http.server 9000

check: lint test