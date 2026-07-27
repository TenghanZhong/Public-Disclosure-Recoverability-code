.PHONY: validate figures test smoke

validate:
	pytest tests

figures:
	python scripts/analysis/make_reported_figures.py

test:
	pytest tests

smoke: test figures
