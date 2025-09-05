.PHONY: lint
lint:
	ruff format py --check
	ruff check py

.PHONY: format
format:
	ruff format py
	ruff check py --fix

.PHONY: typecheck
typecheck:
	mypy py