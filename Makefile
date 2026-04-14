NAME = src
RUN  = uv run

all: run

install:
	uv sync

run:
	$(RUN) -m $(NAME)

debug:
	$(RUN) -m pdb -m $(NAME)

clean:
	rm -rf __pycache__ */__pycache__ .mypy_cache dist .ruff_cache

lint:
	$(RUN) flake8 $(NAME)
	$(RUN) mypy $(NAME) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

format:
	$(RUN) ruff format $(NAME)

.PHONY: install run debug clean lint lint-strict
