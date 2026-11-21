PY ?= python
export PYTHONPATH := src

.PHONY: install fullbio test clean all help

install:
	$(PY) -m pip install -e ".[dev]"

fullbio:
	$(PY) -m linearbaseline.cli fullbio --config configs/fullbio.json --out results

test:
	$(PY) -m pytest

all: test fullbio

clean:
	rm -rf results .pytest_cache

help:
	@echo "install  editable install with dev extras"
	@echo "fullbio  LOCO CV on the packaged K562 matrix (~6 s)"
	@echo "test     pytest"
	@echo "all      test + fullbio"
