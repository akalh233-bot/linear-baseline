PY ?= python
export PYTHONPATH := src

.PHONY: install fullbio transfer test clean all help

install:
	$(PY) -m pip install -e ".[dev]"

fullbio:
	$(PY) -m linearbaseline.cli fullbio --config configs/fullbio.json --out results

transfer:
	$(PY) -m linearbaseline.cli transfer --config configs/transfer.json --out results

test:
	$(PY) -m pytest

all: test fullbio transfer

clean:
	rm -rf results .pytest_cache

help:
	@echo "install   editable install with dev extras"
	@echo "fullbio   LOCO CV on the packaged K562 matrix (~6 s)"
	@echo "transfer  zero-shot transfer to the five held-out cell types (~3 s)"
	@echo "test      pytest"
	@echo "all       test + fullbio + transfer"
