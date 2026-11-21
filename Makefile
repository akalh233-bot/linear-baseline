PY ?= python
export PYTHONPATH := src

.PHONY: install synthetic train ablations test all clean

install:
	$(PY) -m pip install -e ".[dev]"

synthetic:
	$(PY) -m linearbaseline.cli synthetic --out data/synthetic.npz

train: synthetic
	$(PY) -m linearbaseline.cli train --config configs/synthetic.json --out results

ablations: synthetic
	$(PY) -m linearbaseline.cli ablations --config configs/synthetic.json --out results

test:
	$(PY) -m pytest

all: test train ablations

clean:
	rm -rf data results .pytest_cache
