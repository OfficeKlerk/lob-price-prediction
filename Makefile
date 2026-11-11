# End-to-end pipeline. Run `make all` to reproduce every number in the README.

PYTHON ?= .venv/bin/python

.PHONY: all setup train export build bench plots test lint clean

all: train export build bench plots

setup:            ## Create the virtualenv and install the Python package
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

train:            ## Download FI-2010 (first run only) and train all models
	$(PYTHON) -m lob.train

export:           ## Export models to ONNX (float32 + int8) and check parity
	$(PYTHON) -m lob.export
	$(PYTHON) -m lob.export_fixed

build:            ## Build the C++ benchmark (downloads ONNX Runtime on first run)
	cmake -S cpp -B build
	cmake --build build

bench: build      ## Measure latency and throughput of every model
	$(PYTHON) -m lob.bench
	build/lob_fixed --model artifacts/cnn.fixed.bin --samples artifacts/samples.bin > results/fixed_point.json
	cat results/fixed_point.json

plots:            ## Render the README figures
	$(PYTHON) -m lob.plots

test: build       ## Run Python and C++ unit tests
	$(PYTHON) -m pytest -q
	ctest --test-dir build --output-on-failure

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

clean:
	rm -rf build artifacts
