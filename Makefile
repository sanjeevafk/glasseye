PYTHON ?= python3.12
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
NPM ?= npm

.PHONY: setup prepare-data validate-data prepare-bfdd train inference benchmark-real demo backend frontend test lint e2e-demo clean-demo

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ./backend[dev]
	$(NPM) --prefix frontend install
	$(NPM) --prefix frontend exec playwright install chromium

prepare-data:
	$(PY) scripts/prepare_synthetic_dataset.py

validate-data:
	$(PY) scripts/validate_dataset.py

prepare-bfdd:
	$(PY) scripts/prepare_bfdd_binary_dataset.py

train: prepare-data validate-data
	$(PY) scripts/train_yolo.py --if-missing

inference: train
	$(PY) scripts/run_inference.py

benchmark-real: train
	$(PY) scripts/benchmark_real_data.py --dataset bfdd

demo: train
	$(PY) scripts/run_demo.py

backend:
	PYTHONPATH=backend $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

frontend:
	$(NPM) --prefix frontend run dev

test:
	PYTHONPATH=backend $(PY) -m pytest backend/tests
	$(NPM) --prefix frontend run test

lint:
	PYTHONPATH=backend $(PY) -m ruff check backend scripts
	$(NPM) --prefix frontend run typecheck

e2e-demo: demo
	$(NPM) --prefix frontend run test:e2e

clean-demo:
	rm -rf artifacts/demo
