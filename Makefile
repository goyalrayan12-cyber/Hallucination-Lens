.PHONY: setup smoke test lint run-local run-gemini report-replication ask

setup:            ## create venv + install everything (Mac/Linux)
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[all]"

smoke:            ## 10-second check that the install works (no model download, no API key)
	hallucination-lens smoke-test

test:             ## run unit tests
	pytest

lint:
	ruff check src tests

run-local:        ## full pipeline with a local model on the Apple GPU
	hallucination-lens run-all --config configs/default.yaml

run-gemini:       ## full pipeline with Gemini (needs GEMINI_API_KEY in .env)
	hallucination-lens run-all --config configs/gemini_flash.yaml

report-replication:  ## re-run the original 150-question setup from the course report
	hallucination-lens run-all --config configs/reproduce_report.yaml

ask:              ## try the guardrail on one question: make ask Q="..." RUN=runs/<name>
	hallucination-lens ask "$(Q)" --run $(RUN)
