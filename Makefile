# ufc-predictor
#
# CLAUDE.md Definition of Done names exactly four commands:
#   make setup && make data && make backtest && make report
# so `backtest` depends on `features`, and `features` on `data`. The finer targets
# exist for iteration; the four-command path is the contract.

.DEFAULT_GOAL := help
.PHONY: help setup data features backtest parlay report test test-leakage lint fmt app clean

UV ?= uv
RUN := $(UV) run

# Stamp files let make skip stages whose outputs already exist.
PROCESSED := data/processed
FEATURES  := $(PROCESSED)/features.parquet
FIGHTS    := $(PROCESSED)/fights.parquet
METRICS   := reports/metrics.json

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install pinned dependencies (Python 3.11)
	$(UV) sync --all-groups
	@$(RUN) python -c "import sys; v=sys.version_info; \
		assert (v.major, v.minor) == (3, 11), f'expected Python 3.11, got {v.major}.{v.minor}'; \
		print(f'Python {v.major}.{v.minor}.{v.micro} ready')"

data: ## Build normalized Parquet tables from the committed snapshot (no network)
	$(RUN) ufc-predictor data

$(FIGHTS): | data

features: ## Build the point-in-time feature table
	$(RUN) ufc-predictor features

$(FEATURES): | features

backtest: features ## Walk-forward backtest; writes reports/metrics.json
	$(RUN) ufc-predictor backtest

parlay: ## Price parlays from stored predictions (research backtest, not betting advice)
	$(RUN) ufc-predictor parlay

report: ## Generate charts and refresh README figures from metrics.json
	$(RUN) ufc-predictor report

test: ## Run the full test suite
	$(RUN) pytest

test-leakage: ## Run only the point-in-time correctness suite (must always pass)
	$(RUN) pytest -m leakage -v

lint: ## Lint and format-check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

fmt: ## Apply formatting and autofixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

app: ## Run the Streamlit demo locally
	$(RUN) streamlit run app/streamlit_app.py

clean: ## Remove derived data, reports, and caches (never touches data/raw)
	rm -rf $(PROCESSED)/* reports/*.png reports/*.json reports/*.csv
	rm -rf .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
