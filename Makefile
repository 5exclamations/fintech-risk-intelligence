PY ?= .venv/bin/python   # Windows: use scripts/dev.ps1

.PHONY: setup data train backtest all test lint api dashboard screenshots docker
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
all:           ## generate -> load -> SQL analytics -> train/evaluate -> monitor (SQLite unless DATABASE_URL is set)
	$(PY) -m riskplatform.pipeline all
data:
	$(PY) -m riskplatform.pipeline generate
train:
	$(PY) -m riskplatform.pipeline train
test:
	$(PY) -m pytest -q
lint:
	.venv/bin/ruff check .
api:
	RISK_API_KEY=$${RISK_API_KEY:-dev-demo-key} .venv/bin/uvicorn riskplatform.api:app --port 8000
dashboard:
	.venv/bin/streamlit run dashboard/app.py
screenshots:   ## needs: pip install playwright + Google Chrome; dashboard running on :8501
	$(PY) scripts/screenshots.py http://localhost:8501
docker:
	docker compose up --build
