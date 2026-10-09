# Verification status (as of the last commit)

Verified by execution (fresh `git clone` of the pushed repository, Python 3.14 venv, macOS):
* `python -m riskplatform.pipeline all` regenerates the dataset and retrains from scratch → PR-AUC 0.730, precision 0.802, recall 0.683 (identical to the documented numbers).
* `RISK_FULL_REPRO=1 pytest -q` → 41 passed, 1 skipped (the PostgreSQL test, which needs `DATABASE_URL`); `ruff check .` clean.
* `pytest tests/test_postgres.py` with `DATABASE_URL` → local PostgreSQL 17: 1 passed; the full pipeline also gave identical metrics on PostgreSQL.
* API (uvicorn, `RISK_API_KEY` set): `/health` open, 401 without/with wrong key, authenticated replay and POST scoring return scores + reason codes. Dashboard (Streamlit) served and rendered all 9 tabs without errors.

Verified on GitHub Actions (run on the latest commit, all jobs green): ruff; pytest on Python 3.11 and 3.12 with a PostgreSQL 16 service (41 passed; the PostgreSQL test is asserted to run, not skip); the full-scale reproducibility test (`RISK_FULL_REPRO=1`); and the `docker` job, which runs `docker compose up --build`, checks the API returns 401 without a key and a score with one, and checks the dashboard health endpoint.

Not verified on the author's machine: `docker compose up` locally (the Colima Docker VM's image store was corrupted when the host disk filled up during an earlier attempt). Not verified anywhere: TLS/production deployment, load/latency testing, any real-world data.

## Windows re-verification (Windows 10 Pro, Intel i3-6100, 8 GB RAM, Python 3.12.10, pandas 3.0.6 / numpy 2.5.3 / scikit-learn 1.9.1)
* Fresh `.venv`, `pip install -r requirements-dev.txt`; `pipeline all` from scratch (~4.5 min) → PR-AUC 0.730, precision 0.802, recall 0.683, F1 0.738, cost saving 55.7% — identical to the macOS numbers despite much newer library versions.
* `RISK_FULL_REPRO=1 pytest tests/test_reproducibility.py` passed; API (`/health`, 401 without key, authenticated score) and Streamlit (HTTP 200) served locally.
* Docker was not exercised locally: Docker Desktop's engine returned HTTP 500 on `docker info` (machine state, not repository). Docker/PostgreSQL remain verified only by GitHub Actions.
* Windows issues found: the Makefile is POSIX-only (added `scripts/dev.ps1`); the Python 3.15 default interpreter is a pre-release (use 3.12); line endings (`.gitattributes` added). No code changes were needed for paths, SQLite or joblib.
* After the additions in `docs/robustness_and_reproducibility.md`: `ruff` clean; `RISK_FULL_REPRO=1 pytest -q -rs` → **47 passed, 1 skipped** (PostgreSQL test, no `DATABASE_URL` locally); `pipeline backtest` completed (4 folds). GitHub Actions status for this commit is recorded below once checked.
