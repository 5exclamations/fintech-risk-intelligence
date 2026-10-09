# Verification status (as of the last commit)

Verified by execution (fresh `git clone` of the pushed repository, Python 3.14 venv, macOS):
* `python -m riskplatform.pipeline all` regenerates the dataset and retrains from scratch → PR-AUC 0.730, precision 0.802, recall 0.683 (identical to the documented numbers).
* `RISK_FULL_REPRO=1 pytest -q` → 41 passed, 1 skipped (the PostgreSQL test, which needs `DATABASE_URL`); `ruff check .` clean.
* `pytest tests/test_postgres.py` with `DATABASE_URL` → local PostgreSQL 17: 1 passed; the full pipeline also gave identical metrics on PostgreSQL.
* API (uvicorn, `RISK_API_KEY` set): `/health` open, 401 without/with wrong key, authenticated replay and POST scoring return scores + reason codes. Dashboard (Streamlit) served and rendered all 9 tabs without errors.

Verified on GitHub Actions (run on the latest commit, all jobs green): ruff; pytest on Python 3.11 and 3.12 with a PostgreSQL 16 service (41 passed; the PostgreSQL test is asserted to run, not skip); the full-scale reproducibility test (`RISK_FULL_REPRO=1`); and the `docker` job, which runs `docker compose up --build`, checks the API returns 401 without a key and a score with one, and checks the dashboard health endpoint.

Not verified on the author's machine: `docker compose up` locally (the Colima Docker VM's image store was corrupted when the host disk filled up during an earlier attempt). Not verified anywhere: TLS/production deployment, load/latency testing, any real-world data.
