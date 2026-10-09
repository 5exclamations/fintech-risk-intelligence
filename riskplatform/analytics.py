"""Run the SQL analytics library and persist results for the dashboard / docs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from .config import ARTIFACT_DIR, SQL_DIR
from .db import read_sql_file

DEFAULT_PARAMS = {"limit": 25, "k": 200, "min_txn": 100}


def list_queries() -> list[Path]:
    return sorted((SQL_DIR / "analytics").glob("*.sql"))


def run_all(engine: Engine, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or ARTIFACT_DIR / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in list_queries():
        sql = path.read_text()
        params = {k: v for k, v in DEFAULT_PARAMS.items() if f":{k}" in sql}
        df = read_sql_file(engine, path, params)
        results[path.stem] = df
        df.to_csv(out_dir / f"{path.stem}.csv", index=False)
    return results
