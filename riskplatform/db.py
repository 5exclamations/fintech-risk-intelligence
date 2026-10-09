"""Database access: PostgreSQL (docker-compose) or SQLite (local / unit tests) through SQLAlchemy."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sqlalchemy import DateTime, create_engine, event, text
from sqlalchemy.engine import Engine

from .config import DATA_DIR, SQL_DIR, db_url

TABLES = ["locations", "customers", "accounts", "merchants", "transactions", "synthetic_truth", "synthetic_truth_merchants"]


def get_engine(url: str | None = None) -> Engine:
    url = url or db_url()
    if url.startswith("sqlite"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _add_math(conn, _):  # SQLite lacks SQRT unless compiled with math functions
            conn.create_function("sqrt", 1, lambda x: None if x is None else math.sqrt(max(x, 0.0)), deterministic=True)
    return engine


def init_schema(engine: Engine) -> None:
    ddl = "\n".join(line.split("--")[0] for line in (SQL_DIR / "schema.sql").read_text().splitlines())
    with engine.begin() as con:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            con.execute(text(stmt))


def load_tables(engine: Engine, tables: dict[str, pd.DataFrame], chunk: int = 20000) -> None:
    init_schema(engine)
    with engine.begin() as con:
        for name in TABLES:
            df = tables[name]
            dtypes = {c: DateTime() for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])}
            df.to_sql(name, con, if_exists="append", index=False, chunksize=chunk, dtype=dtypes)


def read_sql_file(engine: Engine, path: str | Path, params: dict | None = None) -> pd.DataFrame:
    # strip comments: ":name" inside a comment would otherwise be parsed as a bind parameter
    sql = "\n".join(line.split("--")[0] for line in Path(path).read_text().splitlines())
    with engine.connect() as con:
        return pd.read_sql_query(text(sql), con, params=params or {})


def read_transactions(engine: Engine) -> pd.DataFrame:
    with engine.connect() as con:
        df = pd.read_sql_query(text("SELECT * FROM transactions ORDER BY ts_epoch, txn_id"), con)
    df["ts"] = pd.to_datetime(df["ts"])
    return df
