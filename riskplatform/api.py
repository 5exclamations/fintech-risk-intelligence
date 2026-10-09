"""Risk-scoring API (FastAPI).  uvicorn riskplatform.api:app

POST /score             score a new transaction (history looked up server-side, point-in-time)
GET  /score/{txn_id}    replay: re-score a stored transaction using only history before it
GET  /model             model card metadata
GET  /monitoring        latest drift/performance summary
GET  /health
"""
from __future__ import annotations

import json
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy import text

from .config import ARTIFACT_DIR
from .db import get_engine
from .explain import group_contributions, model_reasons
from .features import build_features
from .modeling import RiskModel, load_reference
from .rules import RULE_INDEX, evaluate_rules, rule_score

SYNTHETIC_NOTICE = ("DEMONSTRATION ONLY: model trained on synthetic data; scores illustrate the method, are not validated on real "
                    "transactions and must not be used for real fraud, credit or account decisions.")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TransactionIn(BaseModel):
    customer_id: str = Field(max_length=16, pattern=r"^[A-Za-z0-9_-]+$")
    account_id: str = Field(max_length=16, pattern=r"^[A-Za-z0-9_-]+$")
    merchant_id: str = Field(max_length=16, pattern=r"^[A-Za-z0-9_-]+$")
    location_id: int = Field(description="Where the transaction happens (POS city or IP-geolocated city)")
    amount: float = Field(gt=0, lt=1e7, allow_inf_nan=False)
    channel: Literal["card_present", "online", "atm"]
    device_id: str | None = Field(default=None, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    ts: datetime | None = Field(default=None, description="UTC timestamp; defaults to now")


class Reason(BaseModel):
    code: str
    source: Literal["rule", "model"]
    detail: str
    contribution: float | None = None


class ScoreOut(BaseModel):
    risk_score: float = Field(description="Calibrated estimate of P(fraud label); trained on synthetic data")
    risk_band: Literal["allow", "review", "block"]
    thresholds: dict[str, float]
    reasons: list[Reason]
    rule_score: float
    model_version: str
    notice: str = SYNTHETIC_NOTICE


class Scorer:
    def __init__(self, engine, model: RiskModel):
        self.model = model
        self.refs = load_reference(engine)
        with engine.connect() as con:
            tx = pd.read_sql_query(text("SELECT * FROM transactions ORDER BY ts_epoch, txn_id"), con)
        self.tx = tx
        self.by_cust = tx.groupby("customer_id").indices
        self.by_merch = tx.groupby("merchant_id").indices
        self.tx_pos = {t: i for i, t in enumerate(tx.txn_id.values)}
        self.cust_ids = set(self.refs["customers"].customer_id)
        self.acc_ids = set(self.refs["accounts"].account_id)
        self.merch_ids = set(self.refs["merchants"].merchant_id)
        self.loc = self.refs["locations"].set_index("location_id")

    def _history(self, customer_id: str, merchant_id: str, ts_epoch: int, exclude_txn: str | None = None) -> pd.DataFrame:
        idx = np.union1d(self.by_cust.get(customer_id, np.array([], int)), self.by_merch.get(merchant_id, np.array([], int)))
        h = self.tx.iloc[idx]
        h = h[h.ts_epoch < ts_epoch]  # strictly past: also guarantees replay never sees its own row or the future
        return h

    def score_row(self, row: dict, exclude_txn: str | None = None) -> dict:
        h = self._history(row["customer_id"], row["merchant_id"], row["ts_epoch"])
        new = pd.DataFrame([{**row, "txn_id": "__new__", "is_fraud": np.nan}])
        cols = list(self.tx.columns)
        frame = pd.concat([h[cols], new[cols]], ignore_index=True)
        r = self.refs
        f = build_features(frame, r["customers"], r["accounts"], r["merchants"], r["locations"], self.model.meta["split"]["label_delay_days"])
        x = f.iloc[[-1]].reset_index(drop=True)
        p = float(self.model.predict(x)[0])
        hits = evaluate_rules(x)
        contrib = group_contributions(self.model.logit, x[self.model.features], self.model.baseline)
        reasons = [{"code": c, "source": "rule", "detail": RULE_INDEX[c].text} for c in hits.columns[hits.values[0]]]
        reasons += model_reasons(contrib.iloc[0], x.iloc[0])
        return {"risk_score": p, "risk_band": str(self.model.band(np.array([p]))[0]), "thresholds": self.model.thresholds,
                "reasons": reasons, "rule_score": float(rule_score(hits)[0]), "model_version": self.model.meta["model_version"], "notice": SYNTHETIC_NOTICE}

    def score_new(self, t: TransactionIn) -> dict:
        if t.customer_id not in self.cust_ids or t.merchant_id not in self.merch_ids or t.account_id not in self.acc_ids \
                or t.location_id not in self.loc.index:
            raise KeyError("unknown customer_id / account_id / merchant_id / location_id")
        ts = pd.Timestamp(t.ts or datetime.now(timezone.utc)).tz_localize(None) if (t.ts is None or t.ts.tzinfo is None) \
            else pd.Timestamp(t.ts).tz_convert("UTC").tz_localize(None)
        epoch = int((ts - pd.Timestamp("1970-01-01")).total_seconds())
        local = ts + pd.Timedelta(hours=float(self.loc.loc[t.location_id, "utc_offset"]))
        row = {"account_id": t.account_id, "customer_id": t.customer_id, "merchant_id": t.merchant_id, "location_id": t.location_id,
               "ts": ts, "txn_date": ts.strftime("%Y-%m-%d"), "hour_local": local.hour, "dow_local": local.dayofweek,
               "ts_epoch": epoch, "amount": t.amount, "channel": t.channel, "device_id": t.device_id}
        return self.score_row(row)

    def replay(self, txn_id: str) -> dict:
        if txn_id not in self.tx_pos:
            raise KeyError(txn_id)
        row = self.tx.iloc[self.tx_pos[txn_id]].to_dict()
        out = self.score_row(row)
        out["actual_label"] = int(row["is_fraud"])
        return out


def create_app(engine=None, model: RiskModel | None = None, api_key: str | None = None,
               rate_limit_per_min: int | None = None) -> FastAPI:
    """Security model (demo-grade, see docs/api_security.md): API-key header required on every endpoint except /health,
    fail-closed at startup if no key is configured (explicit opt-out: RISK_API_AUTH=disabled for local experiments),
    constant-time key comparison, strict input validation, and a per-client sliding-window rate limit."""
    key = api_key or os.environ.get("RISK_API_KEY")
    auth_disabled = os.environ.get("RISK_API_AUTH", "").lower() == "disabled"
    if not key and not auth_disabled:
        raise RuntimeError("Set RISK_API_KEY (or RISK_API_AUTH=disabled for a throw-away local demo); refusing to start an open scoring API.")
    limit = rate_limit_per_min or int(os.environ.get("RISK_API_RATE_LIMIT", "120"))
    hits: dict[str, deque] = defaultdict(deque)

    def guard(request: Request, supplied: str | None = Depends(_api_key_header)):
        if not auth_disabled and not (supplied and secrets.compare_digest(supplied.encode(), key.encode())):
            raise HTTPException(401, "missing or invalid X-API-Key", headers={"WWW-Authenticate": "ApiKey"})
        client = (supplied or "") + "|" + (request.client.host if request.client else "?")
        q, now = hits[client], time.monotonic()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
        q.append(now)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine or get_engine()
        mdl = model or RiskModel.load()
        app.state.scorer = Scorer(eng, mdl)
        yield

    app = FastAPI(title="Transaction Risk Scoring API - SYNTHETIC-DATA DEMO, NOT FOR PRODUCTION DECISIONS", version="1.0.0",
                  lifespan=lifespan, description=SYNTHETIC_NOTICE, docs_url="/docs", redoc_url=None)

    @app.middleware("http")
    async def demo_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Demo-Notice"] = "synthetic-data-demo; not-a-production-fraud-decision-system"
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_version": request.app.state.scorer.model.meta["model_version"]}

    @app.post("/score", response_model=ScoreOut, dependencies=[Depends(guard)])
    def score(t: TransactionIn, request: Request):
        try:
            return request.app.state.scorer.score_new(t)
        except KeyError as e:
            raise HTTPException(422, str(e))

    @app.get("/score/{txn_id}", dependencies=[Depends(guard)])
    def replay(txn_id: str, request: Request):
        try:
            return request.app.state.scorer.replay(txn_id)
        except KeyError:
            raise HTTPException(404, f"unknown txn_id {txn_id}")

    @app.get("/model", dependencies=[Depends(guard)])
    def model_info(request: Request):
        m = request.app.state.scorer.model
        return {**m.meta, "thresholds": m.thresholds, "features": m.features, "algorithm_name": m.name}

    @app.get("/monitoring", dependencies=[Depends(guard)])
    def monitoring():
        p = ARTIFACT_DIR / "monitoring" / "summary.json"
        if not p.exists():
            raise HTTPException(404, "no monitoring summary yet; run the pipeline")
        return json.loads(p.read_text())

    return app


def __getattr__(name):  # lazy: `uvicorn riskplatform.api:app` builds the app on first access
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
