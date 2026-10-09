"""Dataset assembly, time-based splitting, model training, calibration and operating-point selection."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import __version__
from .anomaly import IsoForest, RobustZ
from .config import ARTIFACT_DIR, COST, SPLIT, CostConfig, SplitConfig
from .evaluation import pick_threshold_by_cost, pick_threshold_precision
from .explain import baseline_values
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features

REF_TABLES = ["customers", "accounts", "merchants", "locations"]


# ------------------------------------------------------------------------------------------------ data
def load_reference(engine: Engine) -> dict[str, pd.DataFrame]:
    with engine.connect() as con:
        return {t: pd.read_sql_query(text(f"SELECT * FROM {t}"), con) for t in REF_TABLES}


def make_dataset(engine: Engine, split: SplitConfig = SPLIT) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Features + metadata for every transaction. The ``synthetic_truth`` tables are deliberately NOT read here."""
    with engine.connect() as con:
        tx = pd.read_sql_query(text("SELECT * FROM transactions ORDER BY ts_epoch, txn_id"), con)
    refs = load_reference(engine)
    feats = build_features(tx, refs["customers"], refs["accounts"], refs["merchants"], refs["locations"], split.label_delay_days)
    epoch0 = (tx.ts_epoch.min() // 86400) * 86400
    meta = tx[["txn_id", "customer_id", "merchant_id", "account_id", "ts_epoch", "channel", "is_fraud"]].copy()
    meta["ts"] = pd.to_datetime(tx.ts)
    meta["day"] = ((tx.ts_epoch - epoch0) // 86400).astype(int)
    cat = refs["merchants"].set_index("merchant_id").category
    meta["category"] = cat.reindex(tx.merchant_id.values).values
    seg = refs["customers"].set_index("customer_id").segment
    meta["segment"] = seg.reindex(tx.customer_id.values).values
    data = meta.merge(feats, on="txn_id", how="left", validate="1:1")
    return data, refs


def split_masks(day: pd.Series, split: SplitConfig = SPLIT) -> dict[str, np.ndarray]:
    """Time-based split with label-maturity gaps (see config.SplitConfig)."""
    d = day.values
    delay = split.label_delay_days
    return {
        "train": (d >= split.warmup_days) & (d < split.train_end_day - delay),
        "valid": (d >= split.train_end_day) & (d < split.valid_end_day - delay),
        "test": d >= split.valid_end_day,
    }


# ----------------------------------------------------------------------------------------------- models
def _signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def make_logreg() -> object:
    return make_pipeline(FunctionTransformer(_signed_log, feature_names_out="one-to-one"),
                         SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.3, class_weight="balanced", max_iter=800))


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=350, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=CATEGORICAL_FEATURES, random_state=seed, early_stopping=False)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    """Monotone sigmoid calibration of a raw model score fitted on the (time-later) validation window."""

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "Platt":
        self.lr_ = LogisticRegression(C=1e6, max_iter=500).fit(_logit(raw)[:, None], y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(_logit(raw)[:, None])[:, 1]


class RiskModel:
    """Everything needed to score a transaction from its feature row. Persisted with joblib."""

    def __init__(self, estimator, calibrator: Platt, features: list[str], thresholds: dict, baseline: dict, meta: dict,
                 name: str):
        self.estimator, self.calibrator, self.features = estimator, calibrator, features
        self.thresholds, self.baseline, self.meta, self.name = thresholds, baseline, meta, name

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.features])[:, 1]

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        return _logit(self.raw(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(fraud label)."""
        return self.calibrator.predict(self.raw(X))

    def band(self, p: np.ndarray) -> np.ndarray:
        t = self.thresholds
        return np.where(p >= t["block"], "block", np.where(p >= t["review"], "review", "allow"))

    def save(self, path=None):
        path = path or ARTIFACT_DIR / "model" / "model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        (path.parent / "model_meta.json").write_text(json.dumps({**self.meta, "thresholds": self.thresholds, "name": self.name,
                                                                  "features": self.features}, indent=2, default=str))
        return path

    @staticmethod
    def load(path=None) -> "RiskModel":
        return joblib.load(path or ARTIFACT_DIR / "model" / "model.joblib")


# --------------------------------------------------------------------------------------------- training
def train_all(data: pd.DataFrame, split: SplitConfig = SPLIT, cost: CostConfig = COST, seed: int = 42,
              features: list[str] | None = None) -> tuple[RiskModel, pd.DataFrame, dict]:
    """Fit unsupervised detectors + supervised candidates on TRAIN, calibrate / choose thresholds on VALID,
    and return the chosen model plus a scores frame for valid + test. TEST labels are not used anywhere here."""
    features = features or NUMERIC_FEATURES
    from .rules import evaluate_rules, rule_score  # local import: rules depend on feature names only

    masks = split_masks(data.day, split)
    tr, va, te = (data[masks[k]] for k in ("train", "valid", "test"))
    ytr, yva = tr.is_fraud.values, va.is_fraud.values

    # --- unsupervised detectors (labels unused)
    rz = RobustZ().fit(tr)
    iso = IsoForest(seed).fit(tr.sample(min(len(tr), 150_000), random_state=seed))

    # --- supervised candidates
    candidates = {"logreg": make_logreg().fit(tr[features], ytr), "hgb": make_hgb(seed).fit(tr[features], ytr)}
    from sklearn.metrics import average_precision_score
    valid_ap = {k: float(average_precision_score(yva, m.predict_proba(va[features])[:, 1])) for k, m in candidates.items()}
    best = max(valid_ap, key=valid_ap.get)
    est = candidates[best]

    # --- calibration + operating points on VALID only
    cal = Platt().fit(est.predict_proba(va[features])[:, 1], yva)
    p_va = cal.predict(est.predict_proba(va[features])[:, 1])
    thr_review, valid_cost = pick_threshold_by_cost(yva, p_va, va.amount.values, cost)
    thr_block = pick_threshold_precision(yva, p_va, 0.90)
    if not np.isfinite(thr_review):  # model never beats "no alerting" on cost: fall back to a 1% alert budget
        thr_review = float(np.quantile(p_va, 0.99))
    thr_block = max(thr_block, thr_review) if np.isfinite(thr_block) else 1.01
    thresholds = {"review": float(thr_review), "block": float(thr_block)}

    legit_tr = (tr.is_fraud.values == 0)
    meta = {
        "model_version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": type(est[-1] if hasattr(est, "steps") else est).__name__, "selected_by": "validation PR-AUC",
        "candidate_valid_pr_auc": valid_ap, "seed": seed,
        "split": asdict(split), "cost_assumptions": asdict(cost),
        "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
        "train_fraud_rate": float(ytr.mean()), "valid_fraud_rate": float(yva.mean()),
        "train_days": [int(tr.day.min()), int(tr.day.max())], "valid_days": [int(va.day.min()), int(va.day.max())],
        "synthetic_data": True,
        "disclaimer": "Trained and evaluated on SYNTHETIC data. Metrics do not describe real-world performance.",
    }
    model = RiskModel(est, cal, features, thresholds, baseline_values(tr[features], legit_tr), meta, best)
    model.detectors = {"robust_z": rz, "iforest": iso}
    recent = tr[tr.day >= tr.day.max() - 60]
    model.reference = recent[features].sample(min(len(recent), 30_000), random_state=seed)  # last 60 train days: drift reference

    # --- score valid + test (every method, same rows)
    ev = data[masks["valid"] | masks["test"]].copy()
    ev["split"] = np.where(ev.day >= split.valid_end_day, "test", "valid")
    hits = evaluate_rules(ev)
    ev["score_rules"] = rule_score(hits)
    ev["rule_hits"] = ["|".join(hits.columns[r]) for r in hits.values]
    ev["score_robust_z"] = rz.score(ev)
    ev["score_iforest"] = iso.score(ev)
    ev["score_logreg"] = candidates["logreg"].predict_proba(ev[features])[:, 1]
    ev["score_hgb"] = candidates["hgb"].predict_proba(ev[features])[:, 1]
    ev["p_model"] = model.predict(ev)
    ev["band"] = model.band(ev.p_model.values)
    info = {"valid_cost_at_review_threshold": valid_cost, "valid_pr_auc": valid_ap}
    return model, ev, info
