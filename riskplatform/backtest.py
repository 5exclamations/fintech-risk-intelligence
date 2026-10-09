"""Rolling-origin (walk-forward) backtest.

The headline result uses ONE train/valid/test split, so it can't say how much performance varies with *when* the model
is trained. This re-runs the identical training procedure at several earlier origins; every origin trains only on
labels that were mature at that origin and is scored on the following ``test_days`` days. Results are SYNTHETIC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COST, SPLIT, SplitConfig
from .evaluation import business_cost, confusion, ranking_metrics
from .modeling import train_all


def origins(split: SplitConfig = SPLIT, n_origins: int = 4, step_days: int = 30) -> list[SplitConfig]:
    """Splits ending at the production origin and ``step_days`` earlier (oldest first); the last equals ``split``."""
    out = [SplitConfig(label_delay_days=split.label_delay_days, warmup_days=split.warmup_days,
                       train_end_day=split.train_end_day - k * step_days, valid_end_day=split.valid_end_day - k * step_days)
           for k in range(n_origins - 1, -1, -1)]
    return [s for s in out if s.train_end_day - s.label_delay_days > s.warmup_days + 30]


def run_backtest(data: pd.DataFrame, split: SplitConfig = SPLIT, n_origins: int = 4, step_days: int = 30, test_days: int = 30,
                 seed: int = 42) -> dict:
    folds = []
    for s in origins(split, n_origins, step_days):
        model, ev, _ = train_all(data, s, COST, seed)
        t = ev[(ev.split == "test") & (ev.day < s.valid_end_day + test_days)]
        y, p = t.is_fraud.values, t.p_model.values
        if y.sum() == 0:
            continue
        a = p >= model.thresholds["review"]
        cm = confusion(y, a)
        c_model, c_none = business_cost(y, a, t.amount.values), business_cost(y, np.zeros(len(y), bool), t.amount.values)
        folds.append({
            "origin_day": s.valid_end_day, "train_days": model.meta["train_days"], "test_days": [int(t.day.min()), int(t.day.max())],
            "n_test": int(len(t)), "n_fraud": int(y.sum()), "algorithm": model.meta["algorithm"],
            "pr_auc": ranking_metrics(y, p)["pr_auc"], "pr_auc_rules": ranking_metrics(y, t.score_rules.values)["pr_auc"],
            "precision": cm["precision"], "recall": cm["recall"], "false_positive_rate": cm["false_positive_rate"],
            "review_threshold": model.thresholds["review"],
            "cost_saving_pct": 100 * (1 - c_model["total_cost"] / c_none["total_cost"]) if c_none["total_cost"] else float("nan")})
    f = pd.DataFrame(folds)
    summary = {c: {"mean": float(f[c].mean()), "std": float(f[c].std(ddof=0)), "min": float(f[c].min()), "max": float(f[c].max())}
               for c in ("pr_auc", "precision", "recall", "cost_saving_pct")} if len(f) else {}
    return {"disclaimer": "SYNTHETIC data; spread across origins shows sensitivity to training date, not real-world uncertainty.",
            "n_folds": len(f), "step_days": step_days, "test_days": test_days, "folds": folds, "summary": summary}
