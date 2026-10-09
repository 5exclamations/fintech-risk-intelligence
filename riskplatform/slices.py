"""Slice (subgroup) performance and a disparity screen.

Aggregate metrics can hide groups the model serves badly or burdens with false alerts. For each categorical slice this
reports volume, fraud rate, recall, false-positive rate and alert rate, and flags slices whose recall is much lower or
whose false-positive rate is much higher than overall. It is a *screen*: slices below ``min_fraud`` frauds or
``min_legit`` legitimate rows are reported but not flagged (too noisy). It is not a fairness assessment - protected
attributes are not in the data (see docs/limitations.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AMOUNT_BINS, AMOUNT_LABELS = [0, 25, 100, 500, 1e12], ["<25", "25-100", "100-500", "500+"]


def slice_report(df: pd.DataFrame, alert: np.ndarray, by: list[str], min_fraud: int = 30, min_legit: int = 500,
                 recall_ratio_flag: float = 0.7, fpr_ratio_flag: float = 1.5) -> dict:
    d = df.assign(alert=np.asarray(alert).astype(bool), fraud=df.is_fraud.astype(bool))
    d["amount_bucket"] = pd.cut(d.amount, AMOUNT_BINS, labels=AMOUNT_LABELS).astype(str)
    tot_f, tot_l = int(d.fraud.sum()), int((~d.fraud).sum())
    overall = {"recall": float((d.fraud & d.alert).sum() / max(tot_f, 1)), "fpr": float((~d.fraud & d.alert).sum() / max(tot_l, 1))}
    out = {}
    for col in by:
        rows = []
        for key, g in d.groupby(col, observed=True):
            nf, nl = int(g.fraud.sum()), int((~g.fraud).sum())
            rec = float((g.fraud & g.alert).sum() / nf) if nf else float("nan")
            fpr = float((~g.fraud & g.alert).sum() / nl) if nl else float("nan")
            flags = []
            if nf >= min_fraud and overall["recall"] > 0 and rec < recall_ratio_flag * overall["recall"]:
                flags.append("low_recall")
            if nl >= min_legit and overall["fpr"] > 0 and fpr > fpr_ratio_flag * overall["fpr"]:
                flags.append("high_false_positive_rate")
            rows.append({"slice": str(key), "n": len(g), "n_fraud": nf, "fraud_rate": nf / len(g), "recall": rec, "fpr": fpr,
                         "alert_rate": float(g.alert.mean()), "flags": flags})
        out[col] = rows
    flagged = [{"by": c, **r} for c, rs in out.items() for r in rs if r["flags"]]
    return {"overall": overall, "slices": out, "flagged": flagged,
            "note": f"flags need >= {min_fraud} frauds (recall) or >= {min_legit} legitimate rows (FPR); screening only, not a fairness audit"}
