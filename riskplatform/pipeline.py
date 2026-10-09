"""Reproducible end-to-end pipeline:  python -m riskplatform.pipeline all"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from . import analytics as analytics_mod
from .config import ARTIFACT_DIR, COST, GEN, SPLIT, GenConfig, SplitConfig
from .db import get_engine, load_tables
from .evaluation import at_budget, block_bootstrap_pr_auc, business_cost, confusion, false_positive_analysis, ranking_metrics, workload
from .explain import group_contributions, model_reasons
from .generate import generate
from .manifest import build_manifest
from .modeling import make_dataset, train_all
from .monitoring import summarize, windowed
from .rules import RULE_INDEX
from .slices import slice_report

DISCLAIMER = ("ALL RESULTS ARE FROM SYNTHETIC DATA generated with documented, invented fraud typologies. "
              "They demonstrate the methodology, not real-world performance.")
SCORE_COLS = {"Rules (noisy-OR)": "score_rules", "Robust z-score": "score_robust_z", "Isolation Forest": "score_iforest",
              "Logistic regression": "score_logreg", "Gradient boosting (calibrated)": "p_model"}


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def step_generate(engine, gen: GenConfig = GEN):
    tables = generate(gen)
    load_tables(engine, tables)
    return {k: len(v) for k, v in tables.items()}


def step_analytics(engine):
    return analytics_mod.run_all(engine)


def step_train_evaluate(engine, split: SplitConfig = SPLIT, out_dir: Path | None = None, seed: int = 42) -> dict:
    out = out_dir or ARTIFACT_DIR
    data, refs = make_dataset(engine, split)
    model, ev, info = train_all(data, split, COST, seed)
    model.save(out / "model" / "model.joblib")
    with engine.connect() as con:
        truth = pd.read_sql_query(text("SELECT txn_id, true_pattern, is_fraud_true FROM synthetic_truth"), con)
    ev = ev.merge(truth, on="txn_id", how="left")  # DIAGNOSTICS ONLY (after training); not available in real life
    test, valid = ev[ev.split == "test"].reset_index(drop=True), ev[ev.split == "valid"].reset_index(drop=True)
    thr = model.thresholds
    n_days_test = test.day.nunique()
    y, p = test.is_fraud.values, test.p_model.values
    alert = p >= thr["review"]

    # --- reason codes for the alerts (model occlusion + rule hits)
    flagged = test[alert].copy()
    contrib = group_contributions(model.logit, flagged[model.features], model.baseline)
    reasons = []
    for i, (idx, row) in enumerate(flagged.iterrows()):
        r = [{"code": c, "source": "rule", "detail": RULE_INDEX[c].text} for c in row.rule_hits.split("|") if c]
        r += model_reasons(contrib.loc[idx], row[model.features])
        reasons.append(r)
    flagged["reasons"] = [json.dumps(r) for r in reasons]
    flagged["top_reason"] = [r[0]["code"] if r else "" for r in reasons]
    test["reasons"] = ""
    test.loc[flagged.index, "reasons"] = flagged["reasons"]
    test["top_reason"] = ""
    test.loc[flagged.index, "top_reason"] = flagged["top_reason"]

    # --- per-method comparison (same test rows, same alert budget as the chosen model)
    budget = int(alert.sum())
    methods = {}
    for name, col in SCORE_COLS.items():
        m = ranking_metrics(y, test[col].values)
        b = at_budget(y, test[col].values, budget)
        m.update({"precision_at_budget": b["precision"], "recall_at_budget": b["recall"], "budget_alerts": budget})
        methods[name] = m
    cm = confusion(y, alert)
    cost_model = business_cost(y, alert, test.amount.values)
    cost_none = business_cost(y, np.zeros(len(y), bool), test.amount.values)
    cost_all = business_cost(y, np.ones(len(y), bool), test.amount.values)
    ci = block_bootstrap_pr_auc(y, p, test.day.values, n_boot=150)

    # --- time breakdown of test performance
    by_month = []
    for w, g in test.groupby((test.day - test.day.min()) // 30):
        gy = g.is_fraud.values.astype(bool); ga = (g.p_model >= thr["review"]).values
        rm = ranking_metrics(gy, g.p_model.values)
        by_month.append({"block_30d": int(w), "first_day": int(g.day.min()), "n_txn": len(g), "fraud_rate": float(gy.mean()),
                         "pr_auc": rm["pr_auc"], "precision": confusion(gy, ga)["precision"], "recall": confusion(gy, ga)["recall"]})

    # --- synthetic-only diagnostics using generator ground truth
    tp = test[test.is_fraud_true == 1]
    by_pattern = {pat: {"n": int(len(g)), "recall_at_review_threshold": float((g.p_model >= thr["review"]).mean()),
                        "mean_score": float(g.p_model.mean())} for pat, g in tp.groupby("true_pattern")}
    fps = test[alert & (test.is_fraud == 0)]
    fp_truly_fraud = float((fps.is_fraud_true == 1).mean()) if len(fps) else 0.0

    metrics = {
        "disclaimer": DISCLAIMER,
        "splits": {k: {"rows": int(len(d)), "fraud_rate": float(d.is_fraud.mean()), "days": [int(d.day.min()), int(d.day.max())]}
                   for k, d in (("valid", valid), ("test", test))},
        "model": {"name": model.name, "thresholds": thr, "candidate_valid_pr_auc": model.meta["candidate_valid_pr_auc"]},
        "methods_test": methods,
        "methods_valid": {n: ranking_metrics(valid.is_fraud.values, valid[c].values) for n, c in SCORE_COLS.items()},
        "test_operating_point": {**cm, **ci, "threshold": thr["review"]},
        "block_band": {**confusion(y, p >= thr["block"]), "threshold": thr["block"]},
        "band_counts": test.band.value_counts().to_dict(),
        "cost": {"model": cost_model, "no_model": cost_none, "alert_everything": cost_all,
                 "saving_vs_no_model": cost_none["total_cost"] - cost_model["total_cost"],
                 "saving_pct": 100 * (1 - cost_model["total_cost"] / cost_none["total_cost"]),
                 "assumptions": asdict(COST)},
        "workload": workload(int(alert.sum()), n_days_test),
        "false_positive_analysis": false_positive_analysis(test.assign(rule_hits=test.rule_hits), alert),
        "by_30d_block": by_month,
        "synthetic_only_diagnostics": {"recall_by_true_pattern": by_pattern, "share_of_false_positives_that_are_undiscovered_fraud": fp_truly_fraud,
                                       "note": "uses generator ground truth; impossible in real life"},
        "calibration": _calibration(test),
        "calibration_summary": _calibration_summary(y, p),
        "slice_performance": slice_report(test.assign(history=pd.cut(test.cust_prior_n.fillna(0), [-1, 9, 49, 1e9],
                                                                     labels=["new (<10)", "established (10-49)", "mature (50+)"]).astype(str)),
                                          alert, ["segment", "channel", "category", "amount_bucket", "history"]),
    }
    _dump(metrics, out / "metrics.json")
    gbm = metrics["methods_test"]["Gradient boosting (calibrated)"]
    _dump(build_manifest(data[["txn_id", "customer_id", "merchant_id", "account_id", "ts_epoch", "channel", "amount", "is_fraud"]], seed,
                         {"pr_auc": gbm["pr_auc"], "precision": cm["precision"], "recall": cm["recall"], "synthetic": True}),
          out / "run_manifest.json")

    # --- monitoring: reference = training sample; windows over validation + test
    p_ref = model.predict(model.reference)
    mon = windowed(ev, model.reference, model.features, p_ref, 14, thr["review"], SPLIT.label_delay_days)
    base_valid = metrics["methods_valid"]["Gradient boosting (calibrated)"]["pr_auc"]
    vy = valid.is_fraud.values.astype(bool)
    base_rec = confusion(vy, valid.p_model.values >= thr["review"])["recall"]
    summ = summarize(mon["drift"], mon["performance"], base_valid, base_rec)
    summ["baseline"] = {"valid_pr_auc": base_valid, "valid_recall": base_rec}
    last_w = int(mon["drift"].window.max())
    last_ev = ev[((ev.day - ev.day.min()) // 14) == last_w]
    from .monitoring import MONOTONIC_FEATURES, feature_drift
    fd = feature_drift(model.reference, last_ev, [f for f in model.features if f not in MONOTONIC_FEATURES])
    mdir = out / "monitoring"; mdir.mkdir(parents=True, exist_ok=True)
    mon["drift"].to_csv(mdir / "drift_windows.csv", index=False)
    mon["performance"].to_csv(mdir / "performance_windows.csv", index=False)
    fd.to_csv(mdir / "feature_psi_latest.csv", index=False)
    _dump(summ, mdir / "summary.json")

    keep = ["txn_id", "split", "day", "ts", "customer_id", "merchant_id", "category", "channel", "amount", "is_fraud", "p_model", "band",
            "score_rules", "score_robust_z", "score_iforest", "score_logreg", "rule_hits", "reasons", "top_reason", "segment", "true_pattern"]
    sc = pd.concat([valid.assign(reasons="", top_reason=""), test])[keep]
    (out / "scored").mkdir(parents=True, exist_ok=True)
    sc.to_parquet(out / "scored" / "scored.parquet", index=False)
    return metrics


def _calibration(test: pd.DataFrame, bins: int = 8) -> list[dict]:
    q = pd.qcut(test.p_model.rank(method="first"), bins, labels=False)
    g = test.groupby(q).agg(mean_pred=("p_model", "mean"), observed=("is_fraud", "mean"), n=("is_fraud", "size"))
    return g.reset_index(drop=True).round(5).to_dict("records")


def _calibration_summary(y, p, bins: int = 10) -> dict:
    """Brier score and (quantile-bin) expected calibration error; base-rate shift between valid and test inflates both."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"y": y, "p": p, "q": q.values}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return {"brier": float(np.mean((p - y) ** 2)), "brier_baseline_constant": float(np.mean((y.mean() - y) ** 2)),
            "ece": float((g.n * (g.y - g.p).abs()).sum() / g.n.sum()), "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


def step_backtest(engine, split: SplitConfig = SPLIT, out_dir: Path | None = None, seed: int = 42, **kw) -> dict:
    from .backtest import run_backtest
    data, _ = make_dataset(engine, split)
    res = run_backtest(data, split, seed=seed, **kw)
    _dump(res, (out_dir or ARTIFACT_DIR) / "backtest.json")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Financial Transaction Risk Intelligence Platform pipeline (synthetic data)")
    ap.add_argument("step", choices=["generate", "analytics", "train", "backtest", "all"])
    ap.add_argument("--customers", type=int, default=GEN.n_customers)
    ap.add_argument("--seed", type=int, default=GEN.seed)
    a = ap.parse_args(argv)
    engine = get_engine()
    gen = GenConfig(**{**asdict(GEN), "n_customers": a.customers, "seed": a.seed})
    if a.step in ("generate", "all"):
        print("generated rows:", step_generate(engine, gen))
    if a.step in ("analytics", "all"):
        r = step_analytics(engine); print("analytics queries:", ", ".join(r))
    if a.step == "backtest":
        r = step_backtest(engine, SPLIT, seed=a.seed)
        print("[SYNTHETIC] rolling-origin backtest:", {k: round(v["mean"], 3) for k, v in r["summary"].items()}, "folds:", r["n_folds"])
    if a.step in ("train", "all"):
        m = step_train_evaluate(engine, SPLIT, seed=a.seed)
        op = m["test_operating_point"]
        print(f"[SYNTHETIC] test PR-AUC={m['methods_test']['Gradient boosting (calibrated)']['pr_auc']:.3f} "
              f"precision={op['precision']:.3f} recall={op['recall']:.3f} f1={op['f1']:.3f}  cost saving={m['cost']['saving_pct']:.1f}%")


if __name__ == "__main__":
    main()
