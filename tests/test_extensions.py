"""Backtest, slice screen, calibration summary and run manifest."""
import json

import numpy as np
import pandas as pd
from conftest import SMALL_SPLIT

from riskplatform.backtest import origins, run_backtest
from riskplatform.manifest import build_manifest, data_fingerprint
from riskplatform.modeling import make_dataset
from riskplatform.pipeline import _calibration_summary, step_train_evaluate
from riskplatform.slices import slice_report


def test_origins_end_at_production_split_and_are_ordered():
    from riskplatform.config import SPLIT
    o = origins(SPLIT, 4, 30)
    assert o[-1] == SPLIT and [s.valid_end_day for s in o] == sorted(s.valid_end_day for s in o)
    assert all(s.train_end_day - s.label_delay_days > s.warmup_days for s in o)


def test_backtest_folds_use_only_earlier_data(engine):
    data, _ = make_dataset(engine, SMALL_SPLIT)
    r = run_backtest(data, SMALL_SPLIT, n_origins=2, step_days=15, test_days=25, seed=3)
    assert r["n_folds"] == 2 and "SYNTHETIC" in r["disclaimer"]
    for f in r["folds"]:
        assert f["train_days"][1] + SMALL_SPLIT.label_delay_days <= f["origin_day"]   # training labels matured before the origin
        assert f["test_days"][0] >= f["origin_day"] and f["test_days"][1] < f["origin_day"] + 25
        assert 0 <= f["precision"] <= 1 and 0 <= f["recall"] <= 1
    assert r["folds"][0]["origin_day"] < r["folds"][1]["origin_day"]


def test_slice_report_flags_blind_spot_and_respects_min_support():
    rng = np.random.default_rng(0)
    n = 20000
    seg = rng.choice(["a", "b", "tiny"], n, p=[0.5, 0.49, 0.01])
    y = rng.random(n) < 0.05
    alert = np.where(seg == "b", y & (rng.random(n) < 0.2), y & (rng.random(n) < 0.9))   # model misses most fraud in "b"
    alert = alert | (~y & (rng.random(n) < 0.01))
    df = pd.DataFrame({"segment": seg, "is_fraud": y.astype(int), "amount": rng.uniform(1, 700, n)})
    r = slice_report(df, alert, ["segment", "amount_bucket"])
    flagged = {(f["by"], f["slice"]) for f in r["flagged"]}
    assert ("segment", "b") in flagged and ("segment", "a") not in flagged
    assert ("segment", "tiny") not in flagged                      # too few frauds -> never flagged
    assert {x["slice"] for x in r["slices"]["amount_bucket"]} <= {"<25", "25-100", "100-500", "500+"}


def test_calibration_summary_perfect_vs_constant():
    rng = np.random.default_rng(1)
    p = rng.random(50000) * 0.2
    y = rng.random(50000) < p
    c = _calibration_summary(y, p)
    assert c["ece"] < 0.01 and c["brier"] < c["brier_baseline_constant"]
    assert _calibration_summary(y, np.full(len(y), 0.9) + rng.random(len(y)) * 1e-6)["ece"] > 0.5


def test_fingerprint_is_order_independent_and_content_sensitive(tables):
    tx = tables["transactions"]
    assert data_fingerprint(tx) == data_fingerprint(tx.sample(frac=1, random_state=0))
    t2 = tx.copy(); t2.loc[t2.index[0], "amount"] += 0.01
    assert data_fingerprint(tx) != data_fingerprint(t2)


def test_pipeline_writes_manifest_slices_and_no_truth_leak(engine, tmp_path):
    m = step_train_evaluate(engine, SMALL_SPLIT, tmp_path, seed=5)
    man = json.loads((tmp_path / "run_manifest.json").read_text())
    assert len(man["data_sha256"]) == 64 and man["headline_synthetic_metrics"]["synthetic"] is True and man["packages"]["pandas"]
    assert set(m["slice_performance"]["slices"]) == {"segment", "channel", "category", "amount_bucket", "history"}
    assert "brier" in m["calibration_summary"]
    assert build_manifest(pd.DataFrame({"txn_id": [1]}), 1, {})["config_sha256"]
