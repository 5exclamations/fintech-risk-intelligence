"""Reproducibility of the reported evaluation metrics.

1. Same seed twice -> numerically identical metrics (determinism of generator, features, training, evaluation).
2. Small-config run matches a committed golden file within tolerance (guards against silent behaviour changes;
   tolerance absorbs cross-platform / library-version float noise). Refresh deliberately: UPDATE_GOLDEN=1 pytest tests/test_reproducibility.py
3. RISK_FULL_REPRO=1: the full default pipeline reproduces the headline numbers quoted in README / model card.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import SMALL_SPLIT

from riskplatform.pipeline import step_train_evaluate

GOLDEN = Path(__file__).parent / "golden" / "small_metrics.json"
GBM = "Gradient boosting (calibrated)"


def summary(m: dict) -> dict:
    op = m["test_operating_point"]
    return {"pr_auc": m["methods_test"][GBM]["pr_auc"], "roc_auc": m["methods_test"][GBM]["roc_auc"], "precision": op["precision"],
            "recall": op["recall"], "f1": op["f1"], "fpr": op["false_positive_rate"], "alert_rate": op["alert_rate"],
            "tp": op["tp"], "fp": op["fp"], "fn": op["fn"], "tn": op["tn"], "review_threshold": m["model"]["thresholds"]["review"],
            "saving_pct": m["cost"]["saving_pct"], "n_test": m["splits"]["test"]["rows"]}


@pytest.fixture(scope="module")
def run1(engine, tmp_path_factory):
    return step_train_evaluate(engine, SMALL_SPLIT, tmp_path_factory.mktemp("a"), seed=11)


def test_same_seed_is_deterministic(engine, run1, tmp_path):
    run2 = step_train_evaluate(engine, SMALL_SPLIT, tmp_path, seed=11)
    a, b = summary(run1), summary(run2)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12), k


def test_matches_golden_within_tolerance(run1):
    s = summary(run1)
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(s, indent=2))
    g = json.loads(GOLDEN.read_text())
    for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "fpr", "alert_rate", "saving_pct"):
        assert s[k] == pytest.approx(g[k], abs=0.03), (k, s[k], g[k])
    for k in ("tp", "fp", "fn", "tn"):
        assert s[k] == pytest.approx(g[k], rel=0.10, abs=5), (k, s[k], g[k])
    assert s["n_test"] == g["n_test"]
    assert s["review_threshold"] == pytest.approx(g["review_threshold"], abs=0.03)


def test_metrics_internal_consistency(run1):
    op = run1["test_operating_point"]
    assert op["tp"] + op["fn"] == round(run1["splits"]["test"]["fraud_rate"] * run1["splits"]["test"]["rows"])
    assert op["precision"] == pytest.approx(op["tp"] / (op["tp"] + op["fp"]))
    assert op["recall"] == pytest.approx(op["tp"] / (op["tp"] + op["fn"]))
    lo, hi = op["pr_auc_ci95"]
    assert lo <= run1["methods_test"][GBM]["pr_auc"] <= hi + 0.05
    assert "SYNTHETIC" in run1["disclaimer"]


@pytest.mark.skipif(not os.environ.get("RISK_FULL_REPRO"), reason="full-scale run (~1 min); set RISK_FULL_REPRO=1")
def test_full_pipeline_reproduces_reported_headline_numbers(tmp_path):
    """Numbers quoted in README.md / docs/model_card.md (synthetic data, seed 42)."""
    from riskplatform.config import SPLIT
    from riskplatform.db import get_engine
    from riskplatform.pipeline import step_generate
    eng = get_engine(f"sqlite:///{tmp_path / 'full.db'}")
    step_generate(eng)
    s = summary(step_train_evaluate(eng, SPLIT, tmp_path / "art", seed=42))
    assert s["n_test"] == 103469
    assert s["pr_auc"] == pytest.approx(0.730, abs=0.01)
    assert s["precision"] == pytest.approx(0.802, abs=0.02)
    assert s["recall"] == pytest.approx(0.683, abs=0.02)
    assert s["f1"] == pytest.approx(0.738, abs=0.02)
    assert s["fpr"] == pytest.approx(0.0021, abs=0.0006)
    assert np.isclose(s["saving_pct"], 55.7, atol=3.0)
