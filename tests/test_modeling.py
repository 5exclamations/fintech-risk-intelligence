import numpy as np
import pytest
from conftest import SMALL_SPLIT

from riskplatform.evaluation import at_budget, business_cost, confusion, pick_threshold_by_cost, ranking_metrics, workload
from riskplatform.modeling import make_dataset, split_masks, train_all
from riskplatform.monitoring import psi, status


@pytest.fixture(scope="module")
def trained(engine):
    data, _ = make_dataset(engine, SMALL_SPLIT)
    model, ev, info = train_all(data, SMALL_SPLIT, seed=1)
    return data, model, ev


def test_time_split_has_no_overlap_and_matured_labels(trained):
    data = trained[0]
    m = split_masks(data.day, SMALL_SPLIT)
    d = data.day.values
    assert not (m["train"] & m["valid"]).any() and not (m["valid"] & m["test"]).any()
    assert d[m["train"]].max() + SMALL_SPLIT.label_delay_days < SMALL_SPLIT.train_end_day + 1   # labels known at T1
    assert d[m["valid"]].max() + SMALL_SPLIT.label_delay_days < SMALL_SPLIT.valid_end_day + 1   # labels known at T2
    assert d[m["train"]].max() < d[m["valid"]].min() <= d[m["valid"]].max() < d[m["test"]].min()
    assert d[m["train"]].min() >= SMALL_SPLIT.warmup_days


def test_model_beats_baselines_out_of_time(trained):
    ev = trained[2]
    t = ev[ev.split == "test"]
    ap = lambda c: ranking_metrics(t.is_fraud.values, t[c].values)["pr_auc"]
    assert ap("p_model") > 3 * t.is_fraud.mean()          # far above random
    assert ap("p_model") >= ap("score_rules") - 0.02


def test_unsupervised_detectors_ignore_labels(trained):
    _, model, _ = trained
    assert set(model.detectors) == {"robust_z", "iforest"}


def test_calibrated_scores_in_unit_interval_and_thresholds_ordered(trained):
    _, model, ev = trained
    assert ev.p_model.between(0, 1).all()
    assert model.thresholds["review"] <= model.thresholds["block"]
    assert set(ev.band.unique()) <= {"allow", "review", "block"}


def test_confusion_and_cost_math():
    y = np.array([1, 1, 0, 0, 0, 1]); a = np.array([1, 0, 1, 0, 0, 1]); amt = np.array([100, 200, 10, 10, 10, 50.0])
    c = confusion(y, a)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2)
    assert c["precision"] == pytest.approx(2 / 3) and c["recall"] == pytest.approx(2 / 3)
    from riskplatform.config import CostConfig
    k = CostConfig(review_cost=5, friction_cost=2, fraud_fixed_cost=10, loss_given_missed=1.0)
    bc = business_cost(y, a, amt, k)
    assert bc["total_cost"] == pytest.approx(3 * 5 + 1 * 2 + (200 + 10))


def test_threshold_by_cost_alerts_when_model_is_informative():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.02).astype(int)
    s = np.clip(0.2 * y + rng.random(20000) * 0.8, 0, 1)
    thr, cost = pick_threshold_by_cost(y, s, np.full(20000, 100.0))
    assert np.isfinite(thr) and cost < business_cost(y, np.zeros(20000), np.full(20000, 100.0))["total_cost"]


def test_budget_and_workload():
    assert at_budget([1, 0, 0, 1], [0.9, 0.1, 0.2, 0.8], 2) == {"k": 2, "precision": 1.0, "recall": 1.0}
    w = workload(650, 10)
    assert w["alerts_per_day"] == 65 and w["analysts_needed"] > 0


def test_psi():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    assert psi(a, rng.normal(size=20000)) < 0.02
    assert status(psi(a, rng.normal(1.0, 1, 20000))) == "alert"
    assert psi(np.r_[a, [np.nan] * 100], np.r_[a, [np.nan] * 100]) < 0.01
