"""End-to-end proof that TRAINING cannot see future information or labels before they are known.

Each test perturbs data that must be invisible to the model and requires bit-for-bit identical results where it matters.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import SMALL_SPLIT
from sqlalchemy import text

from riskplatform.modeling import make_dataset, split_masks, train_all

D = SMALL_SPLIT.label_delay_days
T1, T2 = SMALL_SPLIT.train_end_day, SMALL_SPLIT.valid_end_day


@pytest.fixture(scope="module")
def base(engine):
    with engine.connect() as con:
        tx = pd.read_sql_query(text("SELECT * FROM transactions ORDER BY ts_epoch, txn_id"), con)
    data, _ = make_dataset(engine, SMALL_SPLIT, tx=tx)
    model, ev, _ = train_all(data, SMALL_SPLIT, seed=3)
    return tx, data, model, ev


def _flip(tx, mask):
    t = tx.copy()
    t.loc[mask, "is_fraud"] = 1 - t.loc[mask, "is_fraud"]
    return t


def _retrain(engine, tx):
    data, _ = make_dataset(engine, SMALL_SPLIT, tx=tx)
    model, ev, _ = train_all(data, SMALL_SPLIT, seed=3)
    return data, model, ev


def _day(tx):
    return ((tx.ts_epoch - (tx.ts_epoch.min() // 86400) * 86400) // 86400).values


def test_labels_not_yet_mature_at_training_cutoff_are_unused(engine, base):
    """Flip every label with day >= T1 - delay (unknown when the model is trained at T1). The complete TRAINING INPUT
    (features and labels) must be identical, hence an identically-seeded model is identical."""
    from riskplatform.modeling import make_hgb
    tx, data, model, _ = base
    pert = _flip(tx, _day(tx) >= T1 - D)
    data2, _ = make_dataset(engine, SMALL_SPLIT, tx=pert)
    tr = split_masks(data.day, SMALL_SPLIT)["train"]
    assert tr.sum() > 1000
    pd.testing.assert_frame_equal(data.loc[tr, model.features], data2.loc[tr, model.features])
    np.testing.assert_array_equal(data.loc[tr, "is_fraud"].values, data2.loc[tr, "is_fraud"].values)
    m1 = make_hgb(5).fit(data.loc[tr, model.features], data.loc[tr, "is_fraud"])
    m2 = make_hgb(5).fit(data2.loc[tr, model.features], data2.loc[tr, "is_fraud"])
    np.testing.assert_array_equal(m1.predict_proba(data.loc[tr, model.features]), m2.predict_proba(data2.loc[tr, model.features]))


def test_test_period_labels_never_influence_model_calibration_or_thresholds(engine, base):
    tx, _, model, ev = base
    pert = _flip(tx, _day(tx) >= T2)
    _, model2, ev2 = _retrain(engine, pert)
    assert model2.thresholds == model.thresholds
    va, va2 = ev[ev.split == "valid"], ev2[ev2.split == "valid"]
    np.testing.assert_array_equal(va.p_model.values, va2.p_model.values)
    # a test row at day t may legitimately use labels of rows with day <= t - delay, so only the first `delay` test days are
    # guaranteed not to see any (flipped) test-period label
    te, te2 = ev[(ev.split == "test") & (ev.day < T2 + D)], ev2[(ev2.split == "test") & (ev2.day < T2 + D)]
    assert len(te) > 500
    np.testing.assert_array_equal(te.p_model.values, te2.p_model.values)


def test_future_rows_do_not_change_training_features(engine, base):
    """Dropping every transaction after the training cutoff must not change features of rows before it."""
    tx, data, model, _ = base
    early = tx[_day(tx) < T1]
    data2, _ = make_dataset(engine, SMALL_SPLIT, tx=early)
    keep = data.txn_id.isin(data2.txn_id)
    a = data[keep].set_index("txn_id")[model.features].sort_index()
    b = data2.set_index("txn_id")[model.features].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_split_boundaries_enforce_label_maturity(base):
    _, data, _, _ = base
    m = split_masks(data.day, SMALL_SPLIT)
    assert data.day[m["train"]].max() <= T1 - D - 1
    assert data.day[m["valid"]].min() >= T1 and data.day[m["valid"]].max() <= T2 - D - 1
    assert data.day[m["test"]].min() >= T2
    assert data.day[m["train"]].min() >= SMALL_SPLIT.warmup_days


def test_ground_truth_columns_absent_from_model_inputs(base):
    _, data, model, _ = base
    banned = {"true_pattern", "is_fraud_true", "is_fraud", "pattern", "is_compromised"}
    assert not banned & set(model.features)
