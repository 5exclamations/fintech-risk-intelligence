import numpy as np
from conftest import SMALL_GEN

from riskplatform.generate import generate


def test_referential_integrity(tables):
    tx = tables["transactions"]
    assert tx.txn_id.is_unique
    assert tx.customer_id.isin(tables["customers"].customer_id).all()
    assert tx.account_id.isin(tables["accounts"].account_id).all()
    assert tx.merchant_id.isin(tables["merchants"].merchant_id).all()
    assert tx.location_id.isin(tables["locations"].location_id).all()
    acc = tables["accounts"].set_index("account_id").customer_id
    assert (acc.reindex(tx.account_id).values == tx.customer_id.values).all()


def test_fraud_rate_is_realistic_and_labels_are_noisy(tables):
    tx, truth = tables["transactions"], tables["synthetic_truth"]
    assert 0.003 < tx.is_fraud.mean() < 0.03
    m = tx.merge(truth, on="txn_id")
    assert ((m.is_fraud == 0) & (m.is_fraud_true == 1)).any()      # undiscovered fraud
    assert ((m.is_fraud == 1) & (m.is_fraud_true == 0)).any()      # false disputes


def test_ground_truth_not_in_transactions(tables):
    assert not {"pattern", "true_pattern", "is_fraud_true", "is_compromised"} & set(tables["transactions"].columns)


def test_time_ordered_ids_and_amounts(tables):
    tx = tables["transactions"]
    assert tx.ts.is_monotonic_increasing and (tx.amount > 0).all()
    assert (tx.ts_epoch.diff().dropna() >= 0).all()


def test_deterministic():
    a, b = generate(SMALL_GEN), generate(SMALL_GEN)
    assert np.array_equal(a["transactions"].amount.values, b["transactions"].amount.values)
    assert a["transactions"].is_fraud.sum() == b["transactions"].is_fraud.sum()


def test_new_pattern_only_after_drift(tables):
    tx = tables["transactions"].merge(tables["synthetic_truth"], on="txn_id")
    g = tx[tx.true_pattern == "gift_card_cashout"]
    assert len(g) and (g.ts >= tx.ts.min() + np.timedelta64(SMALL_GEN.drift_day, "D")).all()
