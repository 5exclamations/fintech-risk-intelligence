import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from riskplatform.features import NUMERIC_FEATURES, build_features


@pytest.fixture(scope="module")
def parts(engine):
    with engine.connect() as con:
        g = lambda t: pd.read_sql_query(text(f"SELECT * FROM {t}"), con)
        return g("transactions").sort_values(["ts_epoch", "txn_id"]).reset_index(drop=True), g("customers"), g("accounts"), g("merchants"), g("locations")


def feats(tx, parts_, delay=14):
    _, cu, ac, me, lo = parts_
    return build_features(tx, cu, ac, me, lo, delay).set_index("txn_id")


def test_shape_and_columns(parts):
    f = feats(parts[0], parts)
    assert len(f) == len(parts[0]) and list(f.columns) == NUMERIC_FEATURES
    assert not np.isinf(f.select_dtypes("number").values.astype(float)).any()


def test_no_future_leakage(parts):
    """Features of earlier rows are identical whether or not later rows (and row order) exist."""
    tx = parts[0]
    full = feats(tx, parts)
    cut = tx.ts_epoch.iloc[len(tx) * 2 // 3]
    sub = tx[tx.ts_epoch <= cut].sample(frac=1, random_state=3)
    part = feats(sub, parts)
    a = full.loc[part.index]
    bad = ((a - part).abs() > 1e-6) & ~(a.isna() & part.isna())
    assert not bad.any().any(), bad.sum()[bad.sum() > 0].to_dict()


def test_own_label_never_used(parts):
    tx = parts[0].copy()
    base = feats(tx, parts)
    i = tx.index[tx.is_fraud == 0][500]
    tx2 = tx.copy(); tx2.loc[i, "is_fraud"] = 1
    flipped = feats(tx2, parts)
    tid = tx.txn_id[i]
    assert np.allclose(base.loc[tid].fillna(-9).values, flipped.loc[tid].fillna(-9).values)


def test_label_maturity_delay(parts):
    """A fraud label becomes visible to the merchant-risk feature only after label_delay_days."""
    tx = parts[0].copy()
    m = tx.merchant_id.value_counts().index[0]
    rows = tx[tx.merchant_id == m]
    j = rows.index[len(rows) // 3]
    t_j = tx.ts_epoch[j]
    tx2 = tx.copy(); tx2.loc[j, "is_fraud"] = 1 - tx2.loc[j, "is_fraud"]
    a, b = feats(tx, parts, 14), feats(tx2, parts, 14)
    later = rows[rows.ts_epoch > t_j]
    soon = later[later.ts_epoch < t_j + 14 * 86400].txn_id
    after = later[later.ts_epoch >= t_j + 14 * 86400].txn_id
    assert len(soon) and len(after)
    assert np.allclose(a.loc[soon, "merch_fraud_rate_matured"], b.loc[soon, "merch_fraud_rate_matured"])
    assert not np.allclose(a.loc[after, "merch_fraud_rate_matured"], b.loc[after, "merch_fraud_rate_matured"])


def test_velocity_counts_strictly_past(parts):
    tx = parts[0]
    f = feats(tx, parts)
    c = tx.customer_id.value_counts().index[0]
    r = tx[tx.customer_id == c].reset_index(drop=True)
    row = r.iloc[50]
    expect = ((r.ts_epoch < row.ts_epoch) & (r.ts_epoch >= row.ts_epoch - 3600)).sum()
    assert f.loc[row.txn_id, "cnt_1h"] == expect
    assert f.loc[row.txn_id, "cust_prior_n"] == (r.ts_epoch < row.ts_epoch).sum()
