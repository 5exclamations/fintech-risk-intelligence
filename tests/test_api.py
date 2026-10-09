import pytest
from conftest import SMALL_SPLIT
from fastapi.testclient import TestClient

from riskplatform.api import create_app
from riskplatform.modeling import make_dataset, train_all


@pytest.fixture(scope="module")
def client_and_data(engine):
    data, _ = make_dataset(engine, SMALL_SPLIT)
    model, ev, _ = train_all(data, SMALL_SPLIT, seed=1)
    with TestClient(create_app(engine, model)) as c:
        yield c, ev


def test_replay_matches_batch_scores(client_and_data):
    """No training/serving skew: online score for a stored txn equals its offline score."""
    c, ev = client_and_data
    for _, r in ev[ev.split == "test"].sample(6, random_state=0).iterrows():
        out = c.get(f"/score/{r.txn_id}").json()
        assert out["risk_score"] == pytest.approx(r.p_model, abs=1e-6)
        assert out["risk_band"] == r.band


def test_score_endpoint_contract(client_and_data):
    c, ev = client_and_data
    r = ev[(ev.split == "test") & (ev.band != "allow")].iloc[0]
    out = c.get(f"/score/{r.txn_id}").json()
    assert 0 <= out["risk_score"] <= 1 and out["reasons"] and "synthetic" in out["notice"].lower()
    assert all({"code", "source", "detail"} <= set(x) for x in out["reasons"])


def test_post_score_and_validation(client_and_data):
    c, _ = client_and_data
    body = {"customer_id": "C00001", "account_id": "A000001", "merchant_id": "M00001", "location_id": 1, "amount": 25.0,
            "channel": "card_present", "ts": "2025-06-01T12:00:00Z"}
    acc = c.app.state.scorer.refs["accounts"]
    body["account_id"] = acc[acc.customer_id == "C00001"].account_id.iloc[0]
    assert c.post("/score", json=body).status_code == 200
    assert c.post("/score", json={**body, "customer_id": "NOPE"}).status_code == 422
    assert c.post("/score", json={**body, "amount": -5}).status_code == 422
    assert c.get("/score/T9999999").status_code == 404


def test_burst_scores_higher_than_normal(client_and_data):
    c, ev = client_and_data
    t = ev[ev.split == "test"]
    hi = c.get(f"/score/{t.sort_values('p_model').iloc[-1].txn_id}").json()["risk_score"]
    lo = c.get(f"/score/{t.sort_values('p_model').iloc[len(t) // 2].txn_id}").json()["risk_score"]
    assert hi > lo


def test_model_endpoint(client_and_data):
    c, _ = client_and_data
    m = c.get("/model").json()
    assert m["synthetic_data"] is True and "thresholds" in m
