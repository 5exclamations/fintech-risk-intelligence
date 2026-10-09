import pytest

from riskplatform.analytics import list_queries, run_all


def test_all_queries_run_and_return_rows(engine, tmp_path):
    res = run_all(engine, tmp_path)
    assert len(res) == len(list_queries()) >= 10
    for name, df in res.items():
        assert len(df) > 0, name


def test_frequency_percentages_sum_to_100(engine, tmp_path):
    df = run_all(engine, tmp_path)["01_transaction_frequency"]
    assert df.pct_of_customer_months.sum() == pytest.approx(100, abs=0.1)


def test_sql_flags_agree_with_labels(engine, tmp_path):
    s = run_all(engine, tmp_path)["06_unusual_behavior_summary"]
    base = s[(s.amount_spike == 0) & (s.velocity_burst == 0)].fraud_rate_pct.iloc[0]
    burst = s[s.velocity_burst == 1].fraud_rate_pct.max()
    assert burst > 10 * base
