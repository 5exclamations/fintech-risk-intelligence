import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskplatform.config import GenConfig, SplitConfig  # noqa: E402
from riskplatform.db import get_engine, load_tables  # noqa: E402
from riskplatform.generate import generate  # noqa: E402

SMALL_GEN = GenConfig(seed=7, n_customers=500, n_merchants=250, days=150, target_txn_per_customer_day=0.45, n_ato=40, n_card_testing=20,
                      n_stolen_card=40, n_low_and_slow=100, n_gift_card_cashout=25, drift_day=115)
SMALL_SPLIT = SplitConfig(label_delay_days=14, warmup_days=20, train_end_day=85, valid_end_day=115)


@pytest.fixture(scope="session")
def tables():
    return generate(SMALL_GEN)


@pytest.fixture(scope="session")
def engine(tables, tmp_path_factory):
    eng = get_engine(f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}")
    load_tables(eng, tables)
    return eng
