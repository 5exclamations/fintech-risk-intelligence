"""Central configuration. Every tunable assumption lives here so it is documented in one place."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("RISK_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("RISK_DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.environ.get("RISK_ARTIFACT_DIR", ROOT / "artifacts"))
SQL_DIR = ROOT / "sql"
DEFAULT_DB_URL = f"sqlite:///{DATA_DIR / 'risk.db'}"


def db_url() -> str:
    """PostgreSQL in docker-compose (DATABASE_URL), SQLite file locally / in unit tests."""
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


@dataclass(frozen=True)
class GenConfig:
    seed: int = 42
    n_customers: int = 3000
    n_merchants: int = 700
    start: str = "2025-01-01"
    days: int = 365
    target_txn_per_customer_day: float = 0.38  # mean over customers
    # Synthetic fraud-event counts (events, not transactions; one event emits several transactions)
    n_ato: int = 150
    n_card_testing: int = 60
    n_stolen_card: int = 150
    n_low_and_slow: int = 420
    n_gift_card_cashout: int = 90  # only emitted after `drift_day`
    drift_day: int = 300  # new fraud pattern + holiday amount shift begin here
    compromised_merchant_share: float = 0.04
    label_noise_missed_fraud: float = 0.05  # fraud never discovered -> labelled legit
    label_noise_false_dispute: float = 0.0015  # legit labelled fraud (friendly fraud / disputes)


@dataclass(frozen=True)
class SplitConfig:
    """Time-based split. Gaps >= label delay so no training label depends on a test-period outcome."""
    label_delay_days: int = 14  # a chargeback label is only known this long after the transaction
    warmup_days: int = 30  # first days only provide history (cold-start) and are never trained/evaluated on
    train_end_day: int = 230  # T1: the model is trained "as of" this day
    valid_end_day: int = 290  # T2: threshold/calibration are fitted "as of" this day; test = [T2, end)
    # train      = [warmup, T1 - delay)   labels of every training row were mature at T1
    # validation = [T1, T2 - delay)       labels mature at T2; used ONLY for calibration + threshold + model selection
    # test       = [T2, end)              never touched until final evaluation


@dataclass(frozen=True)
class CostConfig:
    """Business-cost assumptions (illustrative, NOT real-world figures)."""
    review_cost: float = 6.0  # analyst cost of manually reviewing one alert (~6 min @ $60/h)
    friction_cost: float = 3.0  # customer-friction cost per false alert (declined/step-up auth)
    fraud_fixed_cost: float = 25.0  # chargeback handling fee per missed fraud
    loss_given_missed: float = 1.0  # share of the amount actually lost when fraud is missed
    review_minutes_per_alert: float = 6.0
    analyst_hours_per_day: float = 6.5  # productive hours per analyst per day


GEN = GenConfig()
SPLIT = SplitConfig()
COST = CostConfig()
