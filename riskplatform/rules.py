"""Rule-based detection. Rules are transparent, cheap, and double as human-readable reason codes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

RISKY_CATEGORY_CODES = [8, 9, 4]  # gift_cards, crypto_exchange, electronics (indices into features.CATEGORIES)


@dataclass(frozen=True)
class Rule:
    code: str
    weight: float  # prior confidence that a hit is fraud (hand-set; NOT fitted) - used for a noisy-OR score
    text: str
    fn: Callable[[pd.DataFrame], pd.Series]


def _b(x) -> pd.Series:
    return pd.Series(x).fillna(False).astype(bool)


RULES: list[Rule] = [
    Rule("R01_VELOCITY_1H", 0.55, "4+ transactions by this customer in the previous hour",
         lambda f: _b(f.cnt_1h >= 4)),
    Rule("R02_CARD_TESTING", 0.70, "3+ micro-payments (<$5) in the previous 30 minutes",
         lambda f: _b(f.small_cnt_30m >= 3)),
    Rule("R03_AMOUNT_SPIKE", 0.35, "amount is 5x+ the customer's historical average (and >= $100)",
         lambda f: _b((f.cust_prior_n >= 10) & (f.amount_to_mean >= 5) & (f.amount >= 100))),
    Rule("R04_IMPOSSIBLE_TRAVEL", 0.60, "implied travel speed from the previous transaction exceeds 900 km/h",
         lambda f: _b((f.dist_from_last_km >= 500) & (f.implied_speed_kmh >= 900))),
    Rule("R05_NEW_DEVICE_LARGE", 0.45, "first use of this device with an amount 3x+ the customer's average",
         lambda f: _b((f.new_device == 1) & (f.amount_to_mean >= 3))),
    Rule("R06_RISKY_MERCHANT", 0.30, "merchant's historical (matured-label) fraud rate is >= 5%",
         lambda f: _b(f.merch_fraud_rate_matured >= 0.05)),
    Rule("R07_UNUSUAL_NIGHT", 0.20, "night-time purchase >= $50 from a customer who rarely transacts at night",
         lambda f: _b((f.is_night == 1) & (f.cust_night_share < 0.05) & (f.cust_prior_n >= 10) & (f.amount >= 50))),
    Rule("R08_FOREIGN_ONLINE_NEW_COUNTRY", 0.35, "online transaction from a country never seen for this customer",
         lambda f: _b((f.is_foreign == 1) & (f.is_online == 1) & (f.new_country == 1))),
    Rule("R09_NEW_DEVICE_RISKY_CATEGORY", 0.40, "new device buying gift cards / crypto / electronics",
         lambda f: _b((f.new_device == 1) & f.category_code.isin(RISKY_CATEGORY_CODES))),
]
RULE_INDEX = {r.code: r for r in RULES}


def evaluate_rules(f: pd.DataFrame) -> pd.DataFrame:
    """Boolean hit matrix, one column per rule (aligned to f.index)."""
    out = pd.DataFrame({r.code: r.fn(f).values for r in RULES}, index=f.index)
    return out


def rule_score(hits: pd.DataFrame) -> np.ndarray:
    """Noisy-OR combination: P(at least one firing rule is 'right') = 1 - prod(1 - w_i)."""
    w = np.array([RULE_INDEX[c].weight for c in hits.columns])
    return 1.0 - np.prod(np.where(hits.values, 1.0 - w[None, :], 1.0), axis=1)


def rule_reasons(hits_row: pd.Series) -> list[dict]:
    return [{"code": c, "source": "rule", "detail": RULE_INDEX[c].text} for c in hits_row.index[hits_row.values]]
