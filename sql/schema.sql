-- Portable DDL (PostgreSQL 13+ and SQLite 3.35+). All data is SYNTHETIC.
-- Derived convenience columns on `transactions` (txn_date, hour_local, dow_local, ts_epoch) are
-- materialised so the analytical SQL stays dialect-neutral (no EXTRACT / strftime / interval syntax).

DROP TABLE IF EXISTS synthetic_truth_merchants;
DROP TABLE IF EXISTS synthetic_truth;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS merchants;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS locations;

CREATE TABLE locations (
    location_id  INTEGER PRIMARY KEY,
    city         VARCHAR(64)  NOT NULL,
    country      CHAR(2)      NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    utc_offset   DOUBLE PRECISION NOT NULL,
    is_domestic  INTEGER      NOT NULL
);

CREATE TABLE customers (
    customer_id       VARCHAR(10) PRIMARY KEY,
    segment           VARCHAR(32) NOT NULL,   -- generator persona, not a model output
    age               INTEGER,
    home_location_id  INTEGER REFERENCES locations(location_id),
    signup_date       TIMESTAMP NOT NULL,
    kyc_risk_rating   VARCHAR(8)
);

CREATE TABLE accounts (
    account_id    VARCHAR(10) PRIMARY KEY,
    customer_id   VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    account_type  VARCHAR(16) NOT NULL,
    opened_at     TIMESTAMP   NOT NULL,
    credit_limit  DOUBLE PRECISION
);

CREATE TABLE merchants (
    merchant_id  VARCHAR(10) PRIMARY KEY,
    name         VARCHAR(80) NOT NULL,
    category     VARCHAR(32) NOT NULL,
    location_id  INTEGER REFERENCES locations(location_id),
    is_online    INTEGER NOT NULL,
    onboarded_at TIMESTAMP NOT NULL
);

CREATE TABLE transactions (
    txn_id       VARCHAR(12) PRIMARY KEY,
    account_id   VARCHAR(10) NOT NULL REFERENCES accounts(account_id),
    customer_id  VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    merchant_id  VARCHAR(10) NOT NULL REFERENCES merchants(merchant_id),
    location_id  INTEGER     NOT NULL REFERENCES locations(location_id), -- where the txn happened (POS city / IP city)
    ts           TIMESTAMP   NOT NULL,        -- UTC
    txn_date     VARCHAR(10) NOT NULL,        -- UTC date, 'YYYY-MM-DD'
    hour_local   INTEGER     NOT NULL,        -- 0-23 at the transaction location
    dow_local    INTEGER     NOT NULL,        -- 0 = Monday
    ts_epoch     BIGINT      NOT NULL,        -- seconds since 1970-01-01 UTC
    amount       DOUBLE PRECISION NOT NULL,
    channel      VARCHAR(16) NOT NULL,        -- card_present | online | atm
    device_id    VARCHAR(16),                 -- NULL for card-present / ATM
    is_fraud     INTEGER     NOT NULL         -- chargeback label (noisy, delayed; see docs)
);

-- Generator ground truth. Deliberately separate: it must never be joined into features.
CREATE TABLE synthetic_truth (
    txn_id        VARCHAR(12) PRIMARY KEY REFERENCES transactions(txn_id),
    true_pattern  VARCHAR(24) NOT NULL,
    is_fraud_true INTEGER     NOT NULL
);
CREATE TABLE synthetic_truth_merchants (
    merchant_id    VARCHAR(10) PRIMARY KEY REFERENCES merchants(merchant_id),
    is_compromised BOOLEAN NOT NULL
);

CREATE INDEX ix_txn_customer_ts ON transactions (customer_id, ts_epoch);
CREATE INDEX ix_txn_merchant_ts ON transactions (merchant_id, ts_epoch);
CREATE INDEX ix_txn_date        ON transactions (txn_date);
