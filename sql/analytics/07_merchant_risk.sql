-- 07 Merchant-level risk: smoothed (empirical-Bayes) fraud rate so tiny merchants are not over-ranked.
-- smoothed = (fraud + k * prior) / (n + k); prior = portfolio fraud rate, k = pseudo-count (default 200).
-- NOTE: descriptive analytics over ALL labels. The ML feature uses only labels already *mature* at scoring time.
-- Parameters: :k, :min_txn, :limit
WITH base AS (SELECT 1.0 * SUM(is_fraud) / COUNT(*) AS prior FROM transactions),
mm AS (
    SELECT t.merchant_id, COUNT(*) AS n_txn, SUM(t.is_fraud) AS n_fraud, SUM(t.amount) AS volume,
           SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END) AS fraud_amount
    FROM transactions t GROUP BY t.merchant_id
)
SELECT m.merchant_id, m.name, m.category, m.is_online, mm.n_txn, mm.n_fraud,
       ROUND(CAST(100.0 * mm.n_fraud / mm.n_txn AS NUMERIC), 2)                                    AS raw_fraud_rate_pct,
       ROUND(CAST(100.0 * (mm.n_fraud + :k * b.prior) / (mm.n_txn + :k) AS NUMERIC), 2)            AS smoothed_fraud_rate_pct,
       ROUND(CAST(mm.fraud_amount AS NUMERIC), 0)                                                  AS fraud_amount,
       RANK() OVER (ORDER BY (mm.n_fraud + :k * b.prior) / (mm.n_txn + :k) DESC) AS risk_rank
FROM mm JOIN merchants m ON m.merchant_id = mm.merchant_id CROSS JOIN base b
WHERE mm.n_txn >= :min_txn
ORDER BY risk_rank
LIMIT :limit;
