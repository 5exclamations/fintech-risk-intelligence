-- 02 Daily volume + fraud-label rate with a trailing 7-day average.
SELECT txn_date,
       COUNT(*)                                    AS n_txn,
       SUM(amount)                                 AS total_amount,
       SUM(is_fraud)                               AS n_fraud,
       1.0 * SUM(is_fraud) / COUNT(*)              AS fraud_rate,
       AVG(1.0 * COUNT(*)) OVER (ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS n_txn_7d_avg
FROM transactions
GROUP BY txn_date
ORDER BY txn_date;
