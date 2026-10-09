-- 11 Customers with the highest short-term transaction velocity (max txns inside any 1 hour).
WITH v AS (
    SELECT customer_id, txn_id, is_fraud,
           COUNT(*) OVER (PARTITION BY customer_id ORDER BY ts_epoch RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW) AS txns_in_hour
    FROM transactions
)
SELECT customer_id, MAX(txns_in_hour) AS peak_txns_1h, SUM(is_fraud) AS fraud_txns, COUNT(*) AS txns
FROM v GROUP BY customer_id
ORDER BY peak_txns_1h DESC
LIMIT :limit;
