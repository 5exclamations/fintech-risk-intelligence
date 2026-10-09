-- 04 Behavioural customer segmentation (RFM-style, quartiles via NTILE) joined to the label.
-- "As-of" date = day after the last transaction. Analytical (descriptive) use only - NOT a model feature.
WITH asof AS (SELECT MAX(ts_epoch) AS max_ts FROM transactions),
cust AS (
    SELECT t.customer_id,
           COUNT(*)                                     AS frequency,
           SUM(t.amount)                                AS monetary,
           (a.max_ts - MAX(t.ts_epoch)) / 86400.0       AS recency_days,
           SUM(t.is_fraud)                              AS n_fraud
    FROM transactions t CROSS JOIN asof a
    GROUP BY t.customer_id, a.max_ts
),
scored AS (
    SELECT c.*,
           NTILE(4) OVER (ORDER BY frequency)     AS f_q,
           NTILE(4) OVER (ORDER BY monetary)      AS m_q,
           NTILE(4) OVER (ORDER BY recency_days DESC) AS r_q
    FROM cust c
)
SELECT CASE WHEN f_q + m_q + r_q >= 11 THEN 'champions'
            WHEN f_q + m_q + r_q >= 8  THEN 'loyal'
            WHEN r_q <= 1              THEN 'dormant'
            ELSE 'occasional' END              AS rfm_segment,
       COUNT(*)                                AS customers,
       ROUND(CAST(AVG(frequency) AS NUMERIC), 1)                AS avg_txns,
       ROUND(CAST(AVG(monetary) AS NUMERIC), 0)                 AS avg_spend,
       ROUND(CAST(AVG(recency_days) AS NUMERIC), 1)             AS avg_recency_days,
       ROUND(CAST(100.0 * SUM(n_fraud) / SUM(frequency) AS NUMERIC), 3) AS fraud_txn_rate_pct,
       ROUND(CAST(100.0 * AVG(CASE WHEN n_fraud > 0 THEN 1.0 ELSE 0.0 END) AS NUMERIC), 2) AS pct_customers_with_fraud
FROM scored
GROUP BY 1
ORDER BY customers DESC;
