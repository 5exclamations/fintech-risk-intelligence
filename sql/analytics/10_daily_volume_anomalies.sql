-- 10 Time-based anomalies, part 2: days whose volume deviates > 3 sigma from the trailing 28-day baseline
-- (baseline excludes the day itself). Variance via E[x^2] - E[x]^2 to stay portable (no STDDEV in SQLite).
WITH d AS (SELECT txn_date, COUNT(*) AS n_txn, SUM(is_fraud) AS n_fraud FROM transactions GROUP BY txn_date),
b AS (
    SELECT txn_date, n_txn, n_fraud,
           COUNT(*)           OVER w AS m,
           AVG(1.0 * n_txn)   OVER w AS mu,
           AVG(1.0 * n_txn * n_txn) OVER w AS mu2
    FROM d
    WINDOW w AS (ORDER BY txn_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING)
)
SELECT txn_date, n_txn, n_fraud, ROUND(CAST(mu AS NUMERIC), 1) AS baseline_mean,
       ROUND(CAST((n_txn - mu) / sqrt(mu2 - mu * mu) AS NUMERIC), 2) AS z_score
FROM b
WHERE m >= 14 AND mu2 - mu * mu > 0 AND ABS((n_txn - mu) / sqrt(mu2 - mu * mu)) > 3
ORDER BY ABS((n_txn - mu) / sqrt(mu2 - mu * mu)) DESC;
