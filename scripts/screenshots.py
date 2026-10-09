"""Capture dashboard screenshots into docs/screenshots (dev-only; needs `pip install playwright` + Chrome)."""
import sys

from playwright.sync_api import sync_playwright

TABS = ["overview", "risk_distribution", "suspicious_transactions", "fraud_trends", "model_performance", "false_positives",
        "workload_cost", "monitoring", "sql_analytics"]
url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1500, "height": 1300})
    pg.goto(url); pg.wait_for_timeout(7000)
    for i, name in enumerate(TABS):
        pg.get_by_role("tab").nth(i).click(); pg.wait_for_timeout(2500)
        pg.screenshot(path=f"docs/screenshots/{i + 1:02d}_{name}.png", full_page=False)
    b.close()
