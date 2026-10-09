# Business case

**Problem.** Card/transfer fraud losses grow with volume, while manual review capacity and customer patience are finite. Static rules miss new
behaviour; blanket blocking drives false declines and attrition.

**Goal.** Score every transaction at authorisation time, send only the riskiest ~1% to action, explain *why*, and prove (and keep proving) it works.

**What the platform provides**
| Capability | Value |
|---|---|
| SQL analytics (frequency, segments, merchants, time anomalies) | analysts and risk managers can answer ad-hoc questions without a model |
| Layered detection (rules → statistics → ML) | transparent baseline; ML must *earn* its place against it |
| Calibrated score + cost-based thresholds | the alert rate is a business decision (cost per review vs loss), not an accident |
| Reason codes | faster analyst triage, defensible customer communication |
| Workload & cost view | staffing: alerts/day, review hours, FTEs for any threshold |
| Monitoring | detect drift before delayed chargebacks reveal it |

**Illustrative outcome (synthetic data, invented costs)** — on the 75-day test window the model alerts on ~1.05% of transactions (~14/day),
catches 68% of fraud labels at 80% precision, and lowers total cost (review + friction + missed fraud) from $181k to $80k (−56%).
These values depend entirely on the simulator and on `CostConfig`; **real portfolios will differ — possibly drastically.**

**To use on real data you would need:** a real label feed with its true delay, authorisation-time fields (CVV/AVS/3-DS results, device & IP intelligence),
data-protection review, shadow-mode evaluation against champion rules, fairness/impact assessment, and real costs per review, decline and chargeback.
