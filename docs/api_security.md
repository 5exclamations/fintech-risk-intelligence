# API security (demo-grade)

The scoring API is a **synthetic-data demonstration, not a production fraud decision system** (banner in `/docs`, `X-Demo-Notice` response header, `notice` field in every score).

| Control | Behaviour |
|---|---|
| Authentication | `X-API-Key` required on `/score`, `/score/{id}`, `/model`, `/monitoring`; `/health` is open. Constant-time comparison (`secrets.compare_digest`). |
| Fail-closed | If `RISK_API_KEY` is unset the app **refuses to start** (explicit opt-out for throw-away local use: `RISK_API_AUTH=disabled`). No default key is embedded in code; `make api` and docker-compose use a clearly-named demo key that you should override (`RISK_API_KEY=… docker compose up`). |
| Input validation | Pydantic: IDs `^[A-Za-z0-9_-]{1,16}$`, finite `0 < amount < 1e7`, enumerated channel; all DB access uses bound parameters / in-memory frames (no string-built SQL from requests). |
| Rate limiting | Per (key, client IP) sliding window, default 120 req/min (`RISK_API_RATE_LIMIT`), HTTP 429 + `Retry-After`. In-process only. |
| Response hygiene | `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`; errors never echo internals. |
| Network | docker-compose publishes ports on `127.0.0.1` only; container runs as a non-root user. |
| Label exposure | `GET /score/{id}` returns the stored label for replay/audit; it is behind the key and contains synthetic data only. |

Not implemented (needed before any real deployment): TLS termination, per-user identities / OAuth, key rotation and secret management, distributed rate limiting, audit logging, request signing, PII handling, WAF.
Tests: `tests/test_api.py` (401 without/with wrong key, open health, headers, validation rejects, 429, fail-closed startup).
