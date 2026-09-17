# Quick Start

This guide covers the current Phase 10 WAF implementation. Historical phase experiments remain under `docs/` and `handoff/` for audit continuity.

## Prerequisites

- Python 3.11+
- Git
- Linux for the verified process-level nginx/ModSecurity evidence path
- For local WAF enforcement evidence: nginx, ModSecurity nginx connector, curl and OpenSSL

## Install

```bash
./setup.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the API/dashboard

```bash
python backend/server.py
```

Open `/dashboard` after configuring authentication/environment variables.

## Run the canonical gateway

```bash
python -m waf.gateway.proxy
```

Allowed requests are forwarded to the configured upstream. Blocked requests terminate at the WAF gateway with HTTP 403.

## Test

```bash
python -m pytest -q
python -m compileall -q waf tests scripts
```

## Produce challenge evidence

```bash
python scripts/phase10_demo.py
python scripts/phase10_rule_replay.py
python scripts/phase10_tls_e2e.py
python scripts/phase10_waf_enforcement_e2e.py
python scripts/phase10_load_harness.py --url http://127.0.0.1:18081/ --duration 2 --concurrency 20 --rate 100 --payload-profile mixed
python scripts/phase10_master_exam.py
```

## Repository layout

```text
waf/                    canonical runtime
  api/                  authenticated API/dashboard adapter
  core/                 configuration and domain models
  detection/            decision/risk components
  edge/                 WAF pipeline and reverse proxy
  gateway/              request gateway
  ml/                   anomaly/classification/learning controls
  rules/                managed-rule lifecycle
  security/             auth/RBAC/secrets controls
  storage/              bounded telemetry and persistence

deploy/nginx/            process-level nginx + ModSecurity configuration
dashboard/               authenticated operator dashboard
scripts/                 deterministic demos, E2E tests, load harness, release gates
tests/                   regression and integration tests
handoff/                 audit history, claim ledger, negative evidence, manifests
docs/                    current and historical technical documentation
```

## Evidence boundary

Do not treat a local bounded benchmark as proof of Internet-scale capacity. Do not treat the synthetic CI ML corpus as field accuracy. Public certificate issuance/rotation and public Internet HTTPS verification remain outside the demonstrated free CI boundary.

## Future continuation

Read `WAF_PROJECT_STATE.json` and `handoff/START_HERE.md` first, then the Phase 10 result/evidence files and requirement traceability before changing the system.
