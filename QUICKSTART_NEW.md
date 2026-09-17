# Quick Start - Phase 10

This repository contains the Swavlamban WAF ML Challenge 3 implementation. The canonical runtime is the `waf/` package. `backend/server.py` is a compatibility FastAPI entry point and `run_proxy.py` launches the edge proxy.

## Requirements

- Python 3.11+
- Linux is the verified CI/runtime environment
- Git
- For the process-level open-source WAF demonstration: nginx, the ModSecurity nginx connector, curl and OpenSSL
- A Supabase project is required only for persistent production control-plane telemetry; the local security decision path does not require a synchronous database write.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and provide the required runtime configuration. Never commit real secrets.

## Run the API/dashboard

```bash
python backend/server.py
```

The API serves the authenticated operator dashboard at `/dashboard`.

## Run the WAF gateway

```bash
python -m waf.gateway.proxy
```

The gateway inspects requests before forwarding allowed traffic to the configured upstream. Blocked requests receive HTTP 403 and are not forwarded.

## Run tests

```bash
python -m pytest -q
python -m compileall -q waf tests scripts
```

## Evidence-producing commands

```bash
python scripts/phase10_demo.py
python scripts/phase10_rule_replay.py
python scripts/phase10_tls_e2e.py
python scripts/phase10_waf_enforcement_e2e.py
python scripts/phase10_load_harness.py --url http://127.0.0.1:18081/ --duration 2 --concurrency 20 --rate 100 --payload-profile mixed
python scripts/phase10_master_exam.py
```

## What is actually measured

The release evidence includes deterministic local WAF enforcement, local TLS termination through nginx, managed-rule replay validation, bounded load measurements and the dashboard/demo path. The CI gate is intentionally explicit about the measurement boundary.

## What is not claimed

The project does not claim physical Internet-scale million-request capacity, public certificate issuance/rotation or field-traffic ML accuracy from the synthetic CI corpus. Those boundaries are recorded in `handoff/PHASE10_NEGATIVE_EVIDENCE.md`.

## Release evidence

Read `phase10_master_exam_result.json`, `phase10_*_evidence.json`, `handoff/PHASE10_REQUIREMENT_TRACEABILITY.json`, `handoff/PHASE10_CLAIM_LEDGER.md`, and `handoff/PHASE10_NEGATIVE_EVIDENCE.md` before treating any feature or metric as verified.
