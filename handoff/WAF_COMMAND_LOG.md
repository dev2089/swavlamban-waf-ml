# WAF Command / Evidence Log

## Phase 1
- `python -m compileall -q waf tests` -> PASS.
- `python -m unittest discover -s tests -v` -> 13/13 PASS.
- `scripts/phase1_self_test.py` -> PASS.
- 100,000 mixed randomized HTTP-like requests -> ~23,882 decisions/sec.

## Phase 2
- Python compileall -> PASS.
- Phase 2 tests -> 6/6 PASS.
- Nginx syntax/integration -> PASS: allow=200, block=403.
- 5,000-request local E2E benchmark -> 3,477.8 req/s; 4,500 allow; 500 block; 1,000 bounded events.
- Static secret/dangerous-I/O/TODO scans -> PASS.

## Phase 3
- `python -m compileall -q waf tests` -> PASS.
- Full regression -> 34/34 PASS.
- First feature cycle -> FAIL: percent-ratio `TypeError` plus over-specific Unicode-path assertion.
- Remediation -> fixed ratio calculation and corrected semantics-preserving test.
- Header-boundary follow-up -> PASS after limiting normalized header processing to 128 headers.
- Self-test runner -> PASS.
- Randomized feature fuzz -> 20,000 requests, 0 exceptions.
- Latest feature benchmark -> 100,000 extractions, 33,527.0 req/s; 38 features.
- Latest E2E proxy benchmark -> 5,000 requests, 4,500 allow, 500 block, 2,764.7 req/s, 1,000 bounded events.
- Static secret-like scan -> PASS.
- New Phase 3 TODO/FIXME/pass-only scan -> PASS.

## Phase 4
- Stale model artifact cycle -> FAIL; fixed by artifact regeneration and complete-component validation.
- Narrow benign baseline cycle -> FAIL; fixed by broadening benign workload variation and retraining.
- Oversized mandatory self-test -> FAIL by execution-time boundary; fixed by bounded mandatory gate plus separate extended benchmark.
- Learned-behaviour artifact integration -> FAIL; fixed by persisting the learned behavioural component.
- Behavioural serialization -> FAIL because thread lock/window runtime state was not pickle-safe; fixed with explicit serialization state.
- Behavioural state isolation/name regression -> FAIL; fixed by fresh per-edge state and stable `behaviour-v1` name.
- Master-exam import-path cycle -> FAIL; fixed by repository-root path setup.
- `python -m pytest -q` -> **48/48 PASS**.
- `python -m compileall -q waf tests` -> **PASS**.
- `python scripts/phase4_master_exam.py` -> **PASS, 10.0/10.0, 0 critical defects**.
- `python phase4_self_test.py` -> **PASS**.
- Supervised: accuracy 1.0, precision 1.0, recall 1.0, F1 1.0, FPR 0.0, n=1500; synthetic only.
- Unsupervised: FPR 0.0227, attack detection 0.8980, benign n=750, attack n=3000; synthetic only.
- Learned behaviour: normal max 0.406313, burst final 0.980486, escalation=true; synthetic workload only.
- Final direct ML rerun: 2,000 requests, **529.6 req/s**, 1,900 allow, 100 block, 0 alert.
- Final bounded E2E rerun: 1,000 requests, **400.2 req/s**, 850 HTTP 200, 150 HTTP 403, 0 errors, p50 64.240 ms, p95 81.686 ms.
- Retained extended E2E evidence: 5,000 requests, 410.4 req/s, 4,250 HTTP 200, 750 HTTP 403, 0 errors, p50 123.614 ms, p95 160.959 ms, 4,250 upstream hits, 1,000 bounded events.
- Static secret-like scan -> PASS.
- TODO/no-op scan -> PASS.
- Artifact SHA-256 -> `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.

## Final handoff verification
- Final workspace rerun after all documentation synchronization: `48/48 PASS`, compileall PASS, master exam PASS, self-test PASS.
- Final workspace and handoff DB preserve prior failed cycles and remediation history.
- All metrics are local terminal evidence. They are not production-capacity guarantees or real-world Internet WAF accuracy claims.
