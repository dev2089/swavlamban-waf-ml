# Phase 4 Execution Log

## Control point
- Base milestone: Phase 3 `phase3-final`.
- Authoritative branch: `phase4-final`.
- Terminal lab: `/mnt/data/waf-phase4`.
- Main branch intentionally untouched.

## Work performed
1. Added supervised `HistGradientBoostingClassifier` over `http-v2`.
2. Added benign-only `OneClassSVM` anomaly detection with threshold learned from benign baseline scores.
3. Rebuilt behaviour detection as a learned stateful model: per-source sliding-window statistics scored by `LogisticRegression`.
4. Added deterministic labelled HTTP dataset and varied benign-only baseline generation.
5. Persisted all three learned/stateless model components in a versioned `joblib` artifact.
6. Added artifact schema/manifest validation and fresh per-edge behavioural state.
7. Wired signature + supervised + unsupervised + behavioural signals into the live `EdgeWAF`.
8. Added supervised, anomaly, behaviour, artifact, live-edge, regression, fuzz and benchmark gates.
9. Added executable Phase 4 master exam and final handoff state.
10. Extended the portable SQLite project ledger with artifact and environment evidence.

## Failed cycles and remediation

### Cycle A - stale artifact
The detector implementation changed while an older artifact was still present. Runtime failed on the missing behaviour component. The artifact was retrained and explicit artifact component validation was added.

### Cycle B - benign false-positive regression
An overly narrow benign baseline caused ordinary requests to alert. Benign generation was widened across hosts, agents, Accept headers, JSON bodies, content types and query patterns; the models were retrained and retested.

### Cycle C - oversized mandatory self-test
A larger one-command benchmark exceeded the terminal execution-time boundary. The mandatory gate was bounded to deterministic 2,000 direct requests plus a 1,000-request E2E gate, while the 5,000-request benchmark remained a separate extended evidence run.

### Cycle D - learned behaviour artifact compatibility
The new learned behaviour component was initially absent from the old persisted artifact. The artifact was rebuilt and artifact presence/type assertions were added.

### Cycle E - thread-lock serialization
The live behavioural detector initially attempted to pickle its thread lock and window state. Explicit `__getstate__`/`__setstate__` now serialize only learned model/configuration and recreate runtime state after loading.

### Cycle F - behavioural state isolation/name regression
Caching the whole behavioural detector shared live state across runtimes and a detector name changed unexpectedly. The final design shares the immutable learned classifier but creates fresh behavioural state per `EdgeWAF`; the stable detector name is `behaviour-v1`.

### Cycle G - master-exam import path
The master-exam script initially failed to import the repository package. Repository-root insertion into `sys.path` fixed the harness and the complete gate passed.

## Final executable evidence
- `python -m pytest -q` -> **48/48 PASS**.
- `python -m compileall -q waf tests` -> **PASS**.
- `python scripts/phase4_master_exam.py` -> **PASS, 10.0/10.0, 0 critical defects**.
- `python phase4_self_test.py` -> **PASS**.
- Supervised holdout: accuracy `1.0`, precision `1.0`, recall `1.0`, F1 `1.0`, FPR `0.0`, n=`1500`, synthetic only.
- Unsupervised: FPR `0.0227`, attack detection `0.8980`, benign n=`750`, attack n=`3000`, synthetic only.
- Learned behaviour: normal max `0.406313`, burst final `0.980486`, escalation `true`, synthetic workload only.
- Direct ML final self-test: `2000 requests, 538.0 req/s, 1900 allow, 100 block, 0 alert`.
- Mandatory bounded E2E final self-test: `1000 requests, 449.4 req/s, 850 HTTP 200, 150 HTTP 403, 0 errors, p50 57.310 ms, p95 68.851 ms`.
- Final extended E2E: `5000 requests, 410.4 req/s, 4250 HTTP 200, 750 HTTP 403, 0 errors, p50 123.614 ms, p95 160.959 ms, 4250 upstream hits, 1000 events`.
- Artifact: `199123` bytes, SHA-256 `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.
- Static secret-like scan -> **PASS**.
- TODO/no-op scan -> **PASS**.

## Final handoff verification
The final handoff ZIP contains the verified Phase 4 workspace, challenge source PDFs, expanded Phase 3 history, SQLite project ledger, tests, documentation, model artifact and continuation instructions. The archive was extracted into a clean directory and successfully ran 48/48 tests, compileall, master exam and self-test.

## Honest boundary
Phase 4 is complete for the defined milestone only. All ML evaluation data/workloads are deterministic synthetic evidence. They are not real-world Internet WAF accuracy claims. Overall Challenge 3 remains IN_PROGRESS.
