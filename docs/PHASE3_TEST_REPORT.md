# Phase 3 Independent Test Report

| Gate | Result | Evidence |
|---|---|---|
| Compile | PASS | Python compileall |
| Full regression | PASS | 36+ tests after Phase 3 additions |
| Feature schema | PASS | http-v2, 40 numeric bounded features |
| URL normalization | PASS | three-pass decode, path plus preserved |
| Header bounds | PASS | 128 headers, 4096 chars/value |
| Body bounds | PASS | 256 KiB feature scan, 1 MiB request bound |
| Query overflow | PASS | explicit overflow feature |
| Malformed input | PASS | malformed percent and invalid UTF-8 coverage |
| Raw payload retention | PASS | feature vector contains numeric values only |
| Determinism | PASS | repeated identical input yields identical vector |
| Fuzz | PASS | 20,000 randomized HTTP-like inputs, 0 extraction exceptions |
| Live Phase 2 enforcement | PASS | allow/block path remains functional through http-v2 |
| Double-encoded XSS | PASS | blocked after shared normalization |
| Oversized upstream response | PASS | bounded streaming response rejected |
| Nginx integration | PASS | local syntax + integration gate |
| Feature benchmark | PASS | 100,000 extractions measured locally |
| E2E benchmark | PASS | 5,000 requests, expected allow/block distribution |

## Evidence note
All reported performance values are local measurements in the available environment. They are evidence of behavior and relative performance, not a production-capacity guarantee for millions of requests.

## Known environment limitation
A fresh package installation in the terminal may be blocked by external network/DNS constraints. Installed compatible dependencies are acceptable for local execution; CI should remain a separate reproducibility check.

## Phase 3 gate
PASS at 100 percent for the defined Phase 3 milestone. Overall Challenge 3 remains IN_PROGRESS.