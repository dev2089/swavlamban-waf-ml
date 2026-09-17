# Challenge Requirement Mapping

| Need | Current status after Phase 3 | Planned completion |
|---|---|---|
| ML integrated with open-source WAF | PARTIAL: live WAF edge + ML seam; real ML still pending | 4+ |
| HTTP(S) analysis | HTTP live path + `http-v2` feature pipeline; TLS pending | 4/9 |
| traffic baselining | NOT YET | 7 |
| behavioural analysis | NOT YET | 4/7 |
| anomaly detection | SIGNATURE FOUNDATION ONLY | 4 |
| dashboard | LEGACY/DEFERRED | 11 |
| supervised/unsupervised/semi-supervised ML | NOT YET | 4 |
| explainability | reasons/rule IDs foundation | 5 |
| rule recommendation | foundation only | 6 |
| low latency/high throughput | MEASURED local fast-path/E2E evidence | 9 |
| continuous learning/retraining | NOT YET | 7 |
| logs/metrics/reports | event schema + bounded edge events | 9-13 |
| demo | NOT YET | 12 |

## Phase 3 additions

- 38-feature `http-v2` schema.
- Versioned normalization and decoding.
- Bounded query/header/body processing.
- Deterministic feature outputs in `[0,1]`.
- Live edge regression retained.
- Reproducible feature/fuzz/performance evidence recorded.

## Critical honesty boundary
This matrix describes implementation status, not marketing claims. A requirement is not marked complete until executable evidence exists for the relevant final capability.
