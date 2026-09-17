# Challenge Requirement Mapping

| Need | Current status after Phase 4 | Planned completion |
|---|---|---|
| ML integrated with open-source WAF | PASS for Phase 4 seam: live edge consumes signature + supervised + unsupervised + behavioural ML | final validation later |
| HTTP(S) analysis | PASS for live HTTP + `http-v2`; TLS termination remains open | later |
| traffic baselining | PARTIAL: benign-only ML baseline exists; production traffic baseline lifecycle remains open | 7 |
| behavioural analysis | PASS for learned synthetic behavioural workload; production distributed state remains open | 7/9 |
| anomaly detection | PASS: benign-only `OneClassSVM` is live in the ML ensemble | later validation |
| dashboard | LEGACY/DEFERRED | 11 |
| supervised/unsupervised/semi-supervised ML | SUPERVISED + UNSUPERVISED PASS; semi-supervised not introduced in Phase 4 | optional later |
| explainability | FOUNDATION reasons/rule IDs + detector metadata; full attribution/evidence expansion remains open | 5 |
| rule recommendation | foundation only | 6 |
| low latency/high throughput | Measured local evidence; full load matrix remains open | 9 |
| continuous learning/retraining | NOT YET | 7 |
| logs/metrics/reports | event schema + bounded edge events; production telemetry pipeline remains open | 9-13 |
| demo | NOT YET | 12 |
| technical docs/slides | milestone docs exist; final submission package remains open | 13 |

## Phase 4 additions

- Supervised `HistGradientBoostingClassifier` over the `http-v2` representation.
- Benign-only `OneClassSVM` with threshold learned from benign baseline data.
- Learned stateful behavioural `LogisticRegression` over per-source sliding-window features.
- Versioned model artifact with all three components.
- Live `EdgeWAF` integration consuming signature + all three ML signals.
- Reproducible supervised, unsupervised and behavioural evaluation evidence.
- Artifact round-trip, live-edge, regression, fuzz, performance and master-exam gates.

## Critical honesty boundary
A requirement is not marked final-complete merely because code exists. Final challenge status requires executable evidence for the final integrated capability. Synthetic ML metrics are milestone reproducibility evidence only, not real-world Internet WAF accuracy claims.
