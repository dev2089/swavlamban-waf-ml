# Next Phase: Phase 4

Phase 3 is independently verified.

Phase 4 goal: implement real supervised, unsupervised and behavioural ML on the canonical live WAF decision seam.

Required boundaries:
- Model inference must remain on the security fast path without database writes.
- Training/evaluation must remain isolated from request-time enforcement.
- Feature/model schemas must be versioned.
- Inference latency and failure behavior must be measurable.
- Model outputs must carry confidence/evidence suitable for later explainability.

Required verification:
- unit/regression tests
- deterministic inference tests
- malformed and adversarial input tests
- model quality evaluation with held-out evidence
- live edge integration tests
- performance/latency benchmark
- failure/timeout behavior
- durable waf/database state, audit and journal updates
- 100% phase gate before declaring Phase 4 complete.