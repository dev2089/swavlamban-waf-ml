# Next Execution Step: Phase 3

Build the production HTTP feature pipeline on top of the live Phase 2 edge.

## Phase 3 goal
Replace the small baseline feature set with a versioned, security-focused HTTP feature schema covering request structure, payload characteristics, encodings, headers, methods, content types, URL normalization, and safe bounded body parsing without placing database/network work on the security fast path.

## Phase 3 acceptance
- deterministic feature extraction from real intercepted HTTP requests;
- explicit schema version and feature manifest;
- URL/path/query normalization with encoded attack handling;
- bounded parsing and malformed-input tests;
- no secrets/PII unnecessarily retained in feature vectors;
- regression tests against Phase 2 enforcement;
- measurable extraction latency benchmark;
- 9.9+ self-exam with critical-defect fail rule.
