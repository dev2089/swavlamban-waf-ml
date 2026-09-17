# Phase 1 Test Report

## Executed gates

- Python compileall: PASS
- Unit tests: 13/13 PASS
- Feature normalization/range tests: PASS
- Real signature detector test: PASS
- Allow/alert/block pipeline tests: PASS
- Body-size bound test: PASS
- Schema mismatch fail-closed test: PASS
- Invalid detector signal test: PASS
- Secret-like content scan over new core: PASS
- External-I/O scan over new core: PASS
- Runtime/import test: PASS
- Mixed fuzz benchmark: PASS

## Benchmark

100,000 mixed randomized HTTP-like requests completed in approximately 4.19 seconds on the terminal test environment, about 23,882 decisions/sec. This is **not** an end-to-end WAF throughput claim.

## Score

**10.0 / 10.0** for the Phase 1 acceptance criteria.

This score does not transfer to the final challenge submission until every later phase has executable evidence.
