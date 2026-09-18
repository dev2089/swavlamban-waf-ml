# Dependency Rules

| Rule | Requirement |
|---|---|
| Core isolation | waf.core uses only Python standard library. |
| Fast-path purity | Request-time stages perform no durable I/O. |
| Slow-path isolation | Storage/analytics/training cannot change an already-issued request decision. |
| Versioned contracts | Features, evidence, rules and models carry schema/version metadata before persistence. |
| Security boundary | Administrative and rule-management operations require authentication/authorization in later phases. |
| Explicit dependencies | Runtime components receive explicit configuration/dependencies. |
| Reproducibility | Metrics are produced by executable tests/benchmarks. |
| Observability | Security decisions are traceable to request ID and evidence. |
| Privacy | Sensitive material is minimized/redacted before durable storage. |
| Horizontal scaling | Fast-path components remain stateless unless bounded state is explicitly justified. |