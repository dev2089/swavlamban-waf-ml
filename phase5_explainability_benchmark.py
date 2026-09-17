from __future__ import annotations

import statistics
import time

from waf.core.config import WAFConfig
from waf.core.models import RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.explainability import build_decision_evidence


def main() -> None:
    waf = EdgeWAF(WAFConfig())
    requests = [
        RequestEnvelope(
            f"bench-{i}", "GET", "https", "example.test", "/api/item/1", f"page={i % 10}",
            source_ip=f"10.0.0.{(i % 20) + 1}", timestamp=1000.0 + i,
        )
        for i in range(100)
    ]

    core: list[float] = []
    evidence: list[float] = []
    for request in requests:
        start = time.perf_counter()
        features = waf.features.extract(request)
        signature = waf.signature_detector.detect(request, features)
        signals = waf.ml.detect(request, features)
        result = waf.policy.decide(request, (signature, *signals), waf.config.pipeline_version)
        core.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        build_decision_evidence(request, features, result, waf.ml)
        evidence.append((time.perf_counter() - start) * 1000.0)

    core_mean = statistics.mean(core)
    evidence_mean = statistics.mean(evidence)
    print({
        "samples": len(requests),
        "core_mean_ms": round(core_mean, 3),
        "explanation_mean_ms": round(evidence_mean, 3),
        "explanation_overhead_percent_of_core": round((evidence_mean / max(core_mean, 0.001)) * 100.0, 2),
        "scope": "local deterministic benchmark; not a production latency claim",
    })


if __name__ == "__main__":
    main()
