"""Validate an ML-derived managed rule against positive/negative replay cases."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waf.core.config import WAFConfig
from waf.core.models import RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.features.http_v2 import ProductionHTTPFeatureExtractor


def main() -> int:
    env = {"WAF_ENV": "development", "WAF_AUTH_SECRET": "phase10-rule-replay-secret-" + "x" * 32}
    waf = EdgeWAF(WAFConfig.from_env(env))
    extractor = ProductionHTTPFeatureExtractor()

    # Use a source request whose structured feature vector contains the same
    # allowlisted feature that the positive replay request will exercise.
    request = RequestEnvelope(
        "phase10-rule-source",
        "GET",
        "https",
        "replay.example",
        "/search",
        "q=select%20*%20from%20users",
        body=b"",
        source_ip="192.0.2.10",
    )
    result = waf.analyze(request)
    recommendations = waf.recommend_rules(result)
    if not recommendations:
        raise AssertionError("expected at least one ML-derived recommendation")
    rule = recommendations[0]
    validation = waf.validate_rule(rule.rule_id)
    if not validation.valid:
        raise AssertionError(validation.errors)

    positive = RequestEnvelope("phase10-positive", "GET", "https", "replay.example", "/search", "q=select * from users")
    negative = RequestEnvelope("phase10-negative", "GET", "https", "replay.example", "/search", "q=hello-world")
    pos_features = extractor.extract(positive)
    neg_features = extractor.extract(negative)
    matches_positive = rule.rule_id in waf.rule_lifecycle.match(pos_features)
    matches_negative = rule.rule_id in waf.rule_lifecycle.match(neg_features)
    if not matches_positive or matches_negative:
        raise AssertionError(f"replay mismatch: positive={matches_positive}, negative={matches_negative}")

    evidence = {
        "rule_id": rule.rule_id,
        "rule_name": rule.name,
        "matcher": dict(rule.matcher),
        "source": rule.source,
        "source_detector": rule.source_detector,
        "validation": {"valid": validation.valid, "errors": list(validation.errors)},
        "positive_example": {"query_profile": "sql-like-select", "matched": matches_positive},
        "negative_example": {"query_profile": "benign", "matched": matches_negative},
        "decision_source_request": result.request_id,
        "evidence_schema": result.evidence.schema_version if result.evidence else None,
        "scope": "deterministic synthetic replay corpus; not a production corpus",
    }
    out = ROOT / "phase10_rule_replay_evidence.json"
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
