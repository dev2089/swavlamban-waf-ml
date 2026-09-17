from __future__ import annotations

import re

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope
from waf.features.http_v2 import normalize_target


class OpenSourceWAFRuleEngine:
    """Deterministic WAF signature layer operating on the canonical target."""

    name = "open-source-waf-rules"
    RULES = (
        ("WAF-SQL-001", re.compile(r"(?:union\s+select|or\s+\d\s*=\s*\d|drop\s+table)", re.I), "SQL injection signature"),
        ("WAF-XSS-001", re.compile(r"<\s*script\b|javascript\s*:|on(?:error|load|click)\s*=", re.I), "XSS signature"),
        ("WAF-TRAV-001", re.compile(r"(?:\.\./|\.\.\\)", re.I), "path traversal signature"),
        ("WAF-CMD-001", re.compile(r"(?:;\s*(?:cat|id|uname|curl|wget|bash|sh)\b|\$\(|`[^`]+`|&&|\|\|)", re.I), "command injection signature"),
    )

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        target = normalize_target(request.path, request.query)
        target += "\n" + request.body.decode("utf-8", errors="replace")
        reasons: list[str] = []
        rule_ids: list[str] = []
        for rule_id, pattern, reason in self.RULES:
            if pattern.search(target):
                rule_ids.append(rule_id)
                reasons.append(reason)
        matched = bool(rule_ids)
        return DetectionSignal(
            detector=self.name,
            score=1.0 if matched else 0.0,
            confidence=0.99 if matched else 0.10,
            reasons=tuple(reasons),
            rule_ids=tuple(rule_ids),
            metadata={"ruleset": "builtin-open-source-waf-v2", "feature_schema": features.schema_version},
        )
