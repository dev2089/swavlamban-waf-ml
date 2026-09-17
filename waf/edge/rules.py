from __future__ import annotations

import re
from urllib.parse import unquote_plus

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope


class OpenSourceWAFRuleEngine:
    """Deterministic repository-owned open-source WAF rule layer."""

    name = "open-source-waf-rules"
    RULES = (
        ("WAF-SQL-001", re.compile(r"(?:union\s+select|or\s+\d\s*=\s*\d|drop\s+table)", re.I), "SQL injection signature"),
        ("WAF-XSS-001", re.compile(r"<\s*script\b|javascript\s*:", re.I), "XSS signature"),
        ("WAF-TRAV-001", re.compile(r"(?:\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)", re.I), "path traversal signature"),
        ("WAF-CMD-001", re.compile(r"(?:;\s*(?:cat|id|uname|curl|wget)\b|\$\(|`[^`]+`)", re.I), "command injection signature"),
    )

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        target = (request.path + "?" + request.query if request.query else request.path)
        target += "\n" + request.body.decode("utf-8", errors="replace")
        target = unquote_plus(target)
        reasons: list[str] = []
        rule_ids: list[str] = []
        for rule_id, pattern, reason in self.RULES:
            if pattern.search(target):
                rule_ids.append(rule_id)
                reasons.append(reason)
        score = 1.0 if rule_ids else 0.0
        confidence = 0.99 if rule_ids else 0.10
        return DetectionSignal(
            detector=self.name,
            score=score,
            confidence=confidence,
            reasons=tuple(reasons),
            rule_ids=tuple(rule_ids),
            metadata={"ruleset": "builtin-open-source-waf-v1"},
        )
