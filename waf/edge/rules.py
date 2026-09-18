from __future__ import annotations

import re

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope
from waf.features.http_v2 import normalize_target


class OpenSourceWAFRuleEngine:
    """Deterministic rule layer consumed by the live edge."""

    name = "open-source-waf-rules"

    RULES = (
        (
            "WAF-SQL-001",
            re.compile(
                r"(?:union\s+select|or\s+\d\s*=\s*\d|drop\s+table)",
                re.I,
            ),
            "SQL injection signature",
        ),
        (
            "WAF-XSS-001",
            re.compile(r"<\s*script\b|javascript\s*:", re.I),
            "XSS signature",
        ),
        (
            "WAF-TRAV-001",
            re.compile(r"(?:\.\./|\.\.\\)", re.I),
            "path traversal signature",
        ),
        (
            "WAF-CMD-001",
            re.compile(
                r"(?:;\s*(?:cat|id|uname|curl|wget)\b|\$\(|\x60[^\x60]+\x60)",
                re.I,
            ),
            "command injection signature",
        ),
    )

    _FEATURE_RULES = (
        ("has_sql_keyword", "WAF-SQL-001", "SQL injection signature"),
        ("has_xss_token", "WAF-XSS-001", "XSS signature"),
        ("has_traversal", "WAF-TRAV-001", "path traversal signature"),
        ("has_command_token", "WAF-CMD-001", "command injection signature"),
    )

    def detect(
        self,
        request: RequestEnvelope,
        features: FeatureVector,
    ) -> DetectionSignal:
        target = normalize_target(request.path, request.query)
        target += "\n" + request.body.decode("utf-8", errors="replace")

        reasons: list[str] = []
        rule_ids: list[str] = []

        for rule_id, pattern, reason in self.RULES:
            if pattern.search(target):
                rule_ids.append(rule_id)
                reasons.append(reason)

        for feature_name, rule_id, reason in self._FEATURE_RULES:
            if features.values.get(feature_name, 0.0) >= 1.0 and rule_id not in rule_ids:
                rule_ids.append(rule_id)
                reasons.append(reason)

        score = 1.0 if rule_ids else 0.0
        confidence = 0.99 if rule_ids else 0.10
        return DetectionSignal(
            detector=self.name,
            score=score,
            confidence=confidence,
            reasons=tuple(dict.fromkeys(reasons)),
            rule_ids=tuple(dict.fromkeys(rule_ids)),
            metadata={"ruleset": "builtin-open-source-waf-v2"},
        )
