from __future__ import annotations

import re

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope


class SignatureDetector:
    name = "signature-v1"

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        values = features.values
        rule_ids: list[str] = []
        reasons: list[str] = []
        hits = 0
        if values["has_sql_keyword"]:
            hits += 1
            rule_ids.append("SIG-SQL-001")
            reasons.append("SQL-like keyword pattern")
        if values["has_xss_token"]:
            hits += 1
            rule_ids.append("SIG-XSS-001")
            reasons.append("script/javascript payload pattern")
        if values["has_traversal"]:
            hits += 1
            rule_ids.append("SIG-TRAVERSAL-001")
            reasons.append("path traversal pattern")
        if values["has_command_token"]:
            hits += 1
            rule_ids.append("SIG-CMD-001")
            reasons.append("command execution pattern")
        score = min(1.0, 0.75 + 0.10 * hits) if hits else 0.0
        return DetectionSignal(self.name, score, 1.0 if hits else 0.0, tuple(reasons), tuple(rule_ids))


class RegexSignatureDetector(SignatureDetector):
    """Reserved extension point for richer tested signatures in Phase 2."""

    name = "signature-regex-v1"
    _encoded_delimiter = re.compile(r"%[0-9a-fA-F]{2}")

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        base = super().detect(request, features)
        encoded = bool(self._encoded_delimiter.search(request.query))
        if encoded and base.score == 0:
            return DetectionSignal(self.name, 0.25, 0.4, ("encoded delimiter present",), ())
        return DetectionSignal(self.name, base.score, base.confidence, base.reasons, base.rule_ids)
