from __future__ import annotations

from dataclasses import replace

from waf.core.config import WAFConfig
from waf.core.models import DecisionResult, RequestEnvelope
from waf.edge.policy import EdgeDecisionPolicy
from waf.edge.rules import OpenSourceWAFRuleEngine
from waf.features.http import HTTPFeatureExtractor


class EdgeWAF:
    """Phase 2 security path used by the live reverse proxy."""

    def __init__(self, config: WAFConfig) -> None:
        self.config = config
        self.features = HTTPFeatureExtractor()
        self.detector = OpenSourceWAFRuleEngine()
        self.policy = EdgeDecisionPolicy(config.block_threshold, config.alert_threshold)

    def analyze(self, request: RequestEnvelope) -> DecisionResult:
        if len(request.body) > self.config.max_body_bytes:
            request = replace(request, body=request.body[: self.config.max_body_bytes])
        features = self.features.extract(request)
        if features.schema_version != self.config.feature_schema_version:
            raise ValueError(
                f"feature schema mismatch: expected {self.config.feature_schema_version}, got {features.schema_version}"
            )
        signal = self.detector.detect(request, features)
        return self.policy.decide(request, (signal,), self.config.pipeline_version)
