from __future__ import annotations

from dataclasses import replace
import os

from waf.core.config import WAFConfig
from waf.core.models import DecisionResult, RequestEnvelope
from waf.edge.policy import EdgeDecisionPolicy
from waf.edge.rules import OpenSourceWAFRuleEngine
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.ensemble import load_runtime


class EdgeWAF:
    """Phase 4 live security path: signatures + supervised + anomaly + behaviour."""

    def __init__(self, config: WAFConfig) -> None:
        self.config = config
        self.features = ProductionHTTPFeatureExtractor()
        self.signature_detector = OpenSourceWAFRuleEngine()
        artifact = os.getenv("WAF_MODEL_ARTIFACT", "models/phase4_models.joblib")
        self.ml = load_runtime(artifact)
        self.policy = EdgeDecisionPolicy(config.block_threshold, config.alert_threshold)

    def analyze(self, request: RequestEnvelope) -> DecisionResult:
        if len(request.body) > self.config.max_body_bytes:
            request = replace(request, body=request.body[: self.config.max_body_bytes])
        features = self.features.extract(request)
        if features.schema_version != self.config.feature_schema_version:
            raise ValueError(
                f"feature schema mismatch: expected {self.config.feature_schema_version}, got {features.schema_version}"
            )
        signature = self.signature_detector.detect(request, features)
        ml_signals = self.ml.detect(request, features)
        return self.policy.decide(request, (signature, *ml_signals), self.config.pipeline_version)
