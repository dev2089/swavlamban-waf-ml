from __future__ import annotations

from dataclasses import replace

from .config import WAFConfig
from .contracts import DecisionPolicy, Detector, EventSink, FeatureExtractor
from .models import DecisionResult, RequestEnvelope
from waf.telemetry.events import decision_event


class WAFPipeline:
    """Single security decision seam shared by future proxy, API and tests."""

    def __init__(
        self,
        config: WAFConfig,
        feature_extractor: FeatureExtractor,
        policy: DecisionPolicy,
        detectors: tuple[Detector, ...] = (),
        event_sink: EventSink | None = None,
    ) -> None:
        self.config = config
        self.feature_extractor = feature_extractor
        self.policy = policy
        self.detectors = detectors
        self.event_sink = event_sink

    def analyze(self, request: RequestEnvelope) -> DecisionResult:
        if not request.request_id:
            raise ValueError("request_id is required")
        if len(request.body) > self.config.max_body_bytes:
            request = replace(request, body=request.body[: self.config.max_body_bytes])
        features = self.feature_extractor.extract(request)
        if features.schema_version != self.config.feature_schema_version:
            raise ValueError(
                f"feature schema mismatch: expected {self.config.feature_schema_version}, got {features.schema_version}"
            )
        signals = tuple(detector.detect(request, features) for detector in self.detectors)
        result = self.policy.decide(request, signals, self.config.pipeline_version)
        if self.event_sink is not None:
            self.event_sink.publish(decision_event(result))
        return result
