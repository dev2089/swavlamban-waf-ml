from __future__ import annotations

from typing import Protocol, Sequence

from .models import DecisionResult, DetectionSignal, FeatureVector, RequestEnvelope


class FeatureExtractor(Protocol):
    schema_version: str

    def extract(self, request: RequestEnvelope) -> FeatureVector: ...


class Detector(Protocol):
    name: str

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal: ...


class DecisionPolicy(Protocol):
    def decide(
        self,
        request: RequestEnvelope,
        signals: tuple[DetectionSignal, ...],
        pipeline_version: str,
    ) -> DecisionResult: ...


class EventSink(Protocol):
    def publish(self, event: dict) -> None: ...


class RuleProvider(Protocol):
    def active_rule_ids(self) -> Sequence[str]: ...
