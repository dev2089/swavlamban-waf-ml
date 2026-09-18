from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

def utc_now() -> datetime: return datetime.now(timezone.utc)

class DecisionAction(str, Enum):
    ALLOW = "allow"
    ALERT = "alert"
    BLOCK = "block"

@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    received_at: datetime
    method: str
    scheme: str
    host: str
    path: str
    query: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = None
    source_ip: str | None = None
    user_agent: str | None = None
    def __post_init__(self) -> None:
        if not self.request_id.strip(): raise ValueError("request_id is required")
        if not self.method.strip(): raise ValueError("method is required")
        if not self.path.startswith("/"): raise ValueError("path must begin with '/'")
        object.__setattr__(self, "method", self.method.upper())
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))

@dataclass(frozen=True, slots=True)
class DetectionSignal:
    detector: str
    score: float
    reasons: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not self.detector.strip(): raise ValueError("detector is required")
        if not 0.0 <= float(self.score) <= 1.0: raise ValueError("score must be between 0 and 1")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "rule_ids", tuple(self.rule_ids))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

@dataclass(frozen=True, slots=True)
class Decision:
    action: DecisionAction
    risk_score: float
    signals: tuple[DetectionSignal, ...] = ()
    reasons: tuple[str, ...] = ()
    latency_ms: float | None = None
    policy_version: str = "phase1-contract"
    def __post_init__(self) -> None:
        if not 0.0 <= float(self.risk_score) <= 1.0: raise ValueError("risk_score must be between 0 and 1")
        if self.latency_ms is not None and self.latency_ms < 0: raise ValueError("latency_ms cannot be negative")
        object.__setattr__(self, "risk_score", float(self.risk_score))
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "reasons", tuple(self.reasons))

@dataclass(frozen=True, slots=True)
class PipelineState:
    features: Mapping[str, float] = field(default_factory=dict)
    signals: tuple[DetectionSignal, ...] = ()
    decision: Decision | None = None
    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType({str(k): float(v) for k,v in self.features.items()}))
    def with_features(self, values: Mapping[str, float]) -> "PipelineState":
        merged = dict(self.features); merged.update({str(k): float(v) for k,v in values.items()})
        return PipelineState(merged, self.signals, self.decision)
    def with_signals(self, values: Sequence[DetectionSignal]) -> "PipelineState":
        return PipelineState(self.features, self.signals + tuple(values), self.decision)
    def with_decision(self, decision: Decision) -> "PipelineState":
        return PipelineState(self.features, self.signals, decision)

@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event_type: str
    request_id: str
    timestamp: datetime = field(default_factory=utc_now)
    payload: Mapping[str, object] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not self.event_type.strip(): raise ValueError("event_type is required")
        if not self.request_id.strip(): raise ValueError("request_id is required")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))