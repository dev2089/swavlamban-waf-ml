from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Mapping


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ALERT = "alert"


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """Normalized request representation crossing the security boundary."""

    request_id: str
    method: str
    scheme: str
    host: str
    path: str
    query: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    source_ip: str | None = None
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    schema_version: str
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DetectionSignal:
    detector: str
    score: float
    confidence: float
    reasons: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision: Decision
    risk_score: float
    reasons: tuple[str, ...]
    rule_ids: tuple[str, ...]
    signals: tuple[DetectionSignal, ...]
    request_id: str
    pipeline_version: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")
