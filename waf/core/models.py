from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Mapping

class Decision(str,Enum):
    ALLOW="allow"; BLOCK="block"; ALERT="alert"

@dataclass(frozen=True,slots=True)
class RequestEnvelope:
    request_id:str; method:str; scheme:str; host:str; path:str; query:str=""; headers:Mapping[str,str]=field(default_factory=dict); body:bytes=b""; source_ip:str|None=None; timestamp:float=field(default_factory=time)

@dataclass(frozen=True,slots=True)
class FeatureVector:
    schema_version:str; values:Mapping[str,float]

@dataclass(frozen=True,slots=True)
class DetectionSignal:
    detector:str; score:float; confidence:float; reasons:tuple[str,...]=(); rule_ids:tuple[str,...]=(); metadata:Mapping[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        if not 0<=self.score<=1: raise ValueError("score must be in [0, 1]")
        if not 0<=self.confidence<=1: raise ValueError("confidence must be in [0, 1]")

@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    schema_version: str
    decision: str
    risk_score: float
    detector_contributions: tuple[Mapping[str, Any], ...]
    feature_groups: Mapping[str, Mapping[str, Any]]
    feature_attribution: Mapping[str, Mapping[str, float]]
    reasons: tuple[str, ...]
    rule_ids: tuple[str, ...]
    versions: Mapping[str, Any]
    explanation: str
    privacy: Mapping[str, Any]
    request_id: str
    feature_snapshot: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "evidence-v1":
            raise ValueError("unsupported evidence schema")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")
        for key in ("raw_payload_retained", "raw_query_retained", "raw_headers_retained"):
            if self.privacy.get(key) is not False:
                raise ValueError(f"{key} must be false")
        for value in self.feature_snapshot.values():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("feature snapshot value must be in [0, 1]")
        if len(self.feature_snapshot) != 40:
            raise ValueError("evidence requires exactly 40 features")

@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision:Decision; risk_score:float; reasons:tuple[str,...]; rule_ids:tuple[str,...]; signals:tuple[DetectionSignal,...]; request_id:str; pipeline_version:str; evidence: DecisionEvidence | None = None
    def __post_init__(self):
        if not 0<=self.risk_score<=1: raise ValueError("risk_score must be in [0, 1]")
