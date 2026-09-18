from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from waf.core.models import DetectionSignal


@dataclass(frozen=True, slots=True)
class ResponseEnvelope:
    status: int
    headers: Mapping[str, str]
    body: bytes = b""


class OutboundResponseFeatureExtractor:
    schema_version = "http-response-v1"

    @staticmethod
    def _entropy(body: bytes) -> float:
        if not body:
            return 0.0
        counts = np.bincount(np.frombuffer(body, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(body)
        return float(-np.sum(probs * np.log2(probs)))

    def extract(self, response: ResponseEnvelope) -> tuple[str, tuple[float, ...]]:
        body_lower = response.body.lower()
        content_type = str(response.headers.get("Content-Type", "")).lower()
        header_keys = {str(k).lower() for k in response.headers}
        names = (
            "status",
            "body_length_log",
            "body_entropy",
            "header_count",
            "content_type_html",
            "content_type_json",
            "set_cookie_present",
            "script_token_count",
            "stack_trace_token_count",
            "error_token_count",
            "location_length_log",
        )
        values = (
            float(max(0, min(599, response.status))) / 599.0,
            math.log1p(min(len(response.body), 10_485_760)),
            self._entropy(response.body),
            min(len(header_keys), 128) / 128.0,
            1.0 if "html" in content_type else 0.0,
            1.0 if "json" in content_type else 0.0,
            1.0 if "set-cookie" in header_keys else 0.0,
            float(body_lower.count(b"<script")),
            float(sum(body_lower.count(token) for token in (b"traceback", b"stack trace", b"exception", b"at java."))),
            float(sum(body_lower.count(token) for token in (b"internal server error", b"debug", b"secret", b"password="))),
            math.log1p(min(len(str(response.headers.get("Location", ""))), 8192)),
        )
        return names, values


@dataclass(slots=True)
class OutboundAnomalyDetector:
    feature_names: tuple[str, ...]
    model: object
    decision_threshold: float
    baseline_version: str
    name: str = "outbound-oneclasssvm-v1"

    @classmethod
    def train_default(cls, samples: int = 1600, seed: int = 42) -> "OutboundAnomalyDetector":
        rng = random.Random(seed)
        extractor = OutboundResponseFeatureExtractor()
        rows: list[list[float]] = []
        names: tuple[str, ...] | None = None
        for _ in range(max(128, samples)):
            status = rng.choice((200, 200, 200, 201, 204, 304))
            content_type = rng.choice(("application/json", "text/html", "text/plain"))
            size = rng.randint(20, 8000)
            if content_type == "application/json":
                body = (b'{"ok":true,"items":[' + b"0," * rng.randint(1, 15) + b"1]}")[:size]
            elif content_type == "text/html":
                body = (b"<html><body><h1>OK</h1><p>item</p></body></html>")[:size]
            else:
                body = (b"OK\n" + (b"item " * rng.randint(1, 1000)))[:size]
            headers = {"Content-Type": content_type}
            if rng.random() < 0.18:
                headers["Cache-Control"] = "no-store"
            if rng.random() < 0.08:
                headers["Set-Cookie"] = "session=opaque"
            names, values = extractor.extract(ResponseEnvelope(status, headers, body))
            names = tuple(names)
            rows.append(list(values))
        X = np.asarray(rows, dtype=float)
        pipeline = make_pipeline(
            StandardScaler(),
            OneClassSVM(kernel="rbf", gamma="scale", nu=0.025),
        )
        pipeline.fit(X)
        decision_scores = pipeline.decision_function(X)
        threshold = float(np.quantile(decision_scores, 0.025))
        return cls(names or (), pipeline, threshold, f"outbound-baseline-v1-s{len(rows)}-seed{seed}")

    def detect(self, response: ResponseEnvelope) -> DetectionSignal:
        extractor = OutboundResponseFeatureExtractor()
        _, values = extractor.extract(response)
        raw = float(self.model.decision_function(np.asarray([values], dtype=float))[0])
        scale = max(abs(self.decision_threshold), 1e-6)
        margin = (self.decision_threshold - raw) / scale
        score = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, 8.0 * margin))))
        score = min(1.0, max(0.0, score))
        confidence = min(1.0, abs(score - 0.5) * 2.0)
        reasons: list[str] = []
        body = response.body.lower()
        if b"<script" in body:
            reasons.append("outbound signal: script-bearing response")
        if any(token in body for token in (b"traceback", b"exception", b"stack trace")):
            reasons.append("outbound signal: stack-trace/error disclosure")
        if any(token in body for token in (b"password=", b"secret")):
            reasons.append("outbound signal: sensitive-token pattern")
        if len(response.body) > 1_000_000:
            reasons.append("outbound signal: unusually large response body")
        return DetectionSignal(
            detector=self.name,
            score=round(score, 6),
            confidence=round(confidence, 6),
            reasons=tuple(reasons),
            metadata={
                "feature_schema": extractor.schema_version,
                "baseline_version": self.baseline_version,
                "direction": "outbound",
                "raw_response_retained": False,
            },
        )

    def evaluate(self, responses: list[ResponseEnvelope]) -> dict[str, float | int | str]:
        signals = [self.detect(response) for response in responses]
        return {
            "samples": len(signals),
            "anomaly_rate": float(sum(s.score >= 0.5 for s in signals) / max(1, len(signals))),
            "evaluation_scope": "deterministic synthetic HTTP response workload",
        }
