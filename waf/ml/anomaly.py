from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope


@dataclass(slots=True)
class UnsupervisedAnomalyDetector:
    feature_names: tuple[str, ...]
    model: OneClassSVM
    baseline_version: str
    decision_threshold: float
    name: str = "unsupervised-oneclasssvm-v1"

    @classmethod
    def train(cls, X_benign, feature_names, baseline_version):
        X = np.asarray(X_benign, dtype=float)
        if len(X) < 64 or not np.isfinite(X).all():
            raise ValueError("benign baseline is invalid")
        model = OneClassSVM(gamma="scale", nu=0.02)
        model.fit(X)
        threshold = float(np.percentile(model.decision_function(X), 2.0))
        return cls(feature_names, model, baseline_version, threshold)

    def score_rows(self, X):
        rows = np.asarray(X, dtype=float)
        decisions = self.model.decision_function(rows)
        return np.clip(1.0 / (1.0 + np.exp(8.0 * (decisions - self.decision_threshold))), 0.0, 1.0)

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        if features.schema_version != "http-v2":
            raise ValueError("unsupported feature schema")
        x = np.asarray([[float(features.values[name]) for name in self.feature_names]], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("non-finite model input")
        score = float(self.score_rows(x.tolist())[0])
        confidence = float(min(1.0, abs(score - 0.5) * 2.0))
        reason = "deviation from learned benign baseline" if score >= 0.5 else "within learned benign envelope"
        return DetectionSignal(
            self.name, round(score, 6), round(confidence, 6), (reason,), (),
            {
                "baseline_version": self.baseline_version,
                "decision_threshold": round(self.decision_threshold, 6),
                "model_type": "OneClassSVM",
                "score_semantics": "risk_not_calibrated_probability",
            },
        )
