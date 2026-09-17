from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope


@dataclass(slots=True)
class UnsupervisedAnomalyDetector:
    """One-class unsupervised detector trained only on benign HTTP baseline data."""

    feature_names: tuple[str, ...]
    model: OneClassSVM
    baseline_version: str
    decision_threshold: float
    name: str = "unsupervised-oneclasssvm-v1"

    @classmethod
    def train(
        cls,
        X_benign: list[list[float]],
        feature_names: tuple[str, ...],
        baseline_version: str,
    ) -> "UnsupervisedAnomalyDetector":
        X = np.asarray(X_benign, dtype=float)
        model = OneClassSVM(gamma="scale", nu=0.02)
        model.fit(X)
        threshold = float(np.percentile(model.decision_function(X), 2.0))
        return cls(feature_names, model, baseline_version, threshold)

    def _score_array(self, X: np.ndarray) -> tuple[float, float]:
        decision = float(self.model.decision_function(X)[0])
        score = 1.0 / (1.0 + np.exp(8.0 * (decision - self.decision_threshold)))
        score = float(max(0.0, min(1.0, score)))
        confidence = min(1.0, abs(score - 0.5) * 2.0)
        return score, float(confidence)

    def score_rows(self, X: list[list[float]]) -> np.ndarray:
        rows = np.asarray(X, dtype=float)
        decisions = self.model.decision_function(rows)
        return 1.0 / (1.0 + np.exp(8.0 * (decisions - self.decision_threshold)))

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        x = np.asarray([[features.values[name] for name in self.feature_names]], dtype=float)
        score, confidence = self._score_array(x)
        anomalous = score >= 0.5
        reason = "deviation from learned benign baseline" if anomalous else "within learned benign envelope"
        return DetectionSignal(
            detector=self.name,
            score=round(score, 6),
            confidence=round(confidence, 6),
            reasons=(reason,),
            metadata={
                "baseline_version": self.baseline_version,
                "decision_threshold": round(self.decision_threshold, 6),
            },
        )
