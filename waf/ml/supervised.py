from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope


@dataclass(slots=True)
class SupervisedDetector:
    feature_names: tuple[str, ...]
    model: HistGradientBoostingClassifier
    dataset_version: str
    name: str = "supervised-v1"

    @classmethod
    def train(
        cls,
        X: list[list[float]],
        y: list[int],
        feature_names: tuple[str, ...],
        dataset_version: str,
    ) -> "SupervisedDetector":
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=180,
            min_samples_leaf=8,
            random_state=42,
        )
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return cls(feature_names, model, dataset_version)

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        x = np.asarray([[features.values[name] for name in self.feature_names]], dtype=float)
        score = float(self.model.predict_proba(x)[0, 1])
        confidence = min(1.0, abs(score - 0.5) * 2.0)
        reasons: list[str] = []
        for key in (
            "has_sql_keyword",
            "has_xss_token",
            "has_traversal",
            "has_command_token",
            "malformed_percent_flag",
            "double_encoded_flag",
        ):
            if features.values.get(key, 0.0) >= 0.5:
                reasons.append(f"supervised signal: {key}")
        return DetectionSignal(
            detector=self.name,
            score=round(max(0.0, min(1.0, score)), 6),
            confidence=round(confidence, 6),
            reasons=tuple(reasons),
            metadata={"dataset_version": self.dataset_version},
        )
