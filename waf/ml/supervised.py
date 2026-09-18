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
    def train(cls, X, y, feature_names, dataset_version):
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
        if features.schema_version != "http-v2":
            raise ValueError("unsupported feature schema")
        try:
            x = np.asarray([[float(features.values[name]) for name in self.feature_names]], dtype=float)
        except KeyError as exc:
            raise ValueError(f"missing feature: {exc.args[0]}") from exc
        if not np.isfinite(x).all():
            raise ValueError("non-finite model input")
        score = float(np.clip(self.model.predict_proba(x)[0, 1], 0.0, 1.0))
        confidence = float(min(1.0, abs(score - 0.5) * 2.0))
        reasons = tuple(
            f"supervised signal: {key}"
            for key in ("has_sql_keyword","has_xss_token","has_traversal","has_command_token","malformed_percent_flag","double_encoded_flag")
            if features.values.get(key, 0.0) >= 0.5
        )
        return DetectionSignal(
            self.name, round(score, 6), round(confidence, 6), reasons, (),
            {
                "dataset_version": self.dataset_version,
                "model_type": "HistGradientBoostingClassifier",
                "score_semantics": "risk_not_calibrated_probability",
            },
        )
