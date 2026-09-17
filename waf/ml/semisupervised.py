from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope


@dataclass(slots=True)
class SemiSupervisedDetector:
    feature_names: tuple[str, ...]
    model: SelfTrainingClassifier
    dataset_version: str
    labeled_fraction: float
    name: str = "semi-supervised-v1"

    @classmethod
    def train(
        cls,
        X: list[list[float]],
        y: list[int],
        feature_names: tuple[str, ...],
        dataset_version: str,
        labeled_fraction: float = 0.30,
        seed: int = 42,
    ) -> "SemiSupervisedDetector":
        if not 0.05 <= labeled_fraction <= 0.95:
            raise ValueError("labeled_fraction must be in [0.05, 0.95]")
        labels = np.asarray(y, dtype=int).copy()
        rng = np.random.default_rng(seed)
        labeled_count = max(2, int(len(labels) * labeled_fraction))
        labeled_indices = np.sort(rng.choice(len(labels), size=labeled_count, replace=False))
        mask = np.ones(len(labels), dtype=bool)
        mask[labeled_indices] = False
        labels[mask] = -1
        if len(np.unique(labels[labels >= 0])) < 2:
            # Guarantee both classes are represented among the labeled seed set.
            class0 = int(np.flatnonzero(np.asarray(y) == 0)[0])
            class1 = int(np.flatnonzero(np.asarray(y) == 1)[0])
            labels[:] = -1
            labels[class0] = 0
            labels[class1] = 1
            extra = max(0, labeled_count - 2)
            if extra:
                remaining = [i for i in range(len(labels)) if i not in {class0, class1}]
                for idx in remaining[:extra]:
                    labels[idx] = int(y[idx])
        base = LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed)
        model = SelfTrainingClassifier(
            estimator=base,
            threshold=0.85,
            max_iter=12,
            verbose=False,
        )
        model.fit(np.asarray(X, dtype=float), labels)
        return cls(feature_names, model, dataset_version, labeled_count / len(labels))

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        x = np.asarray([[features.values[name] for name in self.feature_names]], dtype=float)
        probability = float(self.model.predict_proba(x)[0, 1])
        score = min(1.0, max(0.0, probability))
        confidence = min(1.0, abs(score - 0.5) * 2.0)
        reasons: list[str] = []
        for key in (
            "has_sql_keyword",
            "has_xss_token",
            "has_traversal",
            "has_command_token",
            "double_encoded_flag",
        ):
            if features.values.get(key, 0.0) >= 0.5:
                reasons.append(f"semi-supervised signal: {key}")
        return DetectionSignal(
            detector=self.name,
            score=round(score, 6),
            confidence=round(confidence, 6),
            reasons=tuple(reasons),
            metadata={
                "dataset_version": self.dataset_version,
                "labeled_fraction": round(self.labeled_fraction, 6),
                "unlabeled_training_used": True,
            },
        )
