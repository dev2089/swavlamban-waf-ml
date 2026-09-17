from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope
from waf.ml.anomaly import UnsupervisedAnomalyDetector
from waf.ml.behaviour import BehaviouralDetector
from waf.ml.dataset import build_training_dataset
from waf.ml.factory import trained_models
from waf.ml.supervised import SupervisedDetector


@dataclass(slots=True)
class Phase4MLEnsemble:
    supervised: SupervisedDetector
    anomaly: UnsupervisedAnomalyDetector
    behaviour: BehaviouralDetector
    feature_names: tuple[str, ...]
    dataset_version: str
    baseline_version: str
    model_version: str = "phase4-ml-v1"

    @classmethod
    def train_default(cls) -> "Phase4MLEnsemble":
        supervised, anomaly, behaviour, feature_names, dataset_version, baseline_version = trained_models()
        return cls(supervised, anomaly, behaviour, feature_names, dataset_version, baseline_version)

    @classmethod
    def load(cls, path: str | Path) -> "Phase4MLEnsemble":
        payload = joblib.load(path)
        if payload.get("artifact_version") != "phase4-model-v1":
            raise ValueError("unsupported model artifact version")
        if payload.get("feature_schema") != "http-v2":
            raise ValueError("model artifact feature schema mismatch")
        if "supervised" not in payload or "anomaly" not in payload or "behaviour" not in payload:
            raise ValueError("model artifact missing one or more required detector components")
        if not payload.get("feature_names"):
            raise ValueError("model artifact missing feature manifest")
        return cls(
            payload["supervised"],
            payload["anomaly"],
            payload["behaviour"],
            tuple(payload["feature_names"]),
            payload["dataset_version"],
            payload["baseline_version"],
            payload.get("model_version", "phase4-ml-v1"),
        )

    @classmethod
    def from_stateless_components(
        cls,
        supervised: SupervisedDetector,
        anomaly: UnsupervisedAnomalyDetector,
        behaviour: BehaviouralDetector,
        feature_names: tuple[str, ...],
        dataset_version: str,
        baseline_version: str,
        model_version: str = "phase4-ml-v1",
    ) -> "Phase4MLEnsemble":
        isolated_behaviour = BehaviouralDetector(
            model=behaviour.model,
            window_seconds=behaviour.window_seconds,
            max_events_per_source=behaviour.max_events_per_source,
            name=behaviour.name,
        )
        return cls(
            supervised,
            anomaly,
            isolated_behaviour,
            feature_names,
            dataset_version,
            baseline_version,
            model_version,
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": "phase4-model-v1",
                "feature_schema": "http-v2",
                "feature_names": self.feature_names,
                "dataset_version": self.dataset_version,
                "baseline_version": self.baseline_version,
                "model_version": self.model_version,
                "supervised": self.supervised,
                "anomaly": self.anomaly,
                "behaviour": self.behaviour,
            },
            path,
        )

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> tuple[DetectionSignal, ...]:
        return (
            self.supervised.detect(request, features),
            self.anomaly.detect(request, features),
            self.behaviour.detect(request, features),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "feature_schema": "http-v2",
            "feature_count": len(self.feature_names),
            "dataset_version": self.dataset_version,
            "baseline_version": self.baseline_version,
            "detectors": [self.supervised.name, self.anomaly.name, self.behaviour.name],
            "scores_are_risk_not_probability": True,
        }


def evaluate_unsupervised(seed: int = 42) -> dict[str, float | str | int]:
    bundle = build_training_dataset(samples=6000, seed=seed)
    X = np.asarray(bundle.X, dtype=float)
    y = np.asarray(bundle.y, dtype=int)
    benign_idx = np.flatnonzero(y == 0)
    attack_idx = np.flatnonzero(y == 1)
    split = max(1, int(len(benign_idx) * 0.75))
    train_idx = benign_idx[:split]
    eval_benign_idx = benign_idx[split:]
    detector = UnsupervisedAnomalyDetector.train(
        X[train_idx].tolist(), bundle.feature_names, f"benign-only-{seed}-{len(train_idx)}"
    )
    benign_scores = detector.score_rows(X[eval_benign_idx].tolist())
    attack_scores = detector.score_rows(X[attack_idx].tolist())
    return {
        "baseline_version": detector.baseline_version,
        "benign_eval_samples": int(len(eval_benign_idx)),
        "attack_eval_samples": int(len(attack_idx)),
        "false_positive_rate": float(np.mean(benign_scores >= 0.5)),
        "attack_detection_rate": float(np.mean(attack_scores >= 0.5)),
        "mean_benign_score": float(np.mean(benign_scores)),
        "mean_attack_score": float(np.mean(attack_scores)),
        "evaluation_scope": "deterministic synthetic HTTP benchmark only",
    }


def evaluate_behaviour() -> dict[str, float | int | str]:
    from waf.core.models import RequestEnvelope
    from waf.features.http_v2 import ProductionHTTPFeatureExtractor

    extractor = ProductionHTTPFeatureExtractor()
    detector = BehaviouralDetector.train_default()

    normal_scores: list[float] = []
    for i in range(10):
        request = RequestEnvelope(
            f"normal-{i}", "GET", "https", "example.test", "/api/item/1", f"q={i}",
            source_ip="10.0.0.10", timestamp=1000.0 + i,
        )
        normal_scores.append(detector.detect(request, extractor.extract(request)).score)

    burst_scores: list[float] = []
    for i in range(40):
        request = RequestEnvelope(
            f"burst-{i}", "GET", "https", "example.test", f"/api/{i % 12}", f"q={i}",
            source_ip="10.0.0.20", timestamp=2000.0 + i * 0.05,
        )
        burst_scores.append(detector.detect(request, extractor.extract(request)).score)

    return {
        "normal_max_score": float(max(normal_scores)),
        "burst_final_score": float(burst_scores[-1]),
        "burst_escalated": bool(burst_scores[-1] > max(normal_scores) and burst_scores[-1] >= 0.5),
        "normal_requests": len(normal_scores),
        "burst_requests": len(burst_scores),
        "evaluation_scope": "deterministic synthetic behavioural workload",
    }


@lru_cache(maxsize=2)
def _cached_stateless_components(
    path: str,
) -> tuple[SupervisedDetector, UnsupervisedAnomalyDetector, BehaviouralDetector, tuple[str, ...], str, str, str]:
    artifact_path = Path(path)
    if artifact_path.exists():
        payload = joblib.load(artifact_path)
        if payload.get("artifact_version") != "phase4-model-v1":
            raise ValueError("unsupported model artifact version")
        if payload.get("feature_schema") != "http-v2":
            raise ValueError("model artifact feature schema mismatch")
        if "supervised" not in payload or "anomaly" not in payload or "behaviour" not in payload:
            raise ValueError("model artifact missing one or more required detector components")
        if not payload.get("feature_names"):
            raise ValueError("model artifact missing feature manifest")
        return (
            payload["supervised"],
            payload["anomaly"],
            payload["behaviour"],
            tuple(payload["feature_names"]),
            payload["dataset_version"],
            payload["baseline_version"],
            payload.get("model_version", "phase4-ml-v1"),
        )
    trained = Phase4MLEnsemble.train_default()
    return (
        trained.supervised,
        trained.anomaly,
        trained.behaviour,
        trained.feature_names,
        trained.dataset_version,
        trained.baseline_version,
        trained.model_version,
    )


def load_runtime(path: str | Path) -> Phase4MLEnsemble:
    components = _cached_stateless_components(str(Path(path)))
    return Phase4MLEnsemble.from_stateless_components(*components)


def evaluate_supervised(seed: int = 42) -> dict[str, float | str | int]:
    bundle = build_training_dataset(samples=6000, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(bundle.X, dtype=float),
        np.asarray(bundle.y, dtype=int),
        test_size=0.25,
        random_state=seed,
        stratify=bundle.y,
    )
    model = SupervisedDetector.train(X_train.tolist(), y_train.tolist(), bundle.feature_names, bundle.dataset_version)
    pred = model.model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, fp + tn))
    return {
        "dataset_version": bundle.dataset_version,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "fpr": fpr,
        "test_samples": int(len(y_test)),
        "evaluation_scope": "deterministic synthetic HTTP benchmark only",
    }
