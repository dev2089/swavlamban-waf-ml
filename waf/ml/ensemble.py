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
from waf.ml.dataset import build_benign_baseline, build_training_dataset
from waf.ml.supervised import SupervisedDetector

ARTIFACT_VERSION = "phase4-model-v1"
MODEL_VERSION = "phase4-ml-v1"
FEATURE_SCHEMA = "http-v2"


@dataclass(slots=True)
class Phase4MLEnsemble:
    supervised: SupervisedDetector
    anomaly: UnsupervisedAnomalyDetector
    behaviour: BehaviouralDetector
    feature_names: tuple[str, ...]
    dataset_version: str
    baseline_version: str
    model_version: str = MODEL_VERSION

    @classmethod
    def train_default(cls) -> "Phase4MLEnsemble":
        bundle = build_training_dataset(samples=5000, seed=42)
        benign_x, baseline_names, baseline_version = build_benign_baseline(samples=2200, seed=123)
        if bundle.feature_names != baseline_names:
            raise ValueError("training and benign baseline feature manifests differ")
        supervised = SupervisedDetector.train(bundle.X, bundle.y, bundle.feature_names, bundle.dataset_version)
        anomaly = UnsupervisedAnomalyDetector.train(benign_x, baseline_names, baseline_version)
        behaviour = BehaviouralDetector.train_default(seed=42)
        return cls(supervised, anomaly, behaviour, bundle.feature_names, bundle.dataset_version, baseline_version)

    @classmethod
    def load(cls, path: str | Path) -> "Phase4MLEnsemble":
        payload = joblib.load(path)
        cls._validate_payload(payload)
        return cls(
            payload["supervised"], payload["anomaly"], payload["behaviour"],
            tuple(payload["feature_names"]), payload["dataset_version"],
            payload["baseline_version"], payload.get("model_version", MODEL_VERSION),
        )

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported model artifact version")
        if payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("model artifact feature schema mismatch")
        if tuple(payload.get("feature_names", ())) == () or len(payload["feature_names"]) != 40:
            raise ValueError("model artifact feature manifest must contain 40 features")
        for component in ("supervised", "anomaly", "behaviour"):
            if component not in payload:
                raise ValueError(f"model artifact missing {component} component")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "artifact_version": ARTIFACT_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "feature_names": self.feature_names,
            "dataset_version": self.dataset_version,
            "baseline_version": self.baseline_version,
            "model_version": self.model_version,
            "supervised": self.supervised,
            "anomaly": self.anomaly,
            "behaviour": self.behaviour,
        }, target)

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> tuple[DetectionSignal, ...]:
        return (
            self.supervised.detect(request, features),
            self.anomaly.detect(request, features),
            self.behaviour.detect(request, features),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "model_version": self.model_version,
            "feature_schema": FEATURE_SCHEMA,
            "feature_count": len(self.feature_names),
            "dataset_version": self.dataset_version,
            "baseline_version": self.baseline_version,
            "detectors": [self.supervised.name, self.anomaly.name, self.behaviour.name],
            "scores_are_risk_not_probability": True,
        }


def evaluate_supervised(seed: int = 42) -> dict[str, float | str | int]:
    bundle = build_training_dataset(samples=6000, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(bundle.X, dtype=float), np.asarray(bundle.y, dtype=int),
        test_size=0.25, random_state=seed, stratify=bundle.y,
    )
    model = SupervisedDetector.train(X_train.tolist(), y_train.tolist(), bundle.feature_names, bundle.dataset_version)
    pred = model.model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "dataset_version": bundle.dataset_version,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "test_samples": int(len(y_test)),
        "evaluation_scope": "deterministic synthetic HTTP benchmark only",
    }


def evaluate_unsupervised(seed: int = 42) -> dict[str, float | str | int]:
    bundle = build_training_dataset(samples=6000, seed=seed)
    X = np.asarray(bundle.X, dtype=float)
    y = np.asarray(bundle.y, dtype=int)
    benign_idx = np.flatnonzero(y == 0)
    attack_idx = np.flatnonzero(y == 1)
    split = max(1, int(len(benign_idx) * 0.75))
    detector = UnsupervisedAnomalyDetector.train(
        X[benign_idx[:split]].tolist(), bundle.feature_names, f"benign-only-v4-seed{seed}-n{split}"
    )
    benign_scores = detector.score_rows(X[benign_idx[split:]].tolist())
    attack_scores = detector.score_rows(X[attack_idx].tolist())
    return {
        "baseline_version": detector.baseline_version,
        "benign_eval_samples": int(len(benign_scores)),
        "attack_eval_samples": int(len(attack_scores)),
        "false_positive_rate": float(np.mean(benign_scores >= 0.5)),
        "attack_detection_rate": float(np.mean(attack_scores >= 0.5)),
        "mean_benign_score": float(np.mean(benign_scores)),
        "mean_attack_score": float(np.mean(attack_scores)),
        "evaluation_scope": "deterministic synthetic HTTP benchmark only",
    }


def evaluate_behaviour() -> dict[str, float | int | str | bool]:
    from waf.features.http_v2 import ProductionHTTPFeatureExtractor
    extractor = ProductionHTTPFeatureExtractor()
    detector = BehaviouralDetector.train_default()
    normal_scores: list[float] = []
    for i in range(10):
        req = RequestEnvelope(f"normal-{i}", "GET", "https", "example.test", "/api/item/1", f"q={i}", source_ip="10.0.0.10", timestamp=1000.0 + i)
        normal_scores.append(detector.detect(req, extractor.extract(req)).score)
    burst_scores: list[float] = []
    for i in range(40):
        req = RequestEnvelope(f"burst-{i}", "GET", "https", "example.test", f"/api/{i % 12}", f"q={i}", source_ip="10.0.0.20", timestamp=2000.0 + i * 0.05)
        burst_scores.append(detector.detect(req, extractor.extract(req)).score)
    return {
        "normal_max_score": float(max(normal_scores)),
        "burst_final_score": float(burst_scores[-1]),
        "burst_escalated": bool(burst_scores[-1] > max(normal_scores) and burst_scores[-1] >= 0.5),
        "normal_requests": len(normal_scores),
        "burst_requests": len(burst_scores),
        "evaluation_scope": "deterministic synthetic behavioural workload",
    }


@lru_cache(maxsize=2)
def _cached_stateless_components(path: str):
    artifact = Path(path)
    if artifact.exists():
        loaded = Phase4MLEnsemble.load(artifact)
    else:
        loaded = Phase4MLEnsemble.train_default()
    return (
        loaded.supervised, loaded.anomaly, loaded.behaviour,
        loaded.feature_names, loaded.dataset_version, loaded.baseline_version, loaded.model_version,
    )


def load_runtime(path: str | Path) -> Phase4MLEnsemble:
    components = _cached_stateless_components(str(Path(path).resolve()))
    supervised, anomaly, behaviour, names, dataset_version, baseline_version, version = components
    isolated_behaviour = BehaviouralDetector(
        model=behaviour.model, window_seconds=behaviour.window_seconds,
        max_events_per_source=behaviour.max_events_per_source, name=behaviour.name,
    )
    return Phase4MLEnsemble(supervised, anomaly, isolated_behaviour, names, dataset_version, baseline_version, version)
