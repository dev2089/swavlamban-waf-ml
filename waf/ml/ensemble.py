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
from waf.ml.outbound import OutboundAnomalyDetector, ResponseEnvelope
from waf.ml.semisupervised import SemiSupervisedDetector
from waf.ml.supervised import SupervisedDetector


@dataclass(slots=True)
class Phase4MLEnsemble:
    supervised: SupervisedDetector
    anomaly: UnsupervisedAnomalyDetector
    behaviour: BehaviouralDetector
    semi_supervised: SemiSupervisedDetector
    outbound: OutboundAnomalyDetector
    feature_names: tuple[str, ...]
    dataset_version: str
    baseline_version: str
    model_version: str = "phase10-ml-v3"

    @classmethod
    def train_default(cls) -> "Phase4MLEnsemble":
        supervised, anomaly, behaviour, semi_supervised, feature_names, dataset_version, baseline_version = trained_models()
        outbound = OutboundAnomalyDetector.train_default(samples=1600, seed=42)
        return cls(supervised, anomaly, behaviour, semi_supervised, outbound, feature_names, dataset_version, baseline_version)

    @classmethod
    def load(cls, path: str | Path) -> "Phase4MLEnsemble":
        payload = joblib.load(path)
        if payload.get("artifact_version") != "phase10-model-v3":
            raise ValueError("unsupported model artifact version")
        if payload.get("feature_schema") != "http-v2":
            raise ValueError("model artifact feature schema mismatch")
        required = {"supervised", "anomaly", "behaviour", "semi_supervised", "outbound"}
        if not required.issubset(payload):
            raise ValueError("model artifact missing one or more required detector components")
        if not payload.get("feature_names"):
            raise ValueError("model artifact missing feature manifest")
        return cls(payload["supervised"], payload["anomaly"], payload["behaviour"], payload["semi_supervised"], payload["outbound"], tuple(payload["feature_names"]), payload["dataset_version"], payload["baseline_version"], payload.get("model_version", "phase10-ml-v3"))

    @classmethod
    def from_stateless_components(
        cls,
        supervised: SupervisedDetector,
        anomaly: UnsupervisedAnomalyDetector,
        behaviour: BehaviouralDetector,
        feature_names: tuple[str, ...],
        dataset_version: str,
        baseline_version: str,
        model_version: str = "phase10-ml-v3",
        semi_supervised: SemiSupervisedDetector | None = None,
        outbound: OutboundAnomalyDetector | None = None,
    ) -> "Phase4MLEnsemble":
        """Construct a complete ensemble while retaining compatibility with Phase 7 callers."""
        if semi_supervised is None:
            semi_supervised = trained_models()[3]
        if outbound is None:
            outbound = OutboundAnomalyDetector.train_default(samples=1600, seed=42)
        isolated_behaviour = BehaviouralDetector(model=behaviour.model, window_seconds=behaviour.window_seconds, max_events_per_source=behaviour.max_events_per_source, name=behaviour.name)
        return cls(supervised, anomaly, isolated_behaviour, semi_supervised, outbound, feature_names, dataset_version, baseline_version, model_version)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"artifact_version": "phase10-model-v3", "feature_schema": "http-v2", "feature_names": self.feature_names, "dataset_version": self.dataset_version, "baseline_version": self.baseline_version, "model_version": self.model_version, "supervised": self.supervised, "anomaly": self.anomaly, "behaviour": self.behaviour, "semi_supervised": self.semi_supervised, "outbound": self.outbound}, path)

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> tuple[DetectionSignal, ...]:
        return (self.supervised.detect(request, features), self.anomaly.detect(request, features), self.behaviour.detect(request, features), self.semi_supervised.detect(request, features))

    def inspect_response(self, response: ResponseEnvelope) -> DetectionSignal:
        return self.outbound.detect(response)

    def metadata(self) -> dict[str, Any]:
        return {"model_version": self.model_version, "feature_schema": "http-v2", "feature_count": len(self.feature_names), "dataset_version": self.dataset_version, "baseline_version": self.baseline_version, "detectors": [self.supervised.name, self.anomaly.name, self.behaviour.name, self.semi_supervised.name, self.outbound.name], "scores_are_risk_not_probability": True, "semi_supervised_labeled_fraction": self.semi_supervised.labeled_fraction, "outbound_feature_schema": "http-response-v1", "outbound_direction": "response"}


def evaluate_unsupervised(seed: int = 42) -> dict[str, float | str | int]:
    bundle = build_training_dataset(samples=6000, seed=seed)
    X = np.asarray(bundle.X, dtype=float)
    y = np.asarray(bundle.y, dtype=int)
    benign_idx = np.flatnonzero(y == 0)
    attack_idx = np.flatnonzero(y == 1)
    split = max(1, int(len(benign_idx) * 0.75))
    train_idx = benign_idx[:split]
    eval_benign_idx = benign_idx[split:]
    detector = UnsupervisedAnomalyDetector.train(X[train_idx].tolist(), bundle.feature_names, f"benign-only-{seed}-{len(train_idx)}")
    benign_scores = detector.score_rows(X[eval_benign_idx].tolist())
    attack_scores = detector.score_rows(X[attack_idx].tolist())
    return {"baseline_version": detector.baseline_version, "benign_eval_samples": int(len(eval_benign_idx)), "attack_eval_samples": int(len(attack_idx)), "false_positive_rate": float(np.mean(benign_scores >= 0.5)), "attack_detection_rate": float(np.mean(attack_scores >= 0.5)), "mean_benign_score": float(np.mean(benign_scores)), "mean_attack_score": float(np.mean(attack_scores)), "evaluation_scope": "deterministic synthetic HTTP benchmark only"}


def evaluate_behaviour() -> dict[str, float | int | str]:
    from waf.core.models import RequestEnvelope
    from waf.features.http_v2 import ProductionHTTPFeatureExtractor
    extractor = ProductionHTTPFeatureExtractor()
    detector = BehaviouralDetector.train_default()
    normal_scores: list[float] = []
    for i in range(10):
        request = RequestEnvelope(f"normal-{i}", "GET", "https", "example.test", "/api/item/1", f"q={i}", source_ip="10.0.0.10", timestamp=1000.0 + i)
        normal_scores.append(detector.detect(request, extractor.extract(request)).score)
    burst_scores: list[float] = []
    for i in range(40):
        request = RequestEnvelope(f"burst-{i}", "GET", "https", "example.test", f"/api/{i % 12}", f"q={i}", source_ip="10.0.0.20", timestamp=2000.0 + i * 0.05)
        burst_scores.append(detector.detect(request, extractor.extract(request)).score)
    return {"normal_max_score": float(max(normal_scores)), "burst_final_score": float(burst_scores[-1]), "burst_escalated": bool(burst_scores[-1] > max(normal_scores) and burst_scores[-1] >= 0.5), "normal_requests": len(normal_scores), "burst_requests": len(burst_scores), "evaluation_scope": "deterministic synthetic behavioural workload"}


def evaluate_semi_supervised(seed: int = 42) -> dict[str, float | str | int]:
    bundle = build_training_dataset(samples=6000, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(bundle.X, dtype=float), np.asarray(bundle.y, dtype=int), test_size=0.25, random_state=seed, stratify=bundle.y)
    model = SemiSupervisedDetector.train(X_train.tolist(), y_train.tolist(), bundle.feature_names, bundle.dataset_version, labeled_fraction=0.30, seed=seed)
    pred = model.model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    labeled_count = max(2, int(len(y_train) * model.labeled_fraction))
    return {"dataset_version": bundle.dataset_version, "accuracy": float(accuracy_score(y_test, pred)), "precision": float(precision_score(y_test, pred, zero_division=0)), "recall": float(recall_score(y_test, pred, zero_division=0)), "f1": float(f1_score(y_test, pred, zero_division=0)), "fpr": float(fp / max(1, fp + tn)), "test_samples": int(len(y_test)), "training_samples": int(len(y_train)), "labeled_fraction": float(model.labeled_fraction), "labeled_samples": labeled_count, "unlabeled_samples": int(len(y_train) - labeled_count), "evaluation_scope": "deterministic synthetic HTTP benchmark with partial labels"}


def evaluate_outbound() -> dict[str, float | int | str]:
    detector = OutboundAnomalyDetector.train_default(samples=1600, seed=42)
    benign = [ResponseEnvelope(200, {"Content-Type": "application/json"}, b'{"ok":true,"items":[1,2,3]}'), ResponseEnvelope(200, {"Content-Type": "text/html"}, b"<html><body>OK</body></html>"), ResponseEnvelope(204, {"Content-Type": "text/plain"}, b"")] * 40
    anomalous = [ResponseEnvelope(500, {"Content-Type": "text/plain"}, b"Traceback (most recent call last): Exception secret password=admin"), ResponseEnvelope(200, {"Content-Type": "text/html"}, b"<html><script>steal()</script><body>debug secret</body></html>"), ResponseEnvelope(200, {"Content-Type": "text/plain"}, b"internal server error " + b"X" * 1_500_000)] * 20
    benign_scores = [detector.detect(x).score for x in benign]
    anomaly_scores = [detector.detect(x).score for x in anomalous]
    return {"detector": detector.name, "benign_samples": len(benign_scores), "anomalous_samples": len(anomaly_scores), "false_positive_rate": float(np.mean(np.asarray(benign_scores) >= 0.5)), "anomaly_detection_rate": float(np.mean(np.asarray(anomaly_scores) >= 0.5)), "evaluation_scope": "deterministic synthetic HTTP response workload"}


@lru_cache(maxsize=2)
def _cached_stateless_components(path: str) -> tuple[SupervisedDetector, UnsupervisedAnomalyDetector, BehaviouralDetector, SemiSupervisedDetector, OutboundAnomalyDetector, tuple[str, ...], str, str, str]:
    artifact_path = Path(path)
    if artifact_path.exists():
        payload = joblib.load(artifact_path)
        if payload.get("artifact_version") != "phase10-model-v3":
            raise ValueError("unsupported model artifact version")
        if payload.get("feature_schema") != "http-v2":
            raise ValueError("model artifact feature schema mismatch")
        required = {"supervised", "anomaly", "behaviour", "semi_supervised", "outbound"}
        if not required.issubset(payload):
            raise ValueError("model artifact missing one or more required detector components")
        if not payload.get("feature_names"):
            raise ValueError("model artifact missing feature manifest")
        return (payload["supervised"], payload["anomaly"], payload["behaviour"], payload["semi_supervised"], payload["outbound"], tuple(payload["feature_names"]), payload["dataset_version"], payload["baseline_version"], payload.get("model_version", "phase10-ml-v3"))
    trained = Phase4MLEnsemble.train_default()
    return (trained.supervised, trained.anomaly, trained.behaviour, trained.semi_supervised, trained.outbound, trained.feature_names, trained.dataset_version, trained.baseline_version, trained.model_version)


def load_runtime(path: str | Path) -> Phase4MLEnsemble:
    components = _cached_stateless_components(str(Path(path)))
    supervised, anomaly, behaviour, semi_supervised, outbound, feature_names, dataset_version, baseline_version, model_version = components
    return Phase4MLEnsemble.from_stateless_components(supervised, anomaly, behaviour, feature_names, dataset_version, baseline_version, model_version=model_version, semi_supervised=semi_supervised, outbound=outbound)


def evaluate_supervised(seed: int = 42) -> dict[str, float | str | int]:
    bundle = build_training_dataset(samples=6000, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(bundle.X, dtype=float), np.asarray(bundle.y, dtype=int), test_size=0.25, random_state=seed, stratify=bundle.y)
    model = SupervisedDetector.train(X_train.tolist(), y_train.tolist(), bundle.feature_names, bundle.dataset_version)
    pred = model.model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {"dataset_version": bundle.dataset_version, "accuracy": float(accuracy_score(y_test, pred)), "precision": float(precision_score(y_test, pred, zero_division=0)), "recall": float(recall_score(y_test, pred, zero_division=0)), "f1": float(f1_score(y_test, pred, zero_division=0)), "fpr": float(fp / max(1, fp + tn)), "test_samples": int(len(y_test)), "evaluation_scope": "deterministic synthetic HTTP benchmark only"}
