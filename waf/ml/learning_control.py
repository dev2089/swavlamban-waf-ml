from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from waf.core.models import DecisionResult, FeatureVector
from waf.features.http_v2 import SCHEMA_VERSION
from waf.ml.dataset import DatasetBundle, build_benign_baseline, build_training_dataset
from waf.ml.ensemble import Phase4MLEnsemble
from waf.ml.supervised import SupervisedDetector

PHASE7_VERSION = "phase7-learning-control-v1"
BASELINE_SCHEMA = "baseline-v1"
FEEDBACK_SCHEMA = "feedback-v1"
DRIFT_SCHEMA = "drift-v1"
MODEL_RUN_SCHEMA = "model-run-v1"
MODEL_REGISTRY_SCHEMA = "model-registry-v1"
DRIFT_MEAN_PSI_THRESHOLD = 0.10
DRIFT_MAX_PSI_THRESHOLD = 0.25
MIN_DRIFT_SAMPLES = 32
MIN_REVIEWED_FEEDBACK = 40
PROMOTION_MAX_F1_DROP = 0.005
PROMOTION_MAX_FPR_RISE = 0.010


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _rows(rows: Sequence[Sequence[float]], feature_names: Sequence[str]) -> list[list[float]]:
    if not rows:
        raise ValueError("at least one feature row is required")
    if any(len(row) != len(feature_names) for row in rows):
        raise ValueError("feature row width does not match feature manifest")
    return [[round(min(1.0, max(0.0, float(v))), 6) for v in row] for row in rows]


def _feature_row(features: FeatureVector, names: Sequence[str]) -> list[float]:
    if features.schema_version != SCHEMA_VERSION:
        raise ValueError("feature schema mismatch")
    return [round(float(features.values.get(name, 0.0)), 6) for name in names]


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    schema_version: str
    baseline_version: str
    feature_schema: str
    feature_names: tuple[str, ...]
    sample_count: int
    source: str
    rows: tuple[tuple[float, ...], ...]
    row_sha256: str
    created_at: str
    privacy: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != BASELINE_SCHEMA or self.feature_schema != SCHEMA_VERSION:
            raise ValueError("unsupported baseline contract")
        if self.sample_count != len(self.rows) or self.sample_count < MIN_DRIFT_SAMPLES:
            raise ValueError("invalid baseline sample count")
        if self.privacy.get("raw_request_material_retained") is not False:
            raise ValueError("raw request material is forbidden in baseline")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["feature_names"] = list(self.feature_names)
        out["rows"] = [list(row) for row in self.rows]
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineRecord":
        return cls(
            str(payload["schema_version"]), str(payload["baseline_version"]), str(payload["feature_schema"]),
            tuple(payload["feature_names"]), int(payload["sample_count"]), str(payload["source"]),
            tuple(tuple(float(v) for v in row) for row in payload["rows"]), str(payload["row_sha256"]),
            str(payload["created_at"]), dict(payload["privacy"]),
        )


class BaselineManager:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)

    def create(self, rows: Sequence[Sequence[float]], source: str, parent_version: str | None = None) -> BaselineRecord:
        normalized = _rows(rows, self.feature_names)
        if not source.strip():
            raise ValueError("baseline source is required")
        row_sha = _hash(normalized)
        version = f"baseline-{_hash({'schema': BASELINE_SCHEMA, 'feature_schema': SCHEMA_VERSION, 'names': self.feature_names, 'source': source, 'parent': parent_version, 'row_sha': row_sha})[:12]}"
        return BaselineRecord(BASELINE_SCHEMA, version, SCHEMA_VERSION, self.feature_names, len(normalized), source, tuple(tuple(row) for row in normalized), row_sha, _now(), {"raw_request_material_retained": False, "stored_data": "normalized http-v2 numeric features only", "raw_fields": []})

    def create_default(self, samples: int = 2200, seed: int = 123) -> BaselineRecord:
        rows, names, version = build_benign_baseline(samples=samples, seed=seed)
        if tuple(names) != self.feature_names:
            raise ValueError("default baseline feature manifest mismatch")
        record = self.create(rows, f"synthetic-benign-builder:{version}")
        return BaselineRecord(record.schema_version, f"{version}-p7", record.feature_schema, record.feature_names, record.sample_count, record.source, record.rows, record.row_sha256, record.created_at, record.privacy)

    def from_feature_vectors(self, feature_vectors: Iterable[FeatureVector], source: str, parent_version: str | None = None) -> BaselineRecord:
        return self.create([_feature_row(v, self.feature_names) for v in feature_vectors], source, parent_version)

    @staticmethod
    def save(record: BaselineRecord, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> BaselineRecord:
        return BaselineRecord.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    schema_version: str
    record_id: str
    request_id: str
    feature_schema: str
    feature_names: tuple[str, ...]
    feature_snapshot: tuple[float, ...]
    observed_decision: str
    reviewed_label: int | None
    review_state: str
    reviewer: str | None
    review_note: str
    evidence_rule_ids: tuple[str, ...]
    model_version: str
    baseline_version: str
    created_at: str
    reviewed_at: str | None
    privacy: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != FEEDBACK_SCHEMA or self.feature_schema != SCHEMA_VERSION:
            raise ValueError("unsupported feedback contract")
        if len(self.feature_snapshot) != len(self.feature_names) or self.review_state not in {"pending", "reviewed", "rejected"}:
            raise ValueError("invalid feedback record")
        if self.reviewed_label not in (None, 0, 1):
            raise ValueError("reviewed label must be 0, 1 or null")
        if self.review_state == "reviewed" and self.reviewed_label is None:
            raise ValueError("reviewed feedback requires a label")
        if self.privacy.get("raw_request_material_retained") is not False:
            raise ValueError("raw request material is forbidden in feedback")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["feature_names"] = list(self.feature_names)
        out["feature_snapshot"] = list(self.feature_snapshot)
        out["evidence_rule_ids"] = list(self.evidence_rule_ids)
        return out


class FeedbackStore:
    def __init__(self) -> None:
        self._records: dict[str, FeedbackRecord] = {}
        self._lock = RLock()

    def add_from_decision(self, result: DecisionResult, features: FeatureVector, model_version: str, baseline_version: str) -> FeedbackRecord:
        names = tuple(sorted(features.values))
        snapshot = _feature_row(features, names)
        record_id = f"FDB-{_hash({'request_id': result.request_id, 'features': snapshot})[:16].upper()}"
        record = FeedbackRecord(FEEDBACK_SCHEMA, record_id, result.request_id, SCHEMA_VERSION, names, tuple(snapshot), result.decision.value, None, "pending", None, "", tuple(sorted(result.rule_ids)), model_version, baseline_version, _now(), None, {"raw_request_material_retained": False, "stored_data": "feature snapshot + decision/evidence metadata only"})
        with self._lock:
            self._records[record_id] = record
        return record

    def review(self, record_id: str, label: int, reviewer: str, note: str = "") -> FeedbackRecord:
        reviewer = reviewer.strip()
        if label not in (0, 1) or not reviewer:
            raise ValueError("review label and reviewer are required")
        with self._lock:
            cur = self._records[record_id]
            updated = FeedbackRecord(cur.schema_version, cur.record_id, cur.request_id, cur.feature_schema, cur.feature_names, cur.feature_snapshot, cur.observed_decision, label, "reviewed", reviewer, note[:240], cur.evidence_rule_ids, cur.model_version, cur.baseline_version, cur.created_at, _now(), cur.privacy)
            self._records[record_id] = updated
            return updated

    def reject(self, record_id: str, reviewer: str, note: str = "") -> FeedbackRecord:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        with self._lock:
            cur = self._records[record_id]
            updated = FeedbackRecord(cur.schema_version, cur.record_id, cur.request_id, cur.feature_schema, cur.feature_names, cur.feature_snapshot, cur.observed_decision, None, "rejected", reviewer, note[:240], cur.evidence_rule_ids, cur.model_version, cur.baseline_version, cur.created_at, _now(), cur.privacy)
            self._records[record_id] = updated
            return updated

    def reviewed(self) -> tuple[FeedbackRecord, ...]:
        with self._lock:
            return tuple(row for row in self._records.values() if row.review_state == "reviewed")

    def snapshot(self) -> tuple[FeedbackRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def export_json(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.snapshot()]


@dataclass(frozen=True, slots=True)
class DriftReport:
    schema_version: str
    report_id: str
    baseline_version: str
    feature_schema: str
    sample_count: int
    mean_psi: float
    max_psi: float
    material_drift: bool
    alert_level: str
    top_features: tuple[Mapping[str, Any], ...]
    thresholds: Mapping[str, float]
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != DRIFT_SCHEMA or self.alert_level not in {"none", "material"} or self.sample_count < MIN_DRIFT_SAMPLES:
            raise ValueError("invalid drift report")


class DriftDetector:
    def __init__(self, mean_threshold: float = DRIFT_MEAN_PSI_THRESHOLD, max_threshold: float = DRIFT_MAX_PSI_THRESHOLD) -> None:
        self.mean_threshold = float(mean_threshold)
        self.max_threshold = float(max_threshold)

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
        bins = np.linspace(0.0, 1.0, 6)
        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)
        eps = 1e-6
        exp = (exp_counts.astype(float) + eps) / (len(expected) + eps * 5.0)
        act = (act_counts.astype(float) + eps) / (len(actual) + eps * 5.0)
        return float(np.sum((act - exp) * np.log(act / exp)))

    def compare(self, baseline: BaselineRecord, current_rows: Sequence[Sequence[float]]) -> DriftReport:
        rows = np.asarray(_rows(current_rows, baseline.feature_names), dtype=float)
        if len(rows) < MIN_DRIFT_SAMPLES:
            raise ValueError(f"drift requires at least {MIN_DRIFT_SAMPLES} samples")
        base = np.asarray(baseline.rows, dtype=float)
        metrics = [{"feature": name, "psi": round(self._psi(base[:, i], rows[:, i]), 6)} for i, name in enumerate(baseline.feature_names)]
        ordered = sorted(metrics, key=lambda x: (-x["psi"], x["feature"]))
        mean_psi = float(np.mean([x["psi"] for x in metrics]))
        max_psi = float(max(x["psi"] for x in metrics))
        material = mean_psi >= self.mean_threshold or max_psi >= self.max_threshold
        report_id = f"DRIFT-{_hash({'baseline': baseline.baseline_version, 'rows': rows.tolist()})[:16].upper()}"
        return DriftReport(DRIFT_SCHEMA, report_id, baseline.baseline_version, baseline.feature_schema, int(len(rows)), round(mean_psi, 6), round(max_psi, 6), material, "material" if material else "none", tuple(ordered[:8]), {"mean_psi": self.mean_threshold, "max_psi": self.max_threshold}, _now())

    @staticmethod
    def alert(report: DriftReport) -> bool:
        return report.material_drift


@dataclass(frozen=True, slots=True)
class ModelRun:
    schema_version: str
    run_id: str
    role: str
    model_version: str
    dataset_version: str
    baseline_version: str
    artifact_path: str
    artifact_sha256: str
    artifact_bytes: int
    evaluation: Mapping[str, Any]
    source_feedback_ids: tuple[str, ...]
    created_at: str

@dataclass(frozen=True, slots=True)
class PromotionDecision:
    candidate_run_id: str
    champion_run_id: str
    eligible: bool
    reasons: tuple[str, ...]
    created_at: str


def _evaluate_supervised(detector: SupervisedDetector, bundle: DatasetBundle, seed: int) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    X, y = np.asarray(bundle.X, dtype=float), np.asarray(bundle.y, dtype=int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    pred = detector.model.predict(X_test)
    tn, fp, _, _ = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {"dataset_version": bundle.dataset_version, "accuracy": round(float(accuracy_score(y_test, pred)), 6), "precision": round(float(precision_score(y_test, pred, zero_division=0)), 6), "recall": round(float(recall_score(y_test, pred, zero_division=0)), 6), "f1": round(float(f1_score(y_test, pred, zero_division=0)), 6), "fpr": round(float(fp / max(1, fp + tn)), 6), "test_samples": int(len(y_test)), "evaluation_scope": "deterministic synthetic HTTP benchmark only"}


class ModelRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"schema_version": MODEL_REGISTRY_SCHEMA, "champion": None, "history": [], "promotions": [], "rollbacks": []}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @property
    def champion(self) -> dict[str, Any] | None:
        return self._state.get("champion")

    def initialize_champion(self, artifact_path: str | Path, model_version: str, dataset_version: str, baseline_version: str) -> dict[str, Any]:
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        evaluation = _evaluate_supervised(Phase4MLEnsemble.load(artifact).supervised, build_training_dataset(samples=6000, seed=777), 777)
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        record = {"run_id": f"RUN-CHAMPION-{_hash({'model_version': model_version, 'artifact': artifact_sha})[:12].upper()}", "role": "champion", "model_version": model_version, "dataset_version": dataset_version, "baseline_version": baseline_version, "artifact_path": str(artifact), "artifact_sha256": artifact_sha, "artifact_bytes": artifact.stat().st_size, "evaluation": evaluation, "created_at": _now()}
        with self._lock:
            if self._state.get("champion") is None:
                self._state["champion"] = record
                self._state["history"].append(record)
                self._save()
        return dict(self._state["champion"])

    def register_candidate(self, run: ModelRun) -> None:
        with self._lock:
            self._state["history"].append(asdict(run))
            self._save()

    def evaluate(self, candidate: ModelRun, champion: Mapping[str, Any]) -> PromotionDecision:
        c, p = candidate.evaluation, champion.get("evaluation", {})
        failures = []
        if float(c.get("f1", 0.0)) < float(p.get("f1", 0.0)) - PROMOTION_MAX_F1_DROP:
            failures.append("challenger f1 regressed beyond tolerance")
        if float(c.get("fpr", 1.0)) > float(p.get("fpr", 1.0)) + PROMOTION_MAX_FPR_RISE:
            failures.append("challenger false-positive rate rose beyond tolerance")
        reasons = tuple(failures or ["challenger meets champion promotion thresholds"])
        return PromotionDecision(candidate.run_id, str(champion["run_id"]), not failures, reasons, _now())

    def promote(self, candidate: ModelRun, approved_by: str) -> dict[str, Any]:
        approver = approved_by.strip()
        if not approver:
            raise ValueError("human approver is required")
        decision = self.evaluate(candidate, self._state["champion"])
        if not decision.eligible:
            raise ValueError("candidate is not promotion-eligible")
        source = Path(candidate.artifact_path)
        promoted_path = source.with_name(f"champion-{candidate.model_version}.joblib")
        shutil.copy2(source, promoted_path)
        with self._lock:
            prior = self._state["champion"]
            new = {"run_id": candidate.run_id, "role": "champion", "model_version": candidate.model_version, "dataset_version": candidate.dataset_version, "baseline_version": candidate.baseline_version, "artifact_path": str(promoted_path), "artifact_sha256": candidate.artifact_sha256, "artifact_bytes": promoted_path.stat().st_size, "evaluation": dict(candidate.evaluation), "approved_by": approver, "approved_at": _now()}
            self._state["promotions"].append({"candidate_run_id": candidate.run_id, "prior_run_id": prior["run_id"], "approved_by": approver, "approved_at": new["approved_at"]})
            self._state["champion"] = new
            self._save()
        return dict(new)

    def rollback(self, approved_by: str) -> dict[str, Any]:
        approver = approved_by.strip()
        if not approver:
            raise ValueError("human approver is required")
        with self._lock:
            current = self._state["champion"]
            prior = next((row for row in reversed(self._state["history"]) if row.get("role") == "champion" and row.get("run_id") != current["run_id"]), None)
            if prior is None:
                raise ValueError("no prior champion available for rollback")
            if not Path(prior["artifact_path"]).exists():
                raise FileNotFoundError(prior["artifact_path"])
            self._state["rollbacks"].append({"from_run_id": current["run_id"], "to_run_id": prior["run_id"], "approved_by": approver, "rolled_back_at": _now()})
            self._state["champion"] = dict(prior)
            self._save()
            return dict(prior)


def train_controlled_challenger(champion_path: str | Path, baseline: BaselineRecord, feedback: FeedbackStore, output_path: str | Path, registry_path: str | Path, drift_report: DriftReport, seed: int = 42) -> ModelRun:
    reviewed = feedback.reviewed()
    if not drift_report.material_drift and len(reviewed) < MIN_REVIEWED_FEEDBACK:
        raise ValueError("controlled retraining requires material drift or enough reviewed feedback")
    labels = [r.reviewed_label for r in reviewed]
    if not reviewed or 0 not in labels or 1 not in labels:
        raise ValueError("reviewed feedback must contain both benign and malicious labels")
    champion = Phase4MLEnsemble.load(champion_path)
    if tuple(champion.feature_names) != baseline.feature_names:
        raise ValueError("champion and baseline feature manifests do not match")
    base = build_training_dataset(samples=6000, seed=seed)
    X = list(base.X) + [list(r.feature_snapshot) for r in reviewed]
    y = list(base.y) + [int(r.reviewed_label) for r in reviewed]
    dataset_version = f"controlled-{_hash({'base': base.dataset_version, 'feedback': [r.record_id for r in reviewed], 'baseline': baseline.baseline_version})[:16]}"
    supervised = SupervisedDetector.train(X, y, champion.feature_names, dataset_version)
    challenger = Phase4MLEnsemble.from_stateless_components(supervised, champion.anomaly, champion.behaviour, champion.feature_names, dataset_version, baseline.baseline_version, model_version=f"phase7-challenger-{dataset_version[-12:]}")
    output = Path(output_path)
    challenger.save(output)
    evaluation = _evaluate_supervised(challenger.supervised, build_training_dataset(samples=6000, seed=777), 777)
    artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    run_id = f"RUN-CHALLENGER-{artifact_sha[:16].upper()}"
    manifest = {"schema_version": MODEL_RUN_SCHEMA, "run_id": run_id, "role": "challenger", "model_version": challenger.model_version, "dataset_version": dataset_version, "baseline_version": baseline.baseline_version, "artifact_sha256": artifact_sha, "artifact_bytes": output.stat().st_size, "source_feedback_ids": [r.record_id for r in reviewed], "drift_report_id": drift_report.report_id, "evaluation": evaluation, "created_at": _now(), "promotion_policy": {"requires_explicit_human_approval": True, "max_f1_drop": PROMOTION_MAX_F1_DROP, "max_fpr_rise": PROMOTION_MAX_FPR_RISE, "runtime_default_replaced_automatically": False}}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    registry = ModelRegistry(registry_path)
    registry.initialize_champion(champion_path, champion.model_version, champion.dataset_version, champion.baseline_version)
    candidate = ModelRun(MODEL_RUN_SCHEMA, run_id, "candidate", challenger.model_version, dataset_version, baseline.baseline_version, str(output), artifact_sha, output.stat().st_size, evaluation, tuple(r.record_id for r in reviewed), manifest["created_at"])
    registry.register_candidate(candidate)
    return candidate


def build_default_baseline(path: str | Path, samples: int = 2200, seed: int = 123) -> BaselineRecord:
    _, names, _ = build_benign_baseline(samples=max(32, samples), seed=seed)
    manager = BaselineManager(names)
    record = manager.create_default(samples=samples, seed=seed)
    manager.save(record, path)
    return record
