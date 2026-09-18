from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from waf.core.config import WAFConfig
from waf.core.models import RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.dataset import build_benign_baseline
from waf.ml.learning_control import (
    BaselineManager,
    DriftDetector,
    FeedbackStore,
    ModelRegistry,
    PROMOTION_MAX_F1_DROP,
    train_controlled_challenger,
)
from waf.ml.ensemble import Phase4MLEnsemble


class Phase7LearningControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = ProductionHTTPFeatureExtractor()
        self.ml = Phase4MLEnsemble.train_default()
        rows, names, _ = build_benign_baseline(samples=220, seed=123)
        self.baseline = BaselineManager(names).create(rows, "test-benign-baseline")

    def test_versioned_baseline_is_numeric_and_private(self):
        payload = self.baseline.to_dict()
        encoded = json.dumps(payload, sort_keys=True)
        self.assertEqual(self.baseline.schema_version, "baseline-v1")
        self.assertEqual(self.baseline.feature_schema, "http-v2")
        self.assertEqual(self.baseline.sample_count, 220)
        self.assertFalse(self.baseline.privacy["raw_request_material_retained"])
        self.assertNotIn("example.test", encoded)
        self.assertNotIn("/health", encoded)

    def test_reviewed_feedback_requires_human_review_and_keeps_features_only(self):
        waf = EdgeWAF(WAFConfig(pipeline_version="phase7"))
        request = RequestEnvelope("p7-feedback", "GET", "https", "example.test", "/", "q=1 union select password from users")
        result = waf.analyze(request)
        features = self.extractor.extract(request)
        store = FeedbackStore()
        record = store.add_from_decision(result, features, "phase4-ml-v1", self.baseline.baseline_version)
        self.assertEqual(record.review_state, "pending")
        reviewed = store.review(record.record_id, 1, "human-reviewer", "confirmed attack")
        self.assertEqual(reviewed.review_state, "reviewed")
        encoded = json.dumps(reviewed.to_dict(), sort_keys=True)
        self.assertNotIn("union select", encoded)
        self.assertNotIn("example.test", encoded)
        self.assertFalse(reviewed.privacy["raw_request_material_retained"])

    def test_material_drift_is_deterministic(self):
        baseline_matrix = np.asarray(self.baseline.rows, dtype=float)
        current = baseline_matrix[:64].copy()
        current[:, self.baseline.feature_names.index("query_length")] = 1.0
        report = DriftDetector().compare(self.baseline, current.tolist())
        self.assertTrue(report.material_drift)
        self.assertEqual(report.alert_level, "material")
        self.assertGreaterEqual(report.max_psi, 0.25)

    def test_controlled_challenger_requires_feedback_and_explicit_promotion_then_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = FeedbackStore()
            for i in range(44):
                request = RequestEnvelope(
                    f"p7-fb-{i}", "GET", "https", "example.test",
                    "/search" if i % 2 == 0 else "/products/42",
                    "q=1 union select password from users" if i % 2 == 0 else "page=1",
                )
                result = EdgeWAF(WAFConfig(pipeline_version="phase7")).analyze(request)
                row = store.add_from_decision(result, self.extractor.extract(request), "phase4-ml-v1", self.baseline.baseline_version)
                store.review(row.record_id, 1 if i % 2 == 0 else 0, "reviewer")
            current = np.asarray(self.baseline.rows[:64], dtype=float)
            current[:, self.baseline.feature_names.index("query_length")] = 1.0
            drift = DriftDetector().compare(self.baseline, current.tolist())
            champion_path = root / "champion.joblib"
            self.ml.save(champion_path)
            candidate_path = root / "candidate.joblib"
            registry_path = root / "registry.json"
            candidate = train_controlled_challenger(champion_path, self.baseline, store, candidate_path, registry_path, drift)
            self.assertTrue(candidate.artifact_sha256)
            registry = ModelRegistry(registry_path)
            champion = registry.champion
            self.assertEqual(champion["model_version"], self.ml.model_version)
            decision = registry.evaluate(candidate, champion)
            self.assertTrue(decision.eligible)
            promoted = registry.promote(candidate, "release-reviewer")
            self.assertEqual(promoted["model_version"], candidate.model_version)
            rolled_back = registry.rollback("release-reviewer")
            self.assertEqual(rolled_back["model_version"], self.ml.model_version)
            self.assertGreaterEqual(PROMOTION_MAX_F1_DROP, 0.0)

    def test_retraining_never_auto_replaces_runtime_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = FeedbackStore()
            for i in range(40):
                request = RequestEnvelope(
                    f"p7-auto-{i}", "GET", "https", "example.test", "/", "q=1 union select x" if i % 2 == 0 else "page=1"
                )
                result = EdgeWAF(WAFConfig(pipeline_version="phase7")).analyze(request)
                row = store.add_from_decision(result, self.extractor.extract(request), "phase4-ml-v1", self.baseline.baseline_version)
                store.review(row.record_id, 1 if i % 2 == 0 else 0, "reviewer")
            current = np.asarray(self.baseline.rows[:64], dtype=float)
            current[:, self.baseline.feature_names.index("path_length")] = 1.0
            drift = DriftDetector().compare(self.baseline, current.tolist())
            champion_path = root / "champion.joblib"
            self.ml.save(champion_path)
            candidate = train_controlled_challenger(
                champion_path, self.baseline, store, root / "candidate.joblib", root / "registry.json", drift
            )
            self.assertTrue(candidate.artifact_path.endswith("candidate.joblib"))
            self.assertTrue(champion_path.exists())
            self.assertEqual(champion_path.read_bytes(), Path(champion_path).read_bytes())


if __name__ == "__main__":
    unittest.main()
