from __future__ import annotations

import json
import unittest

from waf.core.config import WAFConfig
from waf.core.models import Decision, DecisionResult, DetectionSignal, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.edge.policy import EdgeDecisionPolicy
from waf.edge.rules import OpenSourceWAFRuleEngine
from waf.explainability import build_decision_evidence, evidence_to_dict
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.ensemble import Phase4MLEnsemble
from waf.telemetry.events import decision_event


class Phase5EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.extractor = ProductionHTTPFeatureExtractor()
        self.ml = Phase4MLEnsemble.train_default()
        self.policy = EdgeDecisionPolicy()

    def analyze(self, request: RequestEnvelope, runtime=None):
        runtime = runtime or self.ml
        features = self.extractor.extract(request)
        signature = OpenSourceWAFRuleEngine().detect(request, features)
        result = self.policy.decide(request, (signature, *runtime.detect(request, features)), WAFConfig().pipeline_version)
        return result, features

    def test_known_attack_has_reproducible_rule_evidence(self):
        request = RequestEnvelope("p5-attack", "GET", "https", "example.test", "/", "id=1 union select password from users")
        result, features = self.analyze(request)
        evidence = build_decision_evidence(request, features, result, self.ml)
        self.assertEqual(result.decision, Decision.BLOCK)
        self.assertEqual(evidence.schema_version, "evidence-v1")
        self.assertIn("WAF-SQL-001", evidence.rule_ids)
        self.assertEqual(evidence.privacy["raw_payload_retained"], False)
        self.assertEqual(evidence.privacy["raw_query_retained"], False)
        self.assertIn("signature rule(s)", evidence.explanation)

    def test_benign_evidence_contains_no_raw_payload(self):
        request = RequestEnvelope(
            "p5-benign", "POST", "https", "example.test", "/api/profile",
            "view=summary", {"content-type": "application/json"}, b'{"name":"private-value","city":"test"}'
        )
        result, features = self.analyze(request)
        evidence = build_decision_evidence(request, features, result, self.ml)
        encoded = json.dumps(evidence_to_dict(evidence), sort_keys=True)
        self.assertNotIn("private-value", encoded)
        self.assertNotIn("example.test", encoded)
        self.assertNotIn("view=summary", encoded)
        self.assertNotIn("content-type", encoded)
        self.assertEqual(evidence.privacy["raw_headers_retained"], False)
        self.assertTrue(evidence.feature_snapshot)

    def test_unseen_anomaly_path_has_detector_attribution(self):
        request = RequestEnvelope(
            "p5-anomaly", "GET", "https", "example.test", "/%25%25%25/", "x=" + "%25" * 40,
            source_ip="10.10.10.10", timestamp=100.0
        )
        result, features = self.analyze(request)
        evidence = build_decision_evidence(request, features, result, self.ml)
        detectors = {row["detector"] for row in evidence.detector_contributions}
        self.assertIn("supervised-v1", detectors)
        self.assertIn("unsupervised-oneclasssvm-v1", detectors)
        self.assertIn("behaviour-v1", detectors)
        self.assertEqual(set(evidence.feature_attribution), {
            "supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1", "open-source-waf-rules"
        })

    def test_evidence_is_deterministic_for_same_decision(self):
        request = RequestEnvelope("p5-deterministic", "GET", "https", "example.test", "/", "page=1", source_ip="10.0.0.90", timestamp=100.0)
        result, features = self.analyze(request)
        evidence1 = build_decision_evidence(request, features, result, self.ml)
        evidence2 = build_decision_evidence(request, features, result, self.ml)
        self.assertEqual(evidence1.risk_score, evidence2.risk_score)
        self.assertEqual(evidence1.feature_snapshot, evidence2.feature_snapshot)
        self.assertEqual(evidence1.feature_attribution, evidence2.feature_attribution)
        self.assertEqual(evidence1.versions, evidence2.versions)

    def test_live_edge_attaches_evidence_to_every_decision(self):
        waf = EdgeWAF(WAFConfig())
        requests = (
            RequestEnvelope("live-allow", "GET", "https", "example.test", "/health"),
            RequestEnvelope("live-block", "GET", "https", "example.test", "/", "q=1 union select password from users"),
            RequestEnvelope("live-anomaly", "TRACE", "https", "strange.example", "/" + "A" * 4000, "q=" + "Z" * 8000),
        )
        for request in requests:
            result = waf.analyze(request)
            self.assertIsNotNone(result.evidence)
            self.assertEqual(result.evidence.request_id, request.request_id)
            self.assertEqual(result.evidence.versions["evidence_schema"], "evidence-v1")
            self.assertEqual(result.evidence.versions["pipeline_version"], "phase5")

    def test_telemetry_contains_evidence_when_attached(self):
        waf = EdgeWAF(WAFConfig())
        result = waf.analyze(RequestEnvelope("telemetry", "GET", "https", "example.test", "/health"))
        event = decision_event(result)
        self.assertEqual(event["schema_version"], "event-v2")
        self.assertIn("evidence", event)
        self.assertEqual(event["evidence"]["schema_version"], "evidence-v1")
        self.assertFalse(event["evidence"]["privacy"]["raw_payload_retained"])

    def test_privacy_model_rejects_declared_raw_retention(self):
        request = RequestEnvelope("privacy", "GET", "https", "example.test", "/")
        result, features = self.analyze(request)
        evidence = build_decision_evidence(request, features, result, self.ml)
        with self.assertRaises(ValueError):
            type(evidence)(
                schema_version="evidence-v1",
                decision=evidence.decision,
                risk_score=evidence.risk_score,
                detector_contributions=evidence.detector_contributions,
                feature_groups=evidence.feature_groups,
                feature_attribution=evidence.feature_attribution,
                reasons=evidence.reasons,
                rule_ids=evidence.rule_ids,
                versions=evidence.versions,
                explanation=evidence.explanation,
                privacy={**evidence.privacy, "raw_payload_retained": True},
                request_id=evidence.request_id,
                feature_snapshot=evidence.feature_snapshot,
            )

    def test_legacy_result_remains_valid_without_evidence(self):
        result = DecisionResult(
            Decision.ALLOW, 0.0, (), (),
            (DetectionSignal("fixed", 0.0, 1.0),), "legacy", "phase4-v1"
        )
        self.assertIsNone(result.evidence)


if __name__ == "__main__":
    unittest.main()
