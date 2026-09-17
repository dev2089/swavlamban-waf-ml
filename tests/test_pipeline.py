from __future__ import annotations

import unittest

from waf.core.config import WAFConfig
from waf.core.models import Decision, DetectionSignal, FeatureVector, RequestEnvelope
from waf.core.pipeline import WAFPipeline
from waf.features.http import HTTPFeatureExtractor
from waf.storage.memory import InMemoryEventSink
from waf.detection.policy import ThresholdDecisionPolicy


class FixedDetector:
    def __init__(self, name: str, score: float, confidence: float = 1.0, reason: str = "fixed") -> None:
        self.name = name
        self.score = score
        self.confidence = confidence
        self.reason = reason

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        return DetectionSignal(self.name, self.score, self.confidence, (self.reason,), ())


class PipelineTests(unittest.TestCase):
    def make_request(self, **kwargs):
        data = dict(request_id="r-1", method="GET", scheme="http", host="example.test", path="/")
        data.update(kwargs)
        return RequestEnvelope(**data)

    def test_benign_request_is_allowed(self):
        p = WAFPipeline(WAFConfig(), HTTPFeatureExtractor(), ThresholdDecisionPolicy())
        result = p.analyze(self.make_request())
        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertEqual(result.risk_score, 0.0)

    def test_signature_payload_is_blocked(self):
        p = WAFPipeline(WAFConfig(), HTTPFeatureExtractor(), ThresholdDecisionPolicy(), (FixedDetector("ml", 0.9, 1.0),))
        result = p.analyze(self.make_request(query="q=select+password+from+users"))
        self.assertEqual(result.decision, Decision.BLOCK)
        self.assertEqual(result.rule_ids, ())

    def test_event_is_published(self):
        sink = InMemoryEventSink()
        p = WAFPipeline(WAFConfig(), HTTPFeatureExtractor(), ThresholdDecisionPolicy(), (FixedDetector("x", 0.6, 1.0, "anomaly"),), sink)
        result = p.analyze(self.make_request())
        events = sink.snapshot()
        self.assertEqual(result.decision, Decision.ALERT)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["decision"], "alert")

    def test_two_independent_signals_raise_risk(self):
        detectors = (FixedDetector("a", 0.72, 1.0, "a"), FixedDetector("b", 0.72, 1.0, "b"))
        p = WAFPipeline(WAFConfig(), HTTPFeatureExtractor(), ThresholdDecisionPolicy(), detectors)
        result = p.analyze(self.make_request())
        self.assertEqual(result.decision, Decision.BLOCK)
        self.assertGreaterEqual(result.risk_score, 0.80)

    def test_body_is_bounded(self):
        class SizeExtractor:
            def extract(self, request):
                self.size = len(request.body)
                return FeatureVector("http-v2", {"x": 0.0})
        extractor = SizeExtractor()
        p = WAFPipeline(WAFConfig(max_body_bytes=16), extractor, ThresholdDecisionPolicy())
        p.analyze(self.make_request(body=b"x" * 100))
        self.assertEqual(extractor.size, 16)

    def test_schema_mismatch_fails_closed(self):
        class WrongExtractor:
            def extract(self, request):
                return FeatureVector("wrong-v9", {})
        p = WAFPipeline(WAFConfig(), WrongExtractor(), ThresholdDecisionPolicy())
        with self.assertRaises(ValueError):
            p.analyze(self.make_request())

    def test_invalid_signal_rejected(self):
        with self.assertRaises(ValueError):
            DetectionSignal("x", 1.1, 1.0)

    def test_real_signature_detector_blocks_known_pattern(self):
        from waf.detection.rules import SignatureDetector
        p = WAFPipeline(WAFConfig(), HTTPFeatureExtractor(), ThresholdDecisionPolicy(), (SignatureDetector(),))
        result = p.analyze(self.make_request(query="id=1 union select password from users"))
        self.assertEqual(result.decision, Decision.BLOCK)
        self.assertIn("SIG-SQL-001", result.rule_ids)


if __name__ == "__main__":
    unittest.main()
