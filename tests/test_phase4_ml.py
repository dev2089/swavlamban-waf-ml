from __future__ import annotations

from pathlib import Path

import pytest

from waf.core.config import WAFConfig
from waf.core.models import Decision, FeatureVector, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.behaviour import BehaviouralDetector
from waf.ml.ensemble import Phase4MLEnsemble, evaluate_behaviour, evaluate_supervised, evaluate_unsupervised


def req(**kwargs) -> RequestEnvelope:
    data = dict(request_id="ml-1", method="GET", scheme="https", host="example.test", path="/health", source_ip="127.0.0.1", timestamp=1000.0)
    data.update(kwargs)
    return RequestEnvelope(**data)


def test_supervised_evaluation_is_held_out_and_deterministic():
    result = evaluate_supervised(seed=7)
    assert result["test_samples"] == 1500
    assert result["evaluation_scope"] == "deterministic synthetic HTTP benchmark only"
    assert result["f1"] >= 0.95


def test_unsupervised_uses_benign_only_fit_and_reports_fpr():
    result = evaluate_unsupervised(seed=19)
    assert result["evaluation_scope"] == "deterministic synthetic HTTP benchmark only"
    assert result["false_positive_rate"] <= 0.05
    assert result["attack_detection_rate"] >= 0.80
    assert result["mean_attack_score"] > result["mean_benign_score"]


def test_behavioural_model_escalates_a_burst():
    result = evaluate_behaviour()
    assert result["burst_escalated"] is True
    assert result["normal_max_score"] < 0.5
    assert result["burst_final_score"] >= 0.5


def test_live_edge_exposes_all_four_security_signals():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(req())
    assert result.decision is Decision.ALLOW
    assert {s.detector for s in result.signals} == {"open-source-waf-rules", "supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1"}


def test_live_edge_known_attack_remains_hard_block():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(req(query="q=1%20UNION%20SELECT%20password%20FROM%20users"))
    assert result.decision is Decision.BLOCK
    assert result.risk_score == 1.0
    assert "WAF-SQL-001" in result.rule_ids


def test_live_edge_unknown_anomaly_can_alert_without_signature():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(req(method="TRACE", host="strange.example", path="/" + "A" * 4000, query="q=" + "Z" * 8000, headers={f"X-{i}": "v" for i in range(128)}))
    anomaly = next(s for s in result.signals if s.detector == "unsupervised-oneclasssvm-v1")
    assert not result.rule_ids
    assert anomaly.score >= 0.5
    assert result.decision is Decision.ALERT


def test_model_artifact_roundtrip_validates_schema(tmp_path: Path):
    model = Phase4MLEnsemble.train_default()
    path = tmp_path / "phase4.joblib"
    model.save(path)
    loaded = Phase4MLEnsemble.load(path)
    assert loaded.feature_names == model.feature_names
    assert len(loaded.feature_names) == 40
    assert loaded.model_version == "phase4-ml-v1"


def test_bad_model_artifact_schema_is_rejected(tmp_path: Path):
    import joblib
    model = Phase4MLEnsemble.train_default()
    path = tmp_path / "bad.joblib"
    model.save(path)
    payload = joblib.load(path)
    payload["feature_names"] = tuple(payload["feature_names"][:-1])
    joblib.dump(payload, path)
    with pytest.raises(ValueError, match="40 features"):
        Phase4MLEnsemble.load(path)


def test_runtime_behaviour_state_is_isolated():
    a = EdgeWAF(WAFConfig())
    b = EdgeWAF(WAFConfig())
    assert a.ml.supervised.model is b.ml.supervised.model
    assert a.ml.anomaly.model is b.ml.anomaly.model
    assert a.ml.behaviour is not b.ml.behaviour
    assert a.ml.behaviour.model is b.ml.behaviour.model


def test_all_runtime_scores_are_bounded():
    waf = EdgeWAF(WAFConfig())
    for result in (waf.analyze(req()), waf.analyze(req(query="q=1%20UNION%20SELECT%20password%20FROM%20users"))):
        assert 0.0 <= result.risk_score <= 1.0
        assert all(0.0 <= s.score <= 1.0 for s in result.signals)
        assert all(0.0 <= s.confidence <= 1.0 for s in result.signals)


def test_feature_schema_rejects_missing_feature_in_detector():
    model = Phase4MLEnsemble.train_default().supervised
    fv = FeatureVector("http-v2", {name: 0.0 for name in model.feature_names[:-1]})
    with pytest.raises(ValueError, match="missing feature"):
        model.detect(req(), fv)


def test_behaviour_state_is_bounded_per_source():
    detector = BehaviouralDetector.train_default()
    extractor = ProductionHTTPFeatureExtractor()
    for i in range(300):
        r = req(request_id=f"r-{i}", path=f"/p/{i}", source_ip="10.0.0.1", timestamp=2000.0 + i * 0.01)
        detector.detect(r, extractor.extract(r))
    assert len(detector._sources["10.0.0.1"].events) <= detector.max_events_per_source


def test_ml_inference_failure_fails_closed(monkeypatch):
    waf = EdgeWAF(WAFConfig())
    class BrokenML:
        def detect(self, request, features):
            raise RuntimeError("simulated model failure")
    waf.ml = BrokenML()
    result = waf.analyze(req())
    assert result.decision is Decision.BLOCK
    failure = next(s for s in result.signals if s.detector == "ml-runtime-failure")
    assert failure.score == 1.0
    assert failure.metadata["error_type"] == "RuntimeError"
