from __future__ import annotations

import time

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.behaviour import BehaviouralDetector
from waf.ml.ensemble import Phase4MLEnsemble, evaluate_supervised


def request(**kwargs):
    data = dict(
        request_id="ml-1",
        method="GET",
        scheme="https",
        host="example.test",
        path="/health",
        source_ip="127.0.0.1",
    )
    data.update(kwargs)
    return RequestEnvelope(**data)


def test_supervised_training_and_metrics_are_computed():
    result = evaluate_supervised(seed=7)
    assert result["evaluation_scope"] == "deterministic synthetic HTTP benchmark only"
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= result[key] <= 1.0
    assert result["f1"] >= 0.95


def test_unsupervised_detector_is_fitted_and_returns_bounded_score():
    ensemble = Phase4MLEnsemble.train_default()
    fv = ProductionHTTPFeatureExtractor().extract(request())
    sig = ensemble.anomaly.detect(request(), fv)
    assert sig.detector.startswith("unsupervised-")
    assert 0.0 <= sig.score <= 1.0
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.metadata["baseline_version"].startswith("benign-baseline-v4")


def test_learned_behaviour_detector_escalates_burst():
    detector = BehaviouralDetector.train_default()
    extractor = ProductionHTTPFeatureExtractor()
    scores = []
    for i in range(40):
        req = request(request_id=f"b-{i}", path=f"/api/{i % 10}", timestamp=1000.0 + i * 0.05)
        scores.append(detector.detect(req, extractor.extract(req)).score)
    assert scores[-1] > scores[0]
    assert scores[-1] >= 0.5


def test_ensemble_has_all_ml_signals_and_preserves_signature_block():
    waf = EdgeWAF(WAFConfig())
    benign = waf.analyze(request())
    assert benign.decision is Decision.ALLOW
    assert {s.detector for s in benign.signals} == {
        "open-source-waf-rules", "supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1"
    }
    bad = waf.analyze(request(query="q=1%20UNION%20SELECT%20password%20FROM%20users"))
    assert bad.decision is Decision.BLOCK
    assert "WAF-SQL-001" in bad.rule_ids
    assert bad.risk_score == 1.0


def test_ml_stateless_components_are_cached_but_behaviour_state_is_isolated():
    start = time.perf_counter()
    a = EdgeWAF(WAFConfig())
    first = time.perf_counter() - start
    start = time.perf_counter()
    b = EdgeWAF(WAFConfig())
    second = time.perf_counter() - start
    assert a.ml.supervised.model is b.ml.supervised.model
    assert a.ml.anomaly.model is b.ml.anomaly.model
    assert a.ml.behaviour is not b.ml.behaviour
    assert a.ml.behaviour.model is b.ml.behaviour.model
    assert second < max(1.0, first * 0.5)


def test_model_scores_are_bounded_for_benign_and_known_attack():
    waf = EdgeWAF(WAFConfig())
    benign = waf.analyze(request())
    attack = waf.analyze(request(query="q=1%20UNION%20SELECT%20password%20FROM%20users"))
    for result in (benign, attack):
        assert 0.0 <= result.risk_score <= 1.0
        assert all(0.0 <= signal.score <= 1.0 for signal in result.signals)
        assert all(0.0 <= signal.confidence <= 1.0 for signal in result.signals)
