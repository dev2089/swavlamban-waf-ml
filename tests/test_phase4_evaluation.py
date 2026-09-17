from __future__ import annotations

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.ml.ensemble import evaluate_behaviour, evaluate_semi_supervised, evaluate_supervised, evaluate_unsupervised


def test_supervised_eval_has_explicit_metrics_and_scope():
    result = evaluate_supervised(seed=19)
    assert result["evaluation_scope"] == "deterministic synthetic HTTP benchmark only"
    assert result["test_samples"] == 1500
    assert result["f1"] >= 0.95
    assert 0.0 <= result["fpr"] <= 1.0


def test_semi_supervised_eval_uses_partial_labels_and_has_explicit_metrics():
    result = evaluate_semi_supervised(seed=19)
    assert result["evaluation_scope"] == "deterministic synthetic HTTP benchmark with partial labels"
    assert 0.0 < result["labeled_fraction"] < 1.0
    assert result["unlabeled_samples"] > 0
    assert result["test_samples"] == 1500
    assert result["f1"] >= 0.90
    assert 0.0 <= result["fpr"] <= 1.0


def test_unsupervised_eval_measures_benign_fpr_and_attack_detection():
    result = evaluate_unsupervised(seed=19)
    assert result["evaluation_scope"] == "deterministic synthetic HTTP benchmark only"
    assert result["attack_detection_rate"] >= 0.80
    assert result["false_positive_rate"] <= 0.05
    assert result["mean_attack_score"] > result["mean_benign_score"]


def test_behaviour_eval_proves_learned_burst_escalation():
    result = evaluate_behaviour()
    assert result["burst_escalated"] is True
    assert result["normal_max_score"] < 0.5
    assert result["burst_final_score"] >= 0.5


def test_live_edge_allows_common_benign_request_without_headers():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(RequestEnvelope("benign", "GET", "https", "example.test", "/health"))
    assert result.decision is Decision.ALLOW


def test_live_edge_raises_alert_for_unseen_anomalous_request_without_signature():
    waf = EdgeWAF(WAFConfig())
    request = RequestEnvelope(
        "anomaly", "TRACE", "https", "strange.example", "/" + "A" * 4000,
        "q=" + "Z" * 8000, headers={f"X-{i}": "v" for i in range(128)},
    )
    result = waf.analyze(request)
    anomaly = next(s for s in result.signals if s.detector.startswith("unsupervised-"))
    assert not result.rule_ids
    assert anomaly.score >= 0.5
    assert result.decision is Decision.ALERT


def test_live_edge_blocks_known_attack_even_with_ml_artifact_loaded():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(
        RequestEnvelope(
            "attack", "GET", "https", "example.test", "/search",
            "q=1%20UNION%20SELECT%20password%20FROM%20users",
        )
    )
    assert result.decision is Decision.BLOCK
    assert "WAF-SQL-001" in result.rule_ids
    assert result.risk_score == 1.0
