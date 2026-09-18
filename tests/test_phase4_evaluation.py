from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.ml.ensemble import evaluate_behaviour, evaluate_supervised, evaluate_unsupervised


def test_evaluations_have_explicit_scopes():
    assert evaluate_supervised()["evaluation_scope"] == "deterministic synthetic HTTP benchmark only"
    assert evaluate_unsupervised()["evaluation_scope"] == "deterministic synthetic HTTP benchmark only"
    assert evaluate_behaviour()["evaluation_scope"] == "deterministic synthetic behavioural workload"


def test_config_phase4_defaults():
    cfg = WAFConfig()
    assert cfg.pipeline_version == "phase4"
    assert cfg.feature_schema_version == "http-v2"


def test_live_edge_preserves_http_v2():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(RequestEnvelope("p4", "GET", "https", "example.test", "/health"))
    assert result.pipeline_version == "phase4"
