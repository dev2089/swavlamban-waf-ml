from pathlib import Path

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.ml.ensemble import Phase4MLEnsemble


def test_model_artifact_roundtrip(tmp_path: Path):
    model = Phase4MLEnsemble.train_default()
    path = tmp_path / "models.joblib"
    model.save(path)
    loaded = Phase4MLEnsemble.load(path)
    assert loaded.model_version == model.model_version
    assert loaded.feature_names == model.feature_names
    req = RequestEnvelope("r", "GET", "https", "example.test", "/health")
    fv = EdgeWAF(WAFConfig()).features.extract(req)
    assert loaded.supervised.detect(req, fv).detector == "supervised-v1"
    assert loaded.anomaly.detect(req, fv).detector.startswith("unsupervised-")
    assert loaded.behaviour.name == "behaviour-v1"
    assert loaded.behaviour.model is not None
    assert loaded.behaviour.model.__class__.__name__ == "LogisticRegression"
    assert loaded.semi_supervised.name == "semi-supervised-v1"
    assert loaded.semi_supervised.model is not None


def test_edge_can_consume_pretrained_artifact(tmp_path: Path, monkeypatch):
    model = Phase4MLEnsemble.train_default()
    path = tmp_path / "phase4.joblib"
    model.save(path)
    monkeypatch.setenv("WAF_MODEL_ARTIFACT", str(path))
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(RequestEnvelope("r2", "GET", "https", "example.test", "/health"))
    assert result.decision is Decision.ALLOW
    assert {s.detector for s in result.signals} >= {
        "supervised-v1",
        "semi-supervised-v1",
        "behaviour-v1",
    }
