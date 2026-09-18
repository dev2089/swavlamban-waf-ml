from pathlib import Path

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.ml.ensemble import Phase4MLEnsemble


def test_model_artifact_roundtrip_and_runtime(tmp_path: Path, monkeypatch):
    model = Phase4MLEnsemble.train_default()
    path = tmp_path / "phase4_models.joblib"
    model.save(path)
    monkeypatch.setenv("WAF_MODEL_ARTIFACT", str(path))
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(RequestEnvelope("r", "GET", "https", "example.test", "/health"))
    assert result.decision is Decision.ALLOW
    assert len(model.feature_names) == 40
