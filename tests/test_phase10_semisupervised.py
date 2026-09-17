from __future__ import annotations

from waf.core.models import RequestEnvelope
from waf.core.config import WAFConfig
from waf.edge.pipeline import EdgeWAF
from waf.ml.dataset import build_training_dataset
from waf.ml.semisupervised import SemiSupervisedDetector


def test_semi_supervised_model_uses_unlabeled_training_rows_and_scores_attack():
    bundle = build_training_dataset(samples=800, seed=7)
    detector = SemiSupervisedDetector.train(
        bundle.X,
        bundle.y,
        bundle.feature_names,
        bundle.dataset_version,
        labeled_fraction=0.30,
        seed=7,
    )
    benign = RequestEnvelope("benign", "GET", "https", "example.test", "/health")
    attack = RequestEnvelope("attack", "GET", "https", "example.test", "/search", "q=1%20UNION%20SELECT%20password")
    waf = EdgeWAF(WAFConfig())
    benign_fv = waf.features.extract(benign)
    attack_fv = waf.features.extract(attack)
    benign_signal = detector.detect(benign, benign_fv)
    attack_signal = detector.detect(attack, attack_fv)
    assert detector.labeled_fraction == 0.3
    assert detector.name == "semi-supervised-v1"
    assert benign_signal.metadata["unlabeled_training_used"] is True
    assert attack_signal.score > benign_signal.score


def test_edge_exposes_semi_supervised_signal():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(RequestEnvelope("r", "GET", "https", "example.test", "/health"))
    semi = next(signal for signal in result.signals if signal.detector == "semi-supervised-v1")
    assert 0.0 <= semi.score <= 1.0
    assert semi.metadata["unlabeled_training_used"] is True
