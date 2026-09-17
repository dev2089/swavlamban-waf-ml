from __future__ import annotations

from functools import lru_cache

from waf.ml.anomaly import UnsupervisedAnomalyDetector
from waf.ml.behaviour import BehaviouralDetector
from waf.ml.dataset import build_benign_baseline, build_training_dataset
from waf.ml.semisupervised import SemiSupervisedDetector
from waf.ml.supervised import SupervisedDetector


@lru_cache(maxsize=1)
def trained_models() -> tuple[SupervisedDetector, UnsupervisedAnomalyDetector, BehaviouralDetector, SemiSupervisedDetector, tuple[str, ...], str, str]:
    bundle = build_training_dataset(samples=6000, seed=42)
    benign_X, feature_names, baseline_version = build_benign_baseline(samples=2600, seed=123)
    supervised = SupervisedDetector.train(bundle.X, bundle.y, bundle.feature_names, bundle.dataset_version)
    anomaly = UnsupervisedAnomalyDetector.train(benign_X, feature_names, baseline_version)
    behaviour = BehaviouralDetector.train_default(seed=42)
    semi = SemiSupervisedDetector.train(bundle.X, bundle.y, bundle.feature_names, bundle.dataset_version, labeled_fraction=0.30, seed=42)
    return supervised, anomaly, behaviour, semi, bundle.feature_names, bundle.dataset_version, baseline_version
