from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waf.ml.ensemble import Phase4MLEnsemble, evaluate_behaviour, evaluate_supervised, evaluate_unsupervised


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist Phase 4 ML models")
    parser.add_argument("--output", default="models/phase4_models.joblib")
    args = parser.parse_args()
    ensemble = Phase4MLEnsemble.train_default()
    output = Path(args.output)
    ensemble.save(output)
    manifest = {
        **ensemble.metadata(),
        "artifact": str(output),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "artifact_bytes": output.stat().st_size,
        "supervised_evaluation": evaluate_supervised(),
        "unsupervised_evaluation": evaluate_unsupervised(),
        "behaviour_evaluation": evaluate_behaviour(),
        "training_scope": "deterministic synthetic HTTP benchmark only",
        "behaviour_model_scope": "deterministic synthetic behavioural workload",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
