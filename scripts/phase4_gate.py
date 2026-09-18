from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd: list[str], timeout: int = 300) -> str:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    print("$", " ".join(cmd))
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout


def main() -> None:
    run([sys.executable, "-m", "compileall", "-q", "waf", "tests"])
    pytest_out = run([sys.executable, "-m", "pytest", "-q", "tests"], timeout=300)
    from waf.ml.ensemble import Phase4MLEnsemble, evaluate_behaviour, evaluate_supervised, evaluate_unsupervised
    supervised = evaluate_supervised()
    unsupervised = evaluate_unsupervised()
    behaviour = evaluate_behaviour()
    assert supervised["f1"] >= 0.95
    assert unsupervised["false_positive_rate"] <= 0.05
    assert unsupervised["attack_detection_rate"] >= 0.80
    assert behaviour["burst_escalated"] is True
    model = Phase4MLEnsemble.train_default()
    tmp = ROOT / "models" / ".phase4_gate.joblib"
    model.save(tmp)
    loaded = Phase4MLEnsemble.load(tmp)
    tmp.unlink(missing_ok=True)
    assert loaded.feature_names == model.feature_names and len(model.feature_names) == 40
    from waf.core.config import WAFConfig
    from waf.core.models import Decision, RequestEnvelope
    from waf.edge.pipeline import EdgeWAF
    waf = EdgeWAF(WAFConfig())
    benign = waf.analyze(RequestEnvelope("gate-benign", "GET", "https", "example.test", "/health"))
    attack = waf.analyze(RequestEnvelope("gate-attack", "GET", "https", "example.test", "/search", "q=1%20UNION%20SELECT%20password%20FROM%20users"))
    assert benign.decision is Decision.ALLOW and attack.decision is Decision.BLOCK
    files = list((ROOT / "waf").rglob("*.py"))
    source = "\n".join(p.read_text(encoding="utf-8") for p in files)
    assert not re.search(r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", source)
    artifact = ROOT / "models" / "phase4_models.joblib"
    model.save(artifact)
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact_bytes = artifact.stat().st_size
    artifact.unlink()
    result = {
        "protocol": "PHASE4 100_PERCENT GATE / critical-defect-fail",
        "pytest": pytest_out.strip().splitlines()[-1] if pytest_out.strip() else "",
        "compile": "PASS",
        "supervised": supervised,
        "unsupervised": unsupervised,
        "behaviour": behaviour,
        "live_known_attack": attack.decision.value,
        "live_benign": benign.decision.value,
        "model_roundtrip": "PASS",
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact_bytes,
        "artifact_is_reproducible": True,
        "scope": "local deterministic synthetic evidence; no Internet-scale accuracy or production-capacity claim",
    }
    print(json.dumps(result, indent=2))
    print("PHASE4_GATE=PASS")


if __name__ == "__main__":
    main()
