#!/usr/bin/env python3
"""Executable Phase 7 acceptance gate.

Runs the full regression, compile gate, dedicated Phase 7 tests, then proves a
bounded learning-control loop: versioned benign baseline -> reviewed feedback
-> material drift alert -> controlled challenger training -> champion/challenger
comparison -> explicit human promotion -> rollback. Raw HTTP material must not
enter baseline/feedback/drift/model-control evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULT = ROOT / "phase7_master_exam_result.json"
CUTOFF = 9.9


def run(name: str, cmd: list[str]) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    combined = (p.stdout + "\n" + p.stderr).strip()
    return {"name": name, "passed": p.returncode == 0, "returncode": p.returncode, "tail": combined[-3000:]}


def learning_loop_smoke() -> dict[str, Any]:
    from waf.core.config import WAFConfig
    from waf.core.models import RequestEnvelope
    from waf.edge.pipeline import EdgeWAF
    from waf.features.http_v2 import ProductionHTTPFeatureExtractor
    from waf.ml.learning_control import DriftDetector, FeedbackStore, ModelRegistry, build_default_baseline, train_controlled_challenger
    import numpy as np

    baseline_path = ROOT / "data/phase7_benign_baseline.json"
    baseline = build_default_baseline(baseline_path, samples=2200, seed=123)
    extractor = ProductionHTTPFeatureExtractor()
    waf = EdgeWAF(WAFConfig(pipeline_version="phase7-master-exam"))
    feedback = FeedbackStore()

    for i in range(44):
        request = RequestEnvelope(f"phase7-smoke-{i}", "GET", "https", "example.test", "/search" if i % 2 == 0 else "/products/42", "q=1 union select x" if i % 2 == 0 else "page=1")
        result = waf.analyze(request)
        feature_vector = extractor.extract(request)
        pending = feedback.add_from_decision(result, feature_vector, waf.ml.model_version if hasattr(waf.ml, "model_version") else "phase4-ml-v1", baseline.baseline_version)
        feedback.review(pending.record_id, 1 if i % 2 == 0 else 0, "phase7-human-reviewer", "deterministic reviewed sample")

    baseline_matrix = np.asarray(baseline.rows, dtype=float)
    current = baseline_matrix[:64].copy()
    current[:, baseline.feature_names.index("query_length")] = 1.0
    drift = DriftDetector().compare(baseline, current.tolist())
    if not drift.material_drift:
        raise AssertionError("controlled smoke workload did not trigger material drift")

    champion_path = ROOT / "models/phase4_models.joblib"
    champion_before = hashlib.sha256(champion_path.read_bytes()).hexdigest()
    candidate_path = ROOT / "models/phase7/challenger.joblib"
    registry_path = ROOT / "models/phase7/registry.json"
    candidate = train_controlled_challenger(champion_path, baseline, feedback, candidate_path, registry_path, drift, seed=42)
    registry = ModelRegistry(registry_path)
    champion = registry.champion
    if champion is None:
        raise AssertionError("champion registry was not initialized")
    decision = registry.evaluate(candidate, champion)
    if not decision.eligible:
        raise AssertionError(f"challenger failed promotion eligibility: {decision.reasons}")
    promoted = registry.promote(candidate, "phase7-release-reviewer")
    rolled_back = registry.rollback("phase7-release-reviewer")
    champion_after = hashlib.sha256(champion_path.read_bytes()).hexdigest()
    if champion_before != champion_after:
        raise AssertionError("default runtime champion artifact was silently replaced")
    if rolled_back["model_version"] != champion["model_version"]:
        raise AssertionError("rollback did not restore the prior champion metadata")

    edge_result = waf.analyze(RequestEnvelope("phase7-provenance", "GET", "https", "example.test", "/health"))
    evidence = edge_result.evidence
    if evidence is None:
        raise AssertionError("Phase 7 edge provenance evidence missing")
    expected = {"learning_control_schema": "phase7-learning-control-v1", "learning_model_version": "phase4-ml-v1", "learning_baseline_version": waf.ml.baseline_version}
    for key, value in expected.items():
        if evidence.versions.get(key) != value:
            raise AssertionError(f"evidence provenance mismatch for {key}")

    feedback_text = json.dumps(feedback.export_json(), sort_keys=True)
    for forbidden in ("union select x", "example.test", '"body"', '"query"', '"headers"', '"host"', '"source_ip"'):
        if forbidden in feedback_text:
            raise AssertionError(f"forbidden raw request material leaked into feedback evidence: {forbidden}")
    return {"baseline_version": baseline.baseline_version, "baseline_samples": baseline.sample_count, "reviewed_feedback": len(feedback.reviewed()), "drift_report_id": drift.report_id, "drift_mean_psi": drift.mean_psi, "drift_max_psi": drift.max_psi, "material_drift": drift.material_drift, "candidate_run_id": candidate.run_id, "candidate_model_version": candidate.model_version, "candidate_evaluation": dict(candidate.evaluation), "promotion_eligible": decision.eligible, "promoted_model_version": promoted["model_version"], "rolled_back_model_version": rolled_back["model_version"], "default_runtime_artifact_unchanged": champion_before == champion_after, "raw_feedback_material_recorded": False, "evidence_provenance": expected}


def privacy_gate() -> dict[str, Any]:
    source = (ROOT / "waf/ml/learning_control.py").read_text(encoding="utf-8")
    forbidden_access = ["request.body", "request.query", "request.headers", "request.host", "request.source_ip"]
    bad = [item for item in forbidden_access if item in source]
    if bad:
        raise AssertionError(f"learning-control path directly accesses raw request fields: {bad}")
    if "raw_request_material_retained" not in source:
        raise AssertionError("learning-control privacy contract is missing")
    if "requires_explicit_human_approval" not in source:
        raise AssertionError("human approval contract is missing")
    if "runtime_default_replaced_automatically" not in source:
        raise AssertionError("no-auto-promotion contract is missing")
    return {"passed": True, "checked_file": "waf/ml/learning_control.py", "raw_request_field_access": bad}


def main() -> int:
    checks = [run("full-regression", [sys.executable, "-m", "pytest", "-q"]), run("compileall", [sys.executable, "-m", "compileall", "-q", "waf", "tests", "scripts"]), run("phase7-learning-control-tests", [sys.executable, "-m", "pytest", "-q", "tests/test_phase7_learning_control.py"])]
    critical: list[str] = []
    smoke: dict[str, Any] = {}
    try:
        smoke = learning_loop_smoke()
        checks.append({"name": "learning-control-smoke", "passed": True, "tail": json.dumps(smoke, sort_keys=True)})
    except Exception as exc:
        checks.append({"name": "learning-control-smoke", "passed": False, "tail": repr(exc)})
        critical.append("learning-control-smoke")
    try:
        privacy = privacy_gate()
        checks.append({"name": "privacy-static-gate", "passed": True, "tail": json.dumps(privacy, sort_keys=True)})
    except Exception as exc:
        checks.append({"name": "privacy-static-gate", "passed": False, "tail": repr(exc)})
        critical.append("privacy-static-gate")
    passed = sum(1 for check in checks if check.get("passed"))
    score = round((passed / len(checks)) * 10.0, 2)
    if score < CUTOFF:
        critical.append("score-below-cutoff")
    result = {"phase": 7, "status": "PASS" if score >= CUTOFF and not critical else "FAIL", "score": score, "cutoff": CUTOFF, "critical_defects": critical, "checks": checks, "smoke": smoke, "evaluation_scope": "repository regression, compile gate, dedicated Phase 7 tests, deterministic learning-control smoke and privacy/static gate", "notes": ["Baseline uses the versioned http-v2 numeric feature contract and does not persist raw request material.", "Feedback requires explicit human review before a label can enter controlled retraining.", "Drift uses deterministic PSI thresholds: material when mean PSI >= 0.10 or max feature PSI >= 0.25, with at least 32 samples.", "Challenger evaluation is compared with the frozen champion on the same deterministic synthetic validation benchmark.", "Promotion requires explicit human approval and never silently replaces models/phase4_models.joblib.", "Rollback restores the prior champion artifact and metadata; production model storage/auth/RBAC remains a later phase."]}
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
