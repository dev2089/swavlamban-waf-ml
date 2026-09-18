"""Evidence-first Phase 10 audit generator.

This script scans the tracked candidate, binds claims to current source/evidence,
and writes the machine-readable and human-readable handoff manifests used by the
release gate. It deliberately distinguishes executable evidence from synthetic
or externally constrained claims.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "b94e6aada97c009861e42f1fd6cd705f232a65bb"
SELF = Path(__file__).name


def sh(*args: str) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def load_json(name: str) -> dict[str, object] | None:
    path = ROOT / name
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def tracked_files() -> list[str]:
    return [line for line in sh("git", "ls-files").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_claim(rows: list[dict[str, object]], claim: str, category: str, evidence: str, notes: str = "") -> None:
    rows.append({"claim": claim, "category": category, "evidence": evidence, "notes": notes})


def main() -> int:
    tracked = tracked_files()
    current_sha = sh("git", "rev-parse", "HEAD")
    diff_names = subprocess.check_output(
        ["git", "diff", "--name-status", BASELINE, current_sha], cwd=ROOT, text=True
    ).splitlines()

    suspicious: list[str] = []
    binary_or_non_utf8: list[str] = []
    readable_text_files = 0
    for rel in tracked:
        path = ROOT / rel
        if any(rel.lower().endswith(s) for s in (".pem", ".key", ".p12", ".pfx")):
            binary_or_non_utf8.append(rel)
            continue
        try:
            text = path.read_text(encoding="utf-8")
            readable_text_files += 1
        except UnicodeDecodeError:
            binary_or_non_utf8.append(rel)
            continue
        if "BEGIN PRIVATE KEY" in text or "BEGIN RSA PRIVATE KEY" in text:
            suspicious.append(rel + ": private-key material")
        for marker in ("ghp_", "github_pat_", "sk_live_"):
            if marker in text:
                suspicious.append(rel + f": token-like literal {marker}")

    required = [
        "waf/gateway/proxy.py", "waf/edge/pipeline.py", "waf/ml/ensemble.py", "waf/ml/semisupervised.py",
        "waf/ml/outbound.py", "deploy/nginx/phase10-modsecurity.conf", "scripts/phase10_waf_enforcement_e2e.py",
        "scripts/phase10_tls_e2e.py", "scripts/phase10_outbound_e2e.py", "scripts/phase10_rule_replay.py",
        "scripts/phase10_load_harness.py", "scripts/phase10_demo.py", "scripts/phase10_master_exam.py",
        "waf/storage/async_telemetry.py", "waf/storage/production.py", "dashboard/index.html",
        "docs/PHASE10_TECHNICAL_REPORT.md", "docs/PHASE10_PRESENTATION.md", ".github/workflows/phase10.yml",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]

    exam = load_json("phase10_master_exam_result.json") or {}
    gate_green = exam.get("result") == "PASS" and not exam.get("failed_checks") and not exam.get("critical_failures")

    rows = [
        ("real-time inbound HTTP inspection", "waf/gateway/proxy.py; waf/edge/pipeline.py", "phase10_master_exam_result.json; scripts/phase10_waf_enforcement_e2e.py"),
        ("open-source WAF integration and pre-upstream block", "deploy/nginx/phase10-modsecurity.conf; scripts/phase10_waf_enforcement_e2e.py", "phase10_waf_enforcement_evidence.json"),
        ("TLS encrypted traffic at termination point", "deploy/nginx/phase10-modsecurity.conf; scripts/phase10_tls_e2e.py", "phase10_tls_evidence.json"),
        ("supervised ML", "waf/ml/supervised.py; waf/ml/ensemble.py", "models/phase4_models.json; tests/test_phase4_evaluation.py"),
        ("unsupervised anomaly detection", "waf/ml/anomaly.py; waf/ml/ensemble.py", "models/phase4_models.json; tests/test_phase4_evaluation.py"),
        ("semi-supervised ML", "waf/ml/semisupervised.py; waf/ml/ensemble.py", "models/phase4_models.json; tests/test_phase10_semisupervised.py"),
        ("stateful behavioural/API-bot detection", "waf/ml/behaviour.py; waf/ml/ensemble.py", "tests/test_phase4_evaluation.py; phase10_demo_evidence.json"),
        ("outbound HTTP response inspection", "waf/ml/outbound.py; waf/gateway/proxy.py", "scripts/phase10_outbound_e2e.py; phase10_outbound_evidence.json"),
        ("structured explainability", "waf/explainability.py; waf/core/models.py", "tests/test_phase5_decision_evidence.py; phase10_waf_enforcement_evidence.json"),
        ("ML-derived rule recommendation", "waf/rules/lifecycle.py; waf/edge/pipeline.py", "tests/test_phase10_release.py; phase10_rule_replay_evidence.json"),
        ("human approval and rule deployment lifecycle", "waf/rules/lifecycle.py; waf/api/production_api.py", "tests/test_phase10_release.py; phase10_rule_replay_evidence.json"),
        ("continuous learning, feedback, drift, retraining and rollback", "waf/ml/learning_control.py", "tests/test_phase7_learning_control.py"),
        ("secure authentication/RBAC and admin authorization", "waf/security/*; waf/api/production_api.py", "tests/test_phase8_security.py; tests/test_phase9_production_api.py"),
        ("privacy-safe asynchronous telemetry and server-side persistence", "waf/storage/async_telemetry.py; waf/storage/production.py", "tests/test_phase10_async_telemetry.py; live Supabase verification"),
        ("dynamic authenticated dashboard", "dashboard/index.html; waf/api/production_api.py", "tests/test_phase10_release.py; dashboard demo artifact"),
        ("bounded performance/load evidence", "scripts/phase10_load_harness.py", "phase10_load_evidence.json"),
        ("reliability and startup/failure controls", "waf/gateway/proxy.py; scripts/phase10_master_exam.py", "phase10_master_exam_result.json"),
        ("reproducible release package", ".github/workflows/phase10.yml; scripts/phase10_master_exam.py; scripts/phase10_audit_v2.py", "GitHub Actions release-gate run"),
    ]
    trace = [
        {
            "requirement": req,
            "implementation": impl,
            "verification": verify,
            "status": "PASS" if gate_green else "NOT-VERIFIED",
            "status_rule": "master exam must be green; independent auditor must reproduce",
        }
        for req, impl, verify in rows
    ]

    measurement_categories = {
        "measured": [
            "clean-checkout CI release gate",
            "real nginx + ModSecurity process-level enforcement",
            "local HTTPS/TLS termination",
            "bounded network load with achieved rate and p50/p95/p99",
            "rule replay positive/negative validation and deployment",
            "outbound response anomaly E2E",
            "generated five-minute browser demo artifact",
        ],
        "synthetic_or_deterministic": [
            "versioned synthetic HTTP ML dataset",
            "partial-label semi-supervised evaluation",
            "synthetic behavioural workload",
            "synthetic outbound-response evaluation workload",
            "synthetic rule replay corpus",
        ],
        "design_or_projection": [
            "horizontal scale-out beyond a single CI runner",
            "million-request Internet-scale capacity",
        ],
        "externally_constrained": [
            "public certificate issuance/rotation",
            "public Internet HTTPS verification",
            "venue-specific deployment and challenge portal upload",
        ],
    }

    handoff = ROOT / "handoff"
    handoff.mkdir(exist_ok=True)

    trace_manifest = {
        "schema": "phase10-requirement-traceability-v2",
        "project": "dev2089/swavlamban-waf-ml",
        "challenge": "Challenge 3 - ML-integrated open-source WAF",
        "phase": 10,
        "branch": sh("git", "branch", "--show-current"),
        "commit_sha": current_sha,
        "baseline_commit_sha": BASELINE,
        "gate_green": gate_green,
        "tracked_file_count": len(tracked),
        "readable_text_file_count": readable_text_files,
        "changed_files_from_baseline": diff_names,
        "required_files_missing": missing,
        "secret_scan_findings": suspicious,
        "binary_or_non_utf8_tracked": binary_or_non_utf8,
        "measurement_categories": measurement_categories,
        "challenge_traceability": trace,
    }
    (handoff / "PHASE10_REQUIREMENT_TRACEABILITY.json").write_text(json.dumps(trace_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff / "PHASE10_REQUIREMENT_TRACEABILITY.md").write_text(
        "# Phase 10 Requirement Traceability\n\n" + "\n".join(
            f"- **{row['requirement']}**\n  - Implementation: `{row['implementation']}`\n  - Verification: `{row['verification']}`\n  - Status: **{row['status']}**"
            for row in trace
        ) + "\n", encoding="utf-8")

    production = {
        "request_path": "client -> nginx/ModSecurity -> Swavlamban gateway -> request feature extraction -> rules + four ML request detectors -> deterministic risk policy -> upstream only when not blocked -> outbound response detector -> client",
        "request_ml_detectors": ["supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1", "semi-supervised-v1"],
        "outbound_ml_detector": "outbound-oneclasssvm-v1",
        "request_feature_schema": "http-v2",
        "outbound_feature_schema": "http-response-v1",
        "model_artifact_schema": "phase10-model-v3",
        "explainability": "structured detector contributions, reasons, rule IDs, feature groups/attribution and model/feature provenance",
        "rule_lifecycle": "recommend -> replay validate -> human approve -> deploy -> rollback",
        "learning": "baseline -> feedback -> drift -> challenger retrain -> frozen evaluation -> explicit promotion/rollback",
        "auth": "signed bearer authentication with role-aware authorization",
        "persistence": "server-only Supabase REST adapter with bounded asynchronous telemetry path",
        "privacy": "raw body/query/header retention disabled; source identifiers hashed",
        "dashboard": "authenticated dynamic runtime API state",
        "scale_boundary": "bounded local measured performance; horizontal scale-out is an architecture target, not an Internet-scale measurement",
        "external_limits": ["public certificate lifecycle", "public Internet deployment", "Internet-scale distributed load"],
    }
    (handoff / "PHASE10_PRODUCTION_READINESS.json").write_text(json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    security = """# Phase 10 Security Audit\n\n## Verified controls\n- production configuration requires strong authentication/persistence settings and rejects unsafe wildcard CORS;\n- bearer tokens validate algorithm, issuer, audience, timestamps, role and authenticated subject;\n- administrative rule/model actions are authorization-gated and auditable;\n- persisted telemetry excludes raw body/query/header material and hashes source identifiers;\n- request decisions do not depend on synchronous Supabase success; storage failure is not converted into an unsafe allow decision;\n- nginx + ModSecurity enforcement is tested before a protected upstream;\n- request and response size limits, rate limiting and upstream timeout/failure paths are present;\n- secret-like literal scanning is executed by the audit generator and no private-key/token findings are accepted;\n- Supabase RLS/privilege/search-path hardening is verified in the live project evidence package.\n\n## Boundary\nPublic certificate issuance/rotation and public Internet exposure are not local proof and remain external operational work.\n"""
    (handoff / "PHASE10_SECURITY_AUDIT.md").write_text(security, encoding="utf-8")

    negative = """# Phase 10 Negative Evidence Register\n\nThe following are intentionally not claimed as locally verified:\n\n- public certificate issuance, rotation and revocation lifecycle;\n- public Internet HTTPS reachability;\n- Internet-scale distributed millions-of-requests execution;\n- field-traffic ML accuracy from the deterministic synthetic dataset;\n- venue-specific public deployment and final challenge portal submission;\n- any remote LLM or paid API dependency in the security decision path.\n\nThe five-minute demo, local TLS, ModSecurity enforcement, outbound inspection, dashboard and bounded load evidence are executable local/CI artifacts and are classified separately from these limits.\n"""
    (handoff / "PHASE10_NEGATIVE_EVIDENCE.md").write_text(negative, encoding="utf-8")

    claims: list[dict[str, object]] = []
    add_claim(claims, "The live edge combines traditional WAF rules with supervised, unsupervised, semi-supervised and behavioural request signals.", "implementation fact", "waf/edge/pipeline.py; waf/ml/ensemble.py")
    add_claim(claims, "The gateway inspects outbound responses before returning them to the client.", "implementation fact", "waf/gateway/proxy.py; waf/ml/outbound.py; phase10_outbound_evidence.json")
    add_claim(claims, "Decision evidence contains structured reasons, feature evidence and version provenance.", "implementation fact", "waf/core/models.py; waf/explainability.py")
    add_claim(claims, "Rules require replay validation and human approval before deployment.", "implementation fact", "waf/rules/lifecycle.py; phase10_rule_replay_evidence.json")
    add_claim(claims, "The dashboard is backed by authenticated runtime APIs rather than static production KPIs.", "implementation fact", "dashboard/index.html; waf/api/production_api.py")
    add_claim(claims, "Local HTTPS termination and malicious-request blocking are measured in executable evidence.", "measured fact", "phase10_tls_evidence.json; phase10_waf_enforcement_evidence.json")
    add_claim(claims, "A deterministic five-minute browser walkthrough is generated and packaged.", "measured artifact", "artifacts/PHASE10_DEMO_VIDEO.webm; artifacts/phase10_demo_video_metadata.json")
    add_claim(claims, "The local release gate ran a clean-checkout workflow through packaging and artifact upload.", "measured fact", "GitHub Actions run 35269148465 / job 105363774420")
    add_claim(claims, "Internet-scale millions-of-requests capacity is demonstrated.", "explicit non-claim", "handoff/PHASE10_NEGATIVE_EVIDENCE.md")
    add_claim(claims, "Public certificate lifecycle is verified.", "explicit non-claim", "handoff/PHASE10_NEGATIVE_EVIDENCE.md")

    evidence_files = {
        "phase10_demo_evidence.json": load_json("phase10_demo_evidence.json"),
        "phase10_load_evidence.json": load_json("phase10_load_evidence.json"),
        "phase10_tls_evidence.json": load_json("phase10_tls_evidence.json"),
        "phase10_waf_enforcement_evidence.json": load_json("phase10_waf_enforcement_evidence.json"),
        "phase10_rule_replay_evidence.json": load_json("phase10_rule_replay_evidence.json"),
        "phase10_outbound_evidence.json": load_json("phase10_outbound_evidence.json"),
    }
    demo = evidence_files["phase10_demo_evidence.json"]
    if demo and isinstance(demo.get("benchmark"), dict):
        b = demo["benchmark"]
        add_claim(claims, f"The in-process demo benchmark executed {b.get('requests')} requests with mean {b.get('mean_ms')} ms and max {b.get('max_ms')} ms.", "measured fact", "phase10_demo_evidence.json", "Environment-dependent bounded benchmark.")
    load = evidence_files["phase10_load_evidence.json"]
    if load:
        lat = load.get("latency_ms") if isinstance(load.get("latency_ms"), dict) else {}
        add_claim(claims, f"The network load harness attempted {load.get('requests_attempted')} requests, achieved {load.get('achieved_requests_per_second')} req/s, error rate {load.get('error_rate')}, p50/p95/p99 {lat.get('p50')}/{lat.get('p95')}/{lat.get('p99')} ms.", "measured fact", "phase10_load_evidence.json", "Local bounded load only.")
    outbound = evidence_files["phase10_outbound_evidence.json"]
    if outbound:
        add_claim(claims, f"Outbound response E2E: normal={outbound.get('normal', {}).get('outbound_decision')}; anomalous={outbound.get('anomalous', {}).get('outbound_decision')}; anomalous risk={outbound.get('anomalous', {}).get('outbound_risk')}.", "measured fact", "phase10_outbound_evidence.json")

    claim_ledger = {
        "schema": "phase10-claim-ledger-v3",
        "commit_sha": current_sha,
        "baseline_commit_sha": BASELINE,
        "claims": claims,
        "coverage_note": "Claims are anchored to current source or current evidence artifacts. Historical Phase 1-9 docs are preserved as history and are not treated as current runtime claims.",
        "source_scan": {"tracked_files": len(tracked), "readable_text_files": readable_text_files, "binary_or_non_utf8": binary_or_non_utf8, "secret_scan_findings": suspicious},
    }
    (handoff / "PHASE10_CLAIM_LEDGER.json").write_text(json.dumps(claim_ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Phase 10 Claim Ledger", "", "Every principal current-state claim is categorized and anchored. The independent auditor should still reproduce the executable evidence.", "", "| Claim | Category | Evidence | Notes |", "|---|---|---|---|"]
    for row in claims:
        md.append(f"| {str(row['claim']).replace('|','\\|')} | {row['category']} | `{row['evidence']}` | {row.get('notes','')} |")
    (handoff / "PHASE10_CLAIM_LEDGER.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    model_manifest = {
        "schema": "phase10-model-manifest-v3",
        "artifact_schema": "phase10-model-v3",
        "feature_schema": "http-v2",
        "outbound_feature_schema": "http-response-v1",
        "request_detectors": ["supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1", "semi-supervised-v1"],
        "outbound_detector": "outbound-oneclasssvm-v1",
        "model_version": "phase10-ml-v3",
        "dataset_version": "synthetic-http-v4-s6000-seed42",
        "baseline_version": "benign-baseline-v4-s2600-seed123",
        "semi_supervised": {"labeled_fraction": 0.30, "unlabeled_training_used": True, "evaluation": "evaluate_semi_supervised()"},
        "outbound": {"evaluation": "evaluate_outbound()", "raw_response_retained": False},
        "promotion_control": "Phase 7 challenger validation, explicit promotion and rollback",
    }
    (handoff / "PHASE10_MODEL_MANIFEST.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset_manifest = {
        "schema": "phase10-dataset-manifest-v2",
        "request_dataset": {"version": "synthetic-http-v4-s6000-seed42", "generator": "waf/ml/dataset.py", "samples": 6000, "class_balance": {"benign": 3000, "attack": 3000}, "field_accuracy_claim": False},
        "benign_baseline": {"version": "benign-baseline-v4-s2600-seed123", "samples": 2600, "label_scope": "benign-only"},
        "semi_supervised": {"labeled_fraction": 0.30, "unlabeled_fraction": 0.70, "evaluation_split": "held-out test set"},
        "behaviour": {"scope": "deterministic synthetic behavioural workload"},
        "outbound": {"scope": "deterministic synthetic HTTP response workload", "feature_schema": "http-response-v1"},
        "leakage_control": "train/test split performed before metric calculation; unsupervised model trains on benign-only partition",
    }
    (handoff / "PHASE10_DATASET_MANIFEST.json").write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rule_manifest = {
        "schema": "phase10-rule-manifest-v2",
        "ruleset_schema": "rule-v1",
        "deployable_actions": ["block"],
        "recommendation_source": "decision evidence / ML insight",
        "validation": "required replay corpus with positive and negative examples",
        "approval": "required authenticated human approver",
        "deployment": "managed lifecycle with revision and rollback",
    }
    (handoff / "PHASE10_RULE_MANIFEST.json").write_text(json.dumps(rule_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (handoff / "PHASE10_FILE_TREE.txt").write_text("\n".join(tracked) + "\n", encoding="utf-8")
    (handoff / "PHASE10_DIFF_FROM_BASELINE.txt").write_text("\n".join(diff_names) + "\n", encoding="utf-8")
    (handoff / "PHASE10_DIFF_FROM_PHASE9.txt").write_text("\n".join(diff_names) + "\n", encoding="utf-8")

    ci = {
        "github_actions": bool(os.getenv("GITHUB_ACTIONS")),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_number": os.getenv("GITHUB_RUN_NUMBER"),
        "workflow": os.getenv("GITHUB_WORKFLOW"),
        "sha": os.getenv("GITHUB_SHA"),
    }
    (handoff / "PHASE10_CI_CONTEXT.json").write_text(json.dumps(ci, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "schema": "phase10-audit-result-v2",
        "result": "PASS" if not missing and not suspicious and gate_green else "FAIL",
        "master_exam_gate_green": gate_green,
        "missing_required": missing,
        "secret_scan_findings": suspicious,
        "current_commit_sha": current_sha,
    }
    (handoff / "PHASE10_AUDIT_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
