"""Evidence-first Phase 10 audit and handoff material generator."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "b94e6aada97c009861e42f1fd6cd705f232a65bb"
SELF = Path(__file__).name


def sh(*args: str) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def files() -> list[str]:
    return [x for x in sh("git", "ls-files").splitlines() if x]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    tracked = files()
    current_sha = sh("git", "rev-parse", "HEAD")
    diff_names = subprocess.check_output(["git", "diff", "--name-status", BASELINE, "HEAD"], cwd=ROOT, text=True).splitlines()

    suspicious = []
    binary_or_sensitive = []
    for rel in tracked:
        path = ROOT / rel
        if path.name == SELF and rel.startswith("scripts/"):
            continue
        if any(rel.lower().endswith(s) for s in (".pem", ".key", ".p12", ".pfx")):
            binary_or_sensitive.append(rel)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            binary_or_sensitive.append(rel)
            continue
        if "BEGIN PRIVATE KEY" in text or "BEGIN RSA PRIVATE KEY" in text:
            suspicious.append(rel + ": private-key material")
        if "ghp_" in text or "github_pat_" in text or "sk_live_" in text:
            suspicious.append(rel + ": token-like literal")

    required = [
        "waf/gateway/proxy.py", "deploy/nginx/phase10-modsecurity.conf", "scripts/phase10_waf_enforcement_e2e.py",
        "scripts/phase10_tls_e2e.py", "scripts/phase10_rule_replay.py", "scripts/phase10_load_harness.py",
        "scripts/phase10_demo.py", "scripts/phase10_master_exam.py", "waf/storage/async_telemetry.py",
        "dashboard/index.html", "docs/PHASE10_TECHNICAL_REPORT.md", "docs/PHASE10_PRESENTATION.md",
        ".github/workflows/phase10.yml",
    ]
    missing = [x for x in required if not (ROOT / x).exists()]

    trace = []
    rows = [
        ("real-time HTTP request inspection", "waf/edge/pipeline.py; waf/gateway/proxy.py", "tests/test_phase10_gateway.py; scripts/phase10_waf_enforcement_e2e.py"),
        ("open-source WAF integration", "deploy/nginx/phase10-modsecurity.conf", "scripts/phase10_waf_enforcement_e2e.py"),
        ("benign baseline handling", "waf/ml/behaviour.py; waf/ml/ensemble.py", "scripts/phase10_demo.py"),
        ("encrypted traffic via TLS termination", "scripts/phase10_tls_e2e.py", "scripts/phase10_tls_e2e.py"),
        ("zero-day resilience mechanism", "waf/ml/ensemble.py; waf/ml/anomaly.py", "phase10_demo_evidence.json; replay evidence"),
        ("API/bot behavioural anomaly detection", "waf/ml/behaviour.py; waf/gateway/proxy.py", "scripts/phase10_demo.py; tests/test_phase10_gateway.py"),
        ("explainable ML decisions", "waf/explainability.py; waf/core/models.py", "tests/test_phase5_decision_evidence.py"),
        ("ML-derived rule recommendation", "waf/rules/lifecycle.py; waf/edge/pipeline.py", "tests/test_phase10_release.py; scripts/phase10_rule_replay.py"),
        ("human approval and deployment lifecycle", "waf/rules/lifecycle.py; waf/api/production_api.py", "tests/test_phase10_release.py"),
        ("continuous learning and feedback", "waf/ml/learning_control.py", "tests/test_phase7_learning_control.py"),
        ("safe model promotion/rollback", "waf/ml/learning_control.py", "tests/test_phase7_learning_control.py"),
        ("sanitized asynchronous telemetry", "waf/storage/async_telemetry.py; waf/storage/production.py", "tests/test_phase10_async_telemetry.py"),
        ("authenticated administrator dashboard", "dashboard/index.html; waf/api/production_api.py", "tests/test_phase10_release.py; phase10_dashboard_demo evidence"),
        ("performance/load evidence", "scripts/phase10_load_harness.py", "phase10_load_evidence.json"),
        ("security, least privilege and RLS", "waf/security/*; supabase/migrations/*phase*", "security tests + live Supabase verification"),
        ("reproducible build/test package", ".github/workflows/phase10.yml; scripts/phase10_master_exam.py", "clean-checkout GitHub Actions run"),
    ]
    for requirement, implementation, verification in rows:
        trace.append({"requirement": requirement, "implementation": implementation, "verification": verification, "status": "PASS_ONLY_IF_EXECUTABLE_GATE_GREEN"})

    handoff = ROOT / "handoff"
    handoff.mkdir(exist_ok=True)
    manifest = {
        "project": "swavlamban-waf-ml", "challenge": "Challenge 3 - ML-integrated open-source WAF", "phase": 10,
        "branch": sh("git", "branch", "--show-current"), "commit_sha": current_sha, "baseline_commit": BASELINE,
        "python": sys.version, "platform": platform.platform(), "tracked_file_count": len(tracked),
        "changed_files_from_baseline": diff_names, "required_files_missing": missing,
        "secret_scan_findings": suspicious, "binary_or_non_utf8_tracked": binary_or_sensitive,
        "measurement_categories": {
            "measured": ["clean-checkout regression", "process-level WAF enforcement", "local HTTPS termination", "bounded network load p50/p95/p99", "replay validation"],
            "simulated": ["versioned synthetic training/evaluation data", "synthetic replay corpus"],
            "design_or_projection": ["horizontal scale-out and million-request architecture beyond local test capacity"],
            "external_constraints": ["public certificate issuance/rotation", "Internet-scale distributed load in free CI"],
        },
        "challenge_traceability": trace,
    }
    (handoff / "PHASE10_REQUIREMENT_TRACEABILITY.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff / "PHASE10_REQUIREMENT_TRACEABILITY.md").write_text(
        "# Phase 10 Requirement Traceability\n\n" + "\n".join(
            f"- **{row['requirement']}**\n  - Implementation: `{row['implementation']}`\n  - Verification: `{row['verification']}`\n  - Status rule: executable gate required"
            for row in trace
        ) + "\n", encoding="utf-8"
    )

    production = {
        "authentication": "signed bearer HS256 token with issuer/audience/expiry/role checks",
        "authorization": "viewer/operator/reviewer/admin permission map; authenticated-subject approval identity binding",
        "privacy": "raw body/query/headers excluded from persisted telemetry; source identity hashed",
        "persistence": "server-only Supabase REST adapter behind bounded asynchronous dispatcher",
        "waf_enforcement": "nginx + ModSecurity before the Swavlamban gateway; blocked traffic is not forwarded upstream",
        "dashboard": "authenticated dynamic state sourced from runtime API, with rule/model/telemetry views",
        "learning": "baseline/feedback/drift/challenger/promotion controls from Phase 7",
        "scalability_boundary": "bounded configurable load harness with p50/p95/p99 and explicit error rate; no fabricated Internet-scale result",
        "critical_risks": ["public certificate management is outside local proof", "Internet-scale load is architectural projection"],
    }
    (handoff / "PHASE10_PRODUCTION_READINESS.json").write_text(json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (handoff / "PHASE10_SECURITY_AUDIT.md").write_text("""# Phase 10 Security Audit\n\n- Production configuration rejects weak/missing secrets, non-HTTPS Supabase URLs, wildcard CORS and missing persistence.\n- Bearer authentication validates HS256, issuer, audience, timestamps, role and subject.\n- Rule approval is bound to the authenticated subject.\n- Runtime telemetry is bounded/asynchronous and storage failures are surfaced without silently changing an enforcement decision to allow.\n- Raw body/query/headers are excluded from persisted telemetry and source identity is SHA-256 hashed.\n- Supabase RLS and privilege hardening are tracked and live-verified.\n- This scanner intentionally skips its own detection literals to avoid false positives while scanning all other tracked files.\n""", encoding="utf-8")
    (handoff / "PHASE10_CLAIM_LEDGER.md").write_text("""# Phase 10 Claim Ledger\n\n| Claim | Category | Evidence anchor |\n|---|---|---|\n| WAF combines signatures and ML/behaviour signals | implementation fact | `waf/edge/pipeline.py` |\n| Decisions contain structured explainability | implementation fact | `waf/explainability.py` |\n| Managed rules require validation and human approval | implementation fact | `waf/rules/lifecycle.py` and release tests |\n| Dashboard state is API-driven | implementation fact | dashboard + production API |\n| Local HTTPS termination works through nginx | measured fact | `phase10_tls_evidence.json` |\n| ModSecurity blocks the malicious request before protected upstream | measured fact | `phase10_waf_enforcement_evidence.json` |\n| Million-request scale is demonstrated | not claimed; architecture target only | negative-evidence register |\n""", encoding="utf-8")
    (handoff / "PHASE10_NEGATIVE_EVIDENCE.md").write_text("""# Phase 10 Negative Evidence Register\n\n- Public certificate issuance/rotation is not verified.\n- Internet-scale distributed load is not physically executed in the free environment.\n- Synthetic ML metrics are not field-traffic accuracy.\n- No remote LLM or paid API is required for the security decision path.\n- No real secret values are included in code or handoff.\n- No automatic model promotion replaces the active model without a validation gate.\n- Vercel is not a required dependency for the WAF runtime path.\n""", encoding="utf-8")
    (handoff / "PHASE10_FILE_TREE.txt").write_text("\n".join(tracked) + "\n", encoding="utf-8")
    (handoff / "PHASE10_DIFF_FROM_PHASE9.txt").write_text("\n".join(diff_names) + "\n", encoding="utf-8")
    (handoff / "PHASE10_DATASET_MANIFEST.json").write_text(json.dumps({"training_source": "versioned deterministic synthetic HTTP dataset", "generator": "waf/ml/dataset.py", "sample_sizes": [6000], "split_method": "stratified supervised test split; benign-only unsupervised training split", "leakage_control": "evaluation partitions remain separate"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff / "PHASE10_MODEL_MANIFEST.json").write_text(json.dumps({"artifact_schema": "phase4-model-v1", "feature_schema": "http-v2", "runtime_versions": ["phase4-ml-v1"], "detectors": ["supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1"], "promotion_control": "Phase 7 challenger validation and explicit promotion/rollback"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff / "PHASE10_RULE_MANIFEST.json").write_text(json.dumps({"schema": "rule-v1", "deployable_matcher": "feature_threshold", "safe_action_set": ["block"], "approval_required": True, "replay_validation_required": True, "source": "phase5-decision-evidence"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {"missing_required": missing, "secret_scan_findings": suspicious, "result": "PASS" if not missing and not suspicious else "FAIL"}
    (handoff / "PHASE10_AUDIT_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
