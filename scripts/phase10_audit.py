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


def load_json(name: str) -> dict[str, object] | None:
    path = ROOT / name
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def add_claim(rows: list[dict[str, object]], claim: str, category: str, evidence: str, notes: str = "") -> None:
    rows.append({"claim": claim, "category": category, "evidence": evidence, "notes": notes})


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

    exam = load_json("phase10_master_exam_result.json") or {}
    gate_green = exam.get("result") == "PASS" and not exam.get("failed_checks") and not exam.get("critical_failures")

    trace = []
    rows = [
        ("real-time HTTP request inspection", "waf/edge/pipeline.py; waf/gateway/proxy.py", "tests/test_phase10_gateway.py; scripts/phase10_waf_enforcement_e2e.py"),
        ("open-source WAF integration", "deploy/nginx/phase10-modsecurity.conf", "scripts/phase10_waf_enforcement_e2e.py"),
        ("benign baseline handling", "waf/ml/behaviour.py; waf/ml/ensemble.py", "scripts/phase10_demo.py"),
        ("encrypted traffic via TLS termination", "scripts/phase10_tls_e2e.py", "scripts/phase10_tls_e2e.py"),
        ("zero-day resilience mechanism", "waf/ml/ensemble.py; waf/ml/anomaly.py", "phase10_demo_evidence.json; phase10_rule_replay_evidence.json"),
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
        ("reproducible build/test package", ".github/workflows/phase10.yml; scripts/phase10_master_exam.py", "GitHub Actions clean-checkout release gate"),
    ]
    for requirement, implementation, verification in rows:
        trace.append({
            "requirement": requirement,
            "implementation": implementation,
            "verification": verification,
            "status": "PASS" if gate_green else "NOT-VERIFIED",
            "gate_dependency": "phase10_master_exam_result.json",
        })

    handoff = ROOT / "handoff"
    handoff.mkdir(exist_ok=True)
    manifest = {
        "project": "swavlamban-waf-ml", "challenge": "Challenge 3 - ML-integrated open-source WAF", "phase": 10,
        "branch": sh("git", "branch", "--show-current"), "commit_sha": current_sha, "baseline_commit": BASELINE,
        "python": sys.version, "platform": platform.platform(), "tracked_file_count": len(tracked),
        "changed_files_from_baseline": diff_names, "required_files_missing": missing,
        "secret_scan_findings": suspicious, "binary_or_non_utf8_tracked": binary_or_sensitive,
        "measurement_categories": {
            "measured": ["clean-checkout regression", "process-level WAF enforcement", "local HTTPS termination", "bounded network load p50/p95/p99", "replay validation", "generated demo video"],
            "simulated": ["versioned synthetic training/evaluation data", "synthetic replay corpus"],
            "design_or_projection": ["horizontal scale-out and million-request architecture beyond local test capacity"],
            "external_constraints": ["public certificate issuance/rotation", "Internet-scale distributed load in free CI"],
        },
        "challenge_traceability": trace,
    }
    (handoff / "PHASE10_REQUIREMENT_TRACEABILITY.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff / "PHASE10_REQUIREMENT_TRACEABILITY.md").write_text(
        "# Phase 10 Requirement Traceability\n\n" + "\n".join(
            f"- **{row['requirement']}**\n  - Implementation: `{row['implementation']}`\n  - Verification: `{row['verification']}`\n  - Status: **{row['status']}** (master-exam gated)"
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

    claim_rows: list[dict[str, object]] = []
    add_claim(claim_rows, "The live edge combines traditional signature checks with supervised, unsupervised and behavioural ML signals.", "implementation fact", "waf/edge/pipeline.py; waf/ml/ensemble.py")
    add_claim(claim_rows, "ML decisions include structured explainability evidence and detector reasons.", "implementation fact", "waf/explainability.py; tests/test_phase5_decision_evidence.py")
    add_claim(claim_rows, "ML-derived rules require replay validation and human approval before deployment.", "implementation fact", "waf/rules/lifecycle.py; tests/test_phase10_release.py; phase10_rule_replay_evidence.json")
    add_claim(claim_rows, "The dashboard reads dynamic runtime state through authenticated APIs rather than fixed production KPI values.", "implementation fact", "dashboard/index.html; waf/api/production_api.py; tests/test_phase10_release.py")
    add_claim(claim_rows, "Local HTTPS traffic is terminated at nginx and inspected by the WAF path.", "measured fact", "phase10_tls_evidence.json")
    add_claim(claim_rows, "ModSecurity blocks the malicious request before it reaches the protected upstream marker.", "measured fact", "phase10_waf_enforcement_evidence.json")
    add_claim(claim_rows, "A five-minute browser demo artifact is produced by the deterministic dashboard demo script.", "measured artifact", "artifacts/PHASE10_DEMO_VIDEO.webm; artifacts/phase10_demo_video_metadata.json")
    add_claim(claim_rows, "The project includes an 8-10 slide presentation source and a technical report PDF.", "artifact fact", "artifacts/PHASE10_PRESENTATION.pptx; artifacts/PHASE10_TECHNICAL_REPORT.pdf")
    add_claim(claim_rows, "Million-request Internet-scale processing is not demonstrated by the free/local test environment.", "explicit non-claim", "handoff/PHASE10_NEGATIVE_EVIDENCE.md")
    add_claim(claim_rows, "Public certificate issuance and public Internet HTTPS verification are not established by local CI evidence.", "explicit non-claim", "handoff/PHASE10_NEGATIVE_EVIDENCE.md; phase10_tls_evidence.json")

    demo = load_json("phase10_demo_evidence.json")
    if demo and isinstance(demo.get("benchmark"), dict):
        b = demo["benchmark"]
        add_claim(claim_rows, f"The deterministic in-process demo benchmark executed {b.get('requests')} requests with mean {b.get('mean_ms')} ms and max {b.get('max_ms')} ms.", "measured fact", "phase10_demo_evidence.json", "Environment-dependent and excludes distributed Internet-scale capacity.")

    load = load_json("phase10_load_evidence.json")
    if load:
        lat = load.get("latency_ms") if isinstance(load.get("latency_ms"), dict) else {}
        add_claim(
            claim_rows,
            f"The bounded network load harness attempted {load.get('requests_attempted')} requests at requested {load.get('requested_rate_per_second')} req/s, achieved {load.get('achieved_requests_per_second')} req/s, with error rate {load.get('error_rate')} and p50/p95/p99 {lat.get('p50')}/{lat.get('p95')}/{lat.get('p99')} ms.",
            "measured fact",
            "phase10_load_evidence.json",
            "Local bounded load only.",
        )

    tls = load_json("phase10_tls_evidence.json")
    if tls:
        add_claim(claim_rows, f"TLS termination evidence reports allowed HTTPS status {tls.get('https_allow_status')} and SQL-block status {tls.get('https_sql_block_status')}, with blocked traffic not reaching upstream.", "measured fact", "phase10_tls_evidence.json")

    waf = load_json("phase10_waf_enforcement_evidence.json")
    if waf:
        add_claim(claim_rows, f"Process-level WAF enforcement reports SQL blocked at WAF={waf.get('sql_blocked_at_waf')} and blocked_request_reached_upstream={waf.get('blocked_request_reached_upstream')}.", "measured fact", "phase10_waf_enforcement_evidence.json")

    replay = load_json("phase10_rule_replay_evidence.json")
    if replay:
        validation = replay.get("validation") if isinstance(replay.get("validation"), dict) else {}
        positive = replay.get("positive_example") if isinstance(replay.get("positive_example"), dict) else {}
        negative = replay.get("negative_example") if isinstance(replay.get("negative_example"), dict) else {}
        add_claim(claim_rows, f"Rule replay validation={validation.get('valid')}; positive_match={positive.get('matched')}; negative_match={negative.get('matched')}; deployment_status={((replay.get('deployment') or {}).get('status') if isinstance(replay.get('deployment'), dict) else None)}.", "measured fact", "phase10_rule_replay_evidence.json", "Replay corpus is synthetic and deterministic.")

    if exam:
        add_claim(claim_rows, f"The executable Phase 10 master exam result is {exam.get('result')} with score {exam.get('score')} and failed_checks={exam.get('failed_checks')}, critical_failures={exam.get('critical_failures')}.", "measured fact", "phase10_master_exam_result.json")

    claim_ledger = {
        "schema": "phase10-claim-ledger-v2",
        "commit_sha": current_sha,
        "baseline_commit_sha": BASELINE,
        "source_files_scanned": [
            "README.md", "START_HERE.md", "FEATURES.md", "QUICKSTART.md", "QUICKSTART_NEW.md",
            "TECHNICAL_DOCUMENTATION.md", "WAF_ML_SETUP.md", "docs/ARCHITECTURE.md",
            "docs/PHASE10_TECHNICAL_REPORT.md", "docs/PHASE10_PRESENTATION.md", "dashboard/index.html",
            "handoff/START_HERE.md", "handoff/PHASE10_FINAL_STATUS.md", "handoff/PHASE10_RELEASE_CANDIDATE_INDEX.md",
            "handoff/PHASE10_NEGATIVE_EVIDENCE.md", "handoff/PHASE10_REQUIREMENT_TRACEABILITY.md",
        ],
        "claims": claim_rows,
        "coverage_note": "Quantitative claims are generated from the current evidence JSON where available. Qualitative claims are the principal evaluator-facing implementation statements anchored to source/tests. Historical phase documents remain historical evidence and are not rewritten into current metrics.",
    }
    (handoff / "PHASE10_CLAIM_LEDGER.json").write_text(json.dumps(claim_ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        "# Phase 10 Claim Ledger",
        "",
        "Builder claims are categorized and anchored to source or generated evidence. The independent auditor should still re-run all executable checks.",
        "",
        "| Claim | Category | Evidence | Notes |",
        "|---|---|---|---|",
    ]
    for row in claim_rows:
        claim = str(row["claim"]).replace("|", "\\|")
        md_lines.append(f"| {claim} | {row['category']} | `{row['evidence']}` | {row.get('notes', '')} |")
    (handoff / "PHASE10_CLAIM_LEDGER.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    (handoff / "PHASE10_NEGATIVE_EVIDENCE.md").write_text("""# Phase 10 Negative Evidence Register\n\n- Public certificate issuance/rotation is not verified.\n- Public Internet HTTPS verification is not established; TLS evidence uses local self-signed termination.\n- Internet-scale distributed load is not physically executed in the free environment.\n- Synthetic ML metrics are not field-traffic accuracy.\n- No remote LLM or paid API is required for the security decision path.\n- No real secret values are included in code or handoff.\n- No automatic model promotion replaces the active model without a validation gate.\n- Vercel is not a required dependency for the WAF runtime path.\n""", encoding="utf-8")
    (handoff / "PHASE10_FILE_TREE.txt").write_text("\n".join(tracked) + "\n", encoding="utf-8")
    (handoff / "PHASE10_DIFF_FROM_PHASE9.txt").write_text("\n".join(diff_names) + "\n", encoding="utf-8")
    (handoff / "PHASE10_DATASET_MANIFEST.json").write_text(json.dumps({"training_source": "versioned deterministic synthetic HTTP dataset", "generator": "waf/ml/dataset.py", "sample_sizes": [6000], "split_method": "stratified supervised test split; benign-only unsupervised training split", "leakage_control": "evaluation partitions remain separate", "field_accuracy_claim": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff / "PHASE10_MODEL_MANIFEST.json").write_text(json.dumps({"artifact_schema": "phase4-model-v1", "feature_schema": "http-v2", "runtime_versions": ["phase4-ml-v1"], "detectors": ["supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1"], "promotion_control": "Phase 7 challenger validation and explicit promotion/rollback"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff / "PHASE10_RULE_MANIFEST.json").write_text(json.dumps({"schema": "rule-v1", "deployable_matcher": "feature_threshold", "safe_action_set": ["block"], "approval_required": True, "replay_validation_required": True, "source": "phase5-decision-evidence"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {"missing_required": missing, "secret_scan_findings": suspicious, "result": "PASS" if not missing and not suspicious else "FAIL", "master_exam_gate_green": gate_green}
    (handoff / "PHASE10_AUDIT_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
