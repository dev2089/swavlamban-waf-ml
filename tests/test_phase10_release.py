import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SECRET = "phase10-test-secret-" + "x" * 32


def make_client(monkeypatch):
    monkeypatch.setenv("WAF_ENV", "development")
    monkeypatch.setenv("WAF_AUTH_SECRET", SECRET)
    monkeypatch.setenv("WAF_CORS_ORIGINS", "https://console.example.test")
    from waf.api.production_api import create_app
    return TestClient(create_app(env=dict(os.environ)), follow_redirects=True)


def token(role="operator", subject=None):
    from waf.security.production_security import issue_token
    return issue_token(subject=subject or f"phase10-{role}", role=role, secret=SECRET)


def test_dashboard_is_served_and_protected_api_remains(monkeypatch):
    client = make_client(monkeypatch)
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Swavlamban WAF Control Plane" in dashboard.text
    assert client.get("/api/stats").status_code == 401


def test_dashboard_operator_flow_and_dynamic_stats(monkeypatch):
    client = make_client(monkeypatch)
    headers = {"Authorization": "Bearer " + token()}
    before = client.get("/api/stats", headers=headers).json()["total_requests"]
    result = client.post(
        "/api/analyze",
        headers={**headers, "X-Request-ID": "phase10-rel-0001"},
        json={"method":"GET","uri":"/search","body":"<script>alert(1)</script>"},
    )
    assert result.status_code == 200
    payload = result.json()
    assert "raw_payload_retained" in payload["evidence"]["privacy"]
    after = client.get("/api/stats", headers=headers).json()["total_requests"]
    assert after == before + 1
    telemetry = client.get("/api/telemetry", headers=headers).json()
    assert telemetry["runtime"]["total_requests"] >= 1


def test_rule_recommendation_requires_auth_and_returns_state(monkeypatch):
    client = make_client(monkeypatch)
    assert client.post("/api/rules/recommend", json={"method":"GET","uri":"/search","query":"q=' OR 1=1--"}).status_code == 401
    operator = {"Authorization": "Bearer " + token("operator")}
    recommendation = client.post(
        "/api/rules/recommend",
        headers=operator,
        json={"method":"GET","uri":"/search","query":"q=' OR 1=1--"},
    )
    assert recommendation.status_code == 200
    assert recommendation.json()["decision"] == "block"
    assert isinstance(recommendation.json()["rules"], list)


def test_rule_approval_identity_and_model_metadata(monkeypatch):
    client = make_client(monkeypatch)
    reviewer_subject = "phase10-reviewer"
    reviewer = {"Authorization": "Bearer " + token("reviewer", reviewer_subject)}
    model = client.get("/api/models", headers=reviewer)
    assert model.status_code == 200
    assert model.json()["runtime"]["feature_schema"] == "http-v2"
    rec = client.post(
        "/api/rules/recommend",
        headers=reviewer,
        json={"method":"GET","uri":"/search","query":"q=' OR 1=1--"},
    )
    assert rec.status_code == 200
    rules = rec.json()["rules"]
    if not rules:
        return
    rule_id = rules[0]["rule_id"]
    validation = client.post(f"/api/rules/{rule_id}/validate", headers=reviewer)
    assert validation.status_code == 200
    assert validation.json()["valid"] is True
    assert client.post(f"/api/rules/{rule_id}/approve", headers=reviewer, json={"approver":"wrong-subject"}).status_code == 403
    approved = client.post(f"/api/rules/{rule_id}/approve", headers=reviewer, json={"approver":reviewer_subject})
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"


def test_admin_can_deploy_and_view_rule_revision(monkeypatch):
    client = make_client(monkeypatch)
    admin_subject = "phase10-admin"
    admin = {"Authorization": "Bearer " + token("admin", admin_subject)}
    rec = client.post(
        "/api/rules/recommend",
        headers=admin,
        json={"method":"GET","uri":"/search","query":"q=' OR 1=1--"},
    )
    assert rec.status_code == 200
    for rule in rec.json()["rules"]:
        rule_id = rule["rule_id"]
        assert client.post(f"/api/rules/{rule_id}/validate", headers=admin).status_code == 200
        assert client.post(f"/api/rules/{rule_id}/approve", headers=admin, json={"approver":admin_subject}).status_code == 200
    if rec.json()["rules"]:
        deploy = client.post("/api/rules/deploy", headers=admin)
        assert deploy.status_code == 200
        state = client.get("/api/rules", headers=admin)
        assert state.status_code == 200
        assert state.json()["revision"] >= 1


def test_production_config_remains_fail_closed(monkeypatch):
    from waf.api.production_api import create_app
    bad = {
        "WAF_ENV": "production",
        "WAF_AUTH_SECRET": "short",
        "SUPABASE_URL": "http://db.example.test",
        "SUPABASE_SERVICE_ROLE_KEY": "",
        "WAF_CORS_ORIGINS": "*",
    }
    import pytest
    with pytest.raises(RuntimeError):
        create_app(env=bad)


def test_release_materials_exist():
    for relative in [
        "dashboard/index.html",
        "docs/PHASE10_TECHNICAL_REPORT.md",
        "docs/PHASE10_PRESENTATION.md",
        "scripts/phase10_demo.py",
        "scripts/phase10_master_exam.py",
        "scripts/phase10_waf_enforcement_e2e.py",
        "waf/gateway/proxy.py",
        "waf/storage/async_telemetry.py",
        "deploy/nginx/phase10-modsecurity.conf",
    ]:
        assert (ROOT / relative).exists(), relative
