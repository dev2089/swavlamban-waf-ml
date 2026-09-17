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
    return TestClient(create_app(env=dict(os.environ)) , follow_redirects=True)


def token(role="operator"):
    from waf.security.production_security import issue_token
    return issue_token(subject=f"phase10-{role}", role=role, secret=SECRET)


def test_dashboard_is_served_and_protected_api_remains(monkeypatch):
    client = make_client(monkeypatch)
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Swavlamban WAF Control Plane" in dashboard.text
    assert client.get("/api/stats").status_code == 401


def test_dashboard_operator_flow(monkeypatch):
    client = make_client(monkeypatch)
    headers = {"Authorization": "Bearer " + token()}
    health = client.get("/api/health")
    assert health.status_code == 200
    result = client.post("/api/analyze", headers=headers, json={"method":"GET","uri":"/search","body":"<script>alert(1)</script>"})
    assert result.status_code == 200
    assert "raw_payload_retained" in result.json()["evidence"]["privacy"]


def test_release_materials_exist():
    for relative in [
        "dashboard/index.html",
        "docs/PHASE10_TECHNICAL_REPORT.md",
        "docs/PHASE10_PRESENTATION.md",
        "scripts/phase10_demo.py",
        "scripts/phase10_master_exam.py",
    ]:
        assert (ROOT / relative).exists(), relative
