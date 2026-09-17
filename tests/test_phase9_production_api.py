import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SECRET = "phase9-test-secret-" + "x" * 32
ENV = {
    "WAF_ENV": "development",
    "WAF_AUTH_SECRET": SECRET,
    "WAF_CORS_ORIGINS": "https://console.example.test",
    "WAF_TOKEN_ISSUER": "swavlamban-waf",
    "WAF_TOKEN_AUDIENCE": "waf-control-plane",
    "WAF_TOKEN_TTL_SECONDS": "900",
    "WAF_CLOCK_SKEW_SECONDS": "30",
}


def make_client(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    from waf.api.production_api import create_app
    return TestClient(create_app(env=dict(os.environ)))


def token(role: str) -> str:
    from waf.security.production_security import issue_token
    return issue_token(subject=f"phase9-{role}", role=role, secret=SECRET)


def test_auth_and_rbac(monkeypatch):
    client = make_client(monkeypatch)
    assert client.get("/api/security/me").status_code == 401
    viewer = {"Authorization": "Bearer " + token("viewer")}
    operator = {"Authorization": "Bearer " + token("operator")}
    reviewer = {"Authorization": "Bearer " + token("reviewer")}
    admin = {"Authorization": "Bearer " + token("admin")}
    assert client.get("/api/stats", headers=viewer).status_code == 200
    assert client.post("/api/analyze", headers=viewer, json={"method":"GET","uri":"/api/items"}).status_code == 403
    assert client.post("/api/analyze", headers=operator, json={"method":"GET","uri":"/api/items"}).status_code == 200
    assert client.post("/api/rules/deploy", headers=reviewer).status_code == 403
    assert client.post("/api/rules/deploy", headers=admin).status_code == 409


def test_analyze_privacy_and_security_headers(monkeypatch):
    client = make_client(monkeypatch)
    headers = {"Authorization": "Bearer " + token("operator"), "Origin": "https://console.example.test", "X-Request-ID": "phase9-test-001"}
    payload = {"method":"GET","uri":"/search?q=select%20from%20users","source_ip":"203.0.113.10","headers":{"Authorization":"SECRET"},"query":"q=select","body":"<script>alert(1)</script>"}
    response = client.post("/api/analyze", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_id"] == "phase9-test-001"
    assert "raw" in json.dumps(body["evidence"]["privacy"]).lower()
    assert "SECRET" not in response.text
    assert "<script>" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_storage_is_sanitized(monkeypatch):
    client = make_client(monkeypatch)
    headers = {"Authorization": "Bearer " + token("operator")}
    response = client.post("/api/analyze", headers=headers, json={"method":"GET","uri":"/api/items","source_ip":"198.51.100.5","headers":{"X-Secret":"do-not-store"},"body":"PRIVATE-PAYLOAD"})
    assert response.status_code == 200
    store = client.app.state.store
    request_log = store.request_logs[-1]
    assert request_log["body"] is None
    assert request_log["headers"] == {}
    assert request_log["source_ip"] != "198.51.100.5"


def test_production_configuration_fails_closed(monkeypatch):
    from waf.api.production_api import create_app
    bad = dict(ENV)
    bad.update({
        "WAF_ENV": "production",
        "WAF_AUTH_SECRET": "short",
        "SUPABASE_URL": "http://db.example.test",
        "SUPABASE_SERVICE_ROLE_KEY": "",
        "WAF_CORS_ORIGINS": "*",
    })
    with pytest.raises(RuntimeError):
        create_app(env=bad)


def test_api_uses_supplied_config_mapping(monkeypatch):
    from waf.api.production_api import create_app
    env = dict(ENV)
    env.update({"WAF_BLOCK_THRESHOLD": "0.61", "WAF_ALERT_THRESHOLD": "0.42"})
    app = create_app(env=env)
    assert app.state.waf.config.block_threshold == 0.61
    assert app.state.waf.config.alert_threshold == 0.42
