from waf.security.production_security import (
    AuthError,
    PERMISSIONS,
    SecurityConfig,
    audit_record,
    authorize,
    issue_token,
    parse_bearer_token,
    security_headers,
    validate_production_config,
)

SECRET = "p8-" + "x" * 61


def test_issue_and_parse_token_round_trip():
    token = issue_token(subject="alice", role="reviewer", secret=SECRET, now=1_700_000_000)
    claims = parse_bearer_token(
        "Bearer " + token,
        secret=SECRET,
        now=1_700_000_100,
    )
    assert claims["sub"] == "alice"
    assert claims["role"] == "reviewer"
    assert claims["aud"] == "waf-control-plane"


def test_tamper_expiry_and_bad_algorithm_fail_closed():
    token = issue_token(subject="alice", role="reviewer", secret=SECRET, now=1_700_000_000)
    header, body, sig = token.split(".")
    bad = f"{header}.{body[:-1]}x.{sig}"
    try:
        parse_bearer_token("Bearer " + bad, secret=SECRET, now=1_700_000_100)
        assert False
    except AuthError:
        pass
    try:
        parse_bearer_token("Bearer " + token, secret=SECRET, now=1_700_001_000)
        assert False
    except AuthError:
        pass


def test_rbac_is_explicit_and_no_role_gets_implicit_write_access():
    assert "manage:roles" in PERMISSIONS["admin"]
    assert "manage:roles" not in PERMISSIONS["reviewer"]
    assert "approve:models" in PERMISSIONS["reviewer"]
    authorize({"role": "reviewer"}, "approve:models")
    try:
        authorize({"role": "viewer"}, "approve:models")
        assert False
    except AuthError:
        pass


def test_production_config_is_fail_closed():
    config = SecurityConfig(
        environment="production",
        auth_secret="short",
        supabase_url="http://localhost",
        supabase_service_role_key="",
        allowed_origins=("*",),
    )
    findings = validate_production_config(config)
    assert len(findings) >= 4
    assert any("32 bytes" in item for item in findings)
    assert any("https://" in item for item in findings)
    assert any("service_role" in item for item in findings)
    assert any("wildcard" in item for item in findings)


def test_security_headers_and_audit_are_privacy_safe():
    headers = security_headers()
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-store"
    assert "request_body" not in audit_record(actor="alice", action="review", target="f-1", outcome="ok", request_id="r-1")
    assert "headers" not in audit_record(actor="alice", action="review", target="f-1", outcome="ok", request_id="r-1")


def test_missing_bearer_is_rejected():
    try:
        parse_bearer_token("", secret=SECRET)
        assert False
    except AuthError:
        pass
