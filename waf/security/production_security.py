"""Dependency-light production security primitives.

This module is deliberately independent from the FastAPI app so the controls can
be tested without network credentials and reused by adapters at the API edge.
It implements signed bearer tokens, RBAC, fail-closed production configuration,
privacy-safe audit records, and standard response security headers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping


class AuthError(ValueError):
    """Authentication or authorization failure."""


ROLES = ("viewer", "operator", "reviewer", "admin")
PERMISSIONS = {
    "viewer": frozenset({"read:threats", "read:rules", "read:stats"}),
    "operator": frozenset({"read:threats", "read:rules", "read:stats", "analyze:requests"}),
    "reviewer": frozenset({"read:threats", "read:rules", "read:stats", "analyze:requests", "review:feedback", "approve:rules", "approve:models"}),
    "admin": frozenset({
        "read:threats", "read:rules", "read:stats", "analyze:requests",
        "review:feedback", "approve:rules", "approve:models", "manage:roles",
        "manage:secrets", "manage:deployments",
    }),
}


def _b64u(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64u_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class SecurityConfig:
    environment: str
    auth_secret: str
    supabase_url: str
    supabase_service_role_key: str
    allowed_origins: tuple[str, ...]
    issuer: str = "swavlamban-waf"
    audience: str = "waf-control-plane"
    token_ttl_seconds: int = 900
    clock_skew_seconds: int = 30

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SecurityConfig":
        e = os.environ if env is None else env
        origins = tuple(x.strip() for x in e.get("WAF_CORS_ORIGINS", "").split(",") if x.strip())
        return cls(
            environment=e.get("WAF_ENV", "development").strip().lower(),
            auth_secret=e.get("WAF_AUTH_SECRET", ""),
            supabase_url=e.get("SUPABASE_URL", ""),
            supabase_service_role_key=e.get("SUPABASE_SERVICE_ROLE_KEY", ""),
            allowed_origins=origins,
            issuer=e.get("WAF_TOKEN_ISSUER", "swavlamban-waf"),
            audience=e.get("WAF_TOKEN_AUDIENCE", "waf-control-plane"),
            token_ttl_seconds=int(e.get("WAF_TOKEN_TTL_SECONDS", "900")),
            clock_skew_seconds=int(e.get("WAF_CLOCK_SKEW_SECONDS", "30")),
        )


def validate_production_config(config: SecurityConfig) -> list[str]:
    """Return actionable configuration findings. Empty means ready for production startup."""
    findings: list[str] = []
    if config.environment in {"production", "prod"}:
        if len(config.auth_secret.encode("utf-8")) < 32:
            findings.append("WAF_AUTH_SECRET must be at least 32 bytes in production")
        lowered = config.auth_secret.lower()
        if config.auth_secret and any(x in lowered for x in ("changeme", "secret", "password", "example", "dev")):
            findings.append("WAF_AUTH_SECRET appears to use a development/example value")
        if not config.supabase_url.startswith("https://"):
            findings.append("SUPABASE_URL must use https:// in production")
        if not config.supabase_service_role_key:
            findings.append("SUPABASE_SERVICE_ROLE_KEY is required for server-side database access")
        if not config.allowed_origins or "*" in config.allowed_origins:
            findings.append("WAF_CORS_ORIGINS must explicitly allow trusted origins; wildcard is forbidden")
        if config.token_ttl_seconds > 3600 or config.token_ttl_seconds < 60:
            findings.append("WAF_TOKEN_TTL_SECONDS must be between 60 and 3600 seconds")
    return findings


def _sign(header: dict[str, Any], payload: dict[str, Any], secret: str) -> str:
    protected = _b64u(_canonical_json(header))
    body = _b64u(_canonical_json(payload))
    message = f"{protected}.{body}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return f"{protected}.{body}.{_b64u(sig)}"


def issue_token(
    *,
    subject: str,
    role: str,
    secret: str,
    issuer: str = "swavlamban-waf",
    audience: str = "waf-control-plane",
    ttl_seconds: int = 900,
    now: int | None = None,
) -> str:
    if role not in ROLES:
        raise AuthError("invalid role")
    if not subject or len(secret.encode("utf-8")) < 32:
        raise AuthError("strong signing secret and subject are required")
    if not 60 <= ttl_seconds <= 3600:
        raise AuthError("invalid token ttl")
    issued = int(time.time() if now is None else now)
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "role": role,
        "iss": issuer,
        "aud": audience,
        "iat": issued,
        "exp": issued + ttl_seconds,
        "jti": secrets.token_hex(12),
    }
    return _sign(header, payload, secret)


def parse_bearer_token(
    authorization: str,
    *,
    secret: str,
    issuer: str = "swavlamban-waf",
    audience: str = "waf-control-plane",
    now: int | None = None,
    clock_skew_seconds: int = 30,
) -> dict[str, Any]:
    if not authorization.startswith("Bearer "):
        raise AuthError("missing bearer token")
    token = authorization[7:].strip()
    parts = token.split(".")
    if len(parts) != 3 or len(secret.encode("utf-8")) < 32:
        raise AuthError("invalid bearer token")
    try:
        header = json.loads(_b64u_decode(parts[0]))
        payload = json.loads(_b64u_decode(parts[1]))
        signature = _b64u_decode(parts[2])
    except Exception as exc:
        raise AuthError("invalid bearer token") from exc
    if header != {"alg": "HS256", "typ": "JWT"}:
        raise AuthError("unsupported token header")
    expected = hmac.new(
        secret.encode("utf-8"), f"{parts[0]}.{parts[1]}".encode("ascii"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(signature, expected):
        raise AuthError("invalid token signature")
    current = int(time.time() if now is None else now)
    if payload.get("iss") != issuer or payload.get("aud") != audience:
        raise AuthError("invalid token issuer or audience")
    if payload.get("role") not in ROLES or not payload.get("sub"):
        raise AuthError("invalid token claims")
    try:
        iat = int(payload["iat"])
        exp = int(payload["exp"])
    except Exception as exc:
        raise AuthError("invalid token timestamps") from exc
    if iat > current + clock_skew_seconds or exp < current - clock_skew_seconds or exp <= iat:
        raise AuthError("expired or not-yet-valid token")
    return payload


def authorize(claims: Mapping[str, Any], permission: str) -> None:
    role = str(claims.get("role", ""))
    if permission not in PERMISSIONS.get(role, frozenset()):
        raise AuthError("forbidden")


def security_headers() -> dict[str, str]:
    """Headers safe to add to all HTTP responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        "Cache-Control": "no-store",
    }


def audit_record(*, actor: str, action: str, target: str, outcome: str, request_id: str) -> dict[str, str]:
    """Build a privacy-safe audit event with identifiers only, never request bodies/headers."""
    if not all(isinstance(x, str) and x for x in (actor, action, target, outcome, request_id)):
        raise ValueError("audit fields must be non-empty strings")
    return {
        "actor": actor[:128],
        "action": action[:128],
        "target": target[:256],
        "outcome": outcome[:32],
        "request_id": request_id[:128],
        "recorded_at": str(int(time.time())),
    }
