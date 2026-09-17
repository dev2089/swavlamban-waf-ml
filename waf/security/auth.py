from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import hmac
import json
import secrets
from typing import Any, Mapping

AUTH_SCHEMA_VERSION = "auth-v1"
_TOKEN_PREFIX = "waf1"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    pad = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + pad).encode("ascii"))


class AuthError(ValueError):
    """Raised for invalid or expired authentication tokens."""


@dataclass(frozen=True, slots=True)
class Actor:
    actor_id: str
    role: str
    issued_at: int
    expires_at: int
    token_id: str
    schema_version: str = AUTH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.actor_id.strip():
            raise AuthError("actor_id is required")
        if not self.role.strip():
            raise AuthError("role is required")
        if self.expires_at <= self.issued_at:
            raise AuthError("token expiry must be after issue time")
        if self.schema_version != AUTH_SCHEMA_VERSION:
            raise AuthError("unsupported auth schema")

    def claims(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "actor_id": self.actor_id, "role": self.role, "iat": self.issued_at, "exp": self.expires_at, "jti": self.token_id}


class TokenCodec:
    """Stdlib-only signed bearer-token codec for the WAF control plane."""
    def __init__(self, secret: bytes | str, clock: callable | None = None) -> None:
        if isinstance(secret, str): secret = secret.encode("utf-8")
        if len(secret) < 32: raise AuthError("authentication secret must be at least 32 bytes")
        self._secret = bytes(secret)
        self._clock = clock or (lambda: int(datetime.now(timezone.utc).timestamp()))

    @staticmethod
    def generate_secret() -> str: return secrets.token_urlsafe(48)

    def issue(self, actor_id: str, role: str, ttl_seconds: int = 3600, now: int | None = None) -> str:
        current = self._clock() if now is None else int(now)
        if ttl_seconds <= 0: raise AuthError("ttl_seconds must be positive")
        actor = Actor(actor_id.strip(), role.strip(), current, current + ttl_seconds, secrets.token_hex(12))
        payload = _b64(json.dumps(actor.claims(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        signed = f"{_TOKEN_PREFIX}.{payload}".encode("ascii")
        signature = _b64(hmac.new(self._secret, signed, hashlib.sha256).digest())
        return f"{_TOKEN_PREFIX}.{payload}.{signature}"

    def verify(self, token: str, now: int | None = None) -> Actor:
        parts = token.strip().split(".")
        if len(parts) != 3 or parts[0] != _TOKEN_PREFIX: raise AuthError("invalid token format")
        _, payload, signature = parts
        signed = f"{_TOKEN_PREFIX}.{payload}".encode("ascii")
        expected = _b64(hmac.new(self._secret, signed, hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected): raise AuthError("invalid token signature")
        try: claims: Mapping[str, Any] = json.loads(_unb64(payload).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc: raise AuthError("invalid token payload") from exc
        try:
            actor = Actor(str(claims["actor_id"]), str(claims["role"]), int(claims["iat"]), int(claims["exp"]), str(claims["jti"]), str(claims["schema_version"]))
        except (KeyError, TypeError, ValueError) as exc: raise AuthError("invalid token claims") from exc
        current = self._clock() if now is None else int(now)
        if current >= actor.expires_at: raise AuthError("token expired")
        if actor.issued_at > current + 60: raise AuthError("token issued in the future")
        return actor
