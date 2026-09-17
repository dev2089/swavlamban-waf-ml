"""Server-side security storage adapter with a development-safe memory mode."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib import error, request

from waf.security.production_security import audit_record


def hash_identifier(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class MemorySecurityStore:
    threats: list[dict[str, Any]]
    request_logs: list[dict[str, Any]]
    analytics: list[dict[str, Any]]
    audit: list[dict[str, Any]]

    def __init__(self) -> None:
        self.threats = []
        self.request_logs = []
        self.analytics = []
        self.audit = []

    def record_decision(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> None:
        source_hash = hash_identifier(source_ip)
        if result.get("threat_detected"):
            self.threats.append({
                "threat_type": result["threat_type"],
                "severity": result["severity"],
                "source_ip": source_hash or "unknown",
                "target_endpoint": uri,
                "payload": None,
                "confidence": float(result["risk_score"]),
                "blocked": bool(result["blocked"]),
                "metadata": {"request_id": request_id, "privacy": "raw_payload_excluded"},
            })
        self.request_logs.append({
            "request_id": request_id,
            "method": method,
            "uri": uri,
            "source_ip": source_hash or "unknown",
            "user_agent": None,
            "headers": {},
            "query_params": {},
            "body": None,
            "threat_detected": bool(result.get("threat_detected")),
            "blocked": bool(result.get("blocked")),
            "ml_scores": result.get("ml_scores", {}),
        })
        self.analytics.append({"metric_name": "requests_total", "metric_value": 1})
        if result.get("threat_detected"):
            self.analytics.append({"metric_name": "threats_total", "metric_value": 1})

    def record_audit(self, *, actor: str, action: str, target: str, outcome: str, request_id: str) -> None:
        self.audit.append(audit_record(actor=actor, action=action, target=target, outcome=outcome, request_id=request_id))

    def stats(self) -> dict[str, int | float]:
        total = len(self.request_logs)
        blocked = sum(1 for row in self.request_logs if row["blocked"])
        return {
            "total_requests": total,
            "total_threats": len(self.threats),
            "blocked_requests": blocked,
            "active_rules": 0,
            "detection_rate": round((blocked / total) if total else 0.0, 6),
        }

    def recent_threats(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(reversed(self.threats[-max(1, min(limit, 500)):]))


class SupabaseRESTStore:
    """Minimal server-only Supabase REST adapter.

    The service-role key never leaves the backend process. Only sanitized decision
    summaries are transmitted. Raw body, raw headers and raw query material are
    deliberately excluded from every persisted payload.
    """

    def __init__(self, url: str, service_role_key: str, timeout: float = 5.0) -> None:
        if not url.startswith("https://"):
            raise ValueError("Supabase REST storage requires an https:// URL")
        if not service_role_key:
            raise ValueError("server-side Supabase service-role key is required")
        self.base_url = url.rstrip("/")
        self.key = service_role_key
        self.timeout = timeout

    def _post(self, table: str, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/rest/v1/{table}",
            data=encoded,
            method="POST",
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                if response.status >= 300:
                    raise RuntimeError(f"Supabase write failed with HTTP {response.status}")
        except error.HTTPError as exc:
            raise RuntimeError(f"Supabase write failed with HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError("Supabase storage unavailable") from exc

    def record_decision(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> None:
        source_hash = hash_identifier(source_ip) or "unknown"
        if result.get("threat_detected"):
            self._post("threats", {
                "threat_type": result["threat_type"],
                "severity": result["severity"],
                "source_ip": source_hash,
                "target_endpoint": uri,
                "payload": None,
                "confidence": float(result["risk_score"]),
                "blocked": bool(result["blocked"]),
                "metadata": {"request_id": request_id, "privacy": "raw_payload_excluded"},
            })
        self._post("request_logs", {
            "request_id": request_id,
            "method": method,
            "uri": uri,
            "source_ip": source_hash,
            "user_agent": None,
            "headers": {},
            "query_params": {},
            "body": None,
            "threat_detected": bool(result.get("threat_detected")),
            "blocked": bool(result.get("blocked")),
            "ml_scores": result.get("ml_scores", {}),
        })
        self._post("analytics", {"metric_name": "requests_total", "metric_value": 1, "metric_type": "counter"})
        if result.get("threat_detected"):
            self._post("analytics", {"metric_name": "threats_total", "metric_value": 1, "metric_type": "counter"})

    def record_audit(self, *, actor: str, action: str, target: str, outcome: str, request_id: str) -> None:
        record = audit_record(actor=actor, action=action, target=target, outcome=outcome, request_id=request_id)
        self._post("waf_security_audit", {
            "actor_id": record["actor"],
            "action": record["action"],
            "target": record["target"],
            "outcome": record["outcome"],
            "request_id": record["request_id"],
            "metadata": {"recorded_at": record["recorded_at"]},
        })
