"""Server-side security storage adapter with a development-safe memory mode."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from threading import Lock
from typing import Any, Mapping
from urllib import error, request

from waf.security.production_security import audit_record


def hash_identifier(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_path(value: str) -> str:
    """Persist endpoint path only, never query material."""
    raw = str(value or "")
    path = raw.split("?", 1)[0].split("#", 1)[0]
    return path[:8192] or "/"


class _RuntimeView:
    def __init__(self) -> None:
        self._lock = Lock()
        self._total = 0
        self._threats = 0
        self._blocked = 0
        self._recent: deque[dict[str, Any]] = deque(maxlen=500)

    def record(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> None:
        path = _safe_path(uri)
        with self._lock:
            self._total += 1
            if result.get("threat_detected"):
                self._threats += 1
                self._recent.append({
                    "threat_type": str(result.get("threat_type", "WAF_DECISION")),
                    "severity": str(result.get("severity", "low")),
                    "source_ip": hash_identifier(source_ip) or "unknown",
                    "target_endpoint": path,
                    "payload": None,
                    "confidence": float(result.get("risk_score", 0.0)),
                    "blocked": bool(result.get("blocked")),
                    "metadata": {"request_id": request_id, "privacy": "raw_payload_excluded"},
                })
            if result.get("blocked"):
                self._blocked += 1

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "total_requests": self._total,
                "total_threats": self._threats,
                "blocked_requests": self._blocked,
                "active_rules": 0,
                "detection_rate": round((self._blocked / self._total) if self._total else 0.0, 6),
            }

    def recent_threats(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(list(self._recent)[-max(1, min(limit, 500)):]))


class MemorySecurityStore:
    """Bounded in-process store used only for local/dev operation."""

    def __init__(self) -> None:
        self.threats: list[dict[str, Any]] = []
        self.request_logs: list[dict[str, Any]] = []
        self.analytics: list[dict[str, Any]] = []
        self.audit: list[dict[str, Any]] = []
        self._view = _RuntimeView()

    def record_decision_view(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> None:
        self._view.record(request_id=request_id, source_ip=source_ip, method=method, uri=uri, result=result)

    def record_decision(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> None:
        source_hash = hash_identifier(source_ip)
        path = _safe_path(uri)
        if not self._already_previewed(request_id):
            self._view.record(request_id=request_id, source_ip=source_ip, method=method, uri=uri, result=result)
        if result.get("threat_detected"):
            self.threats.append({
                "threat_type": result["threat_type"],
                "severity": result["severity"],
                "source_ip": source_hash or "unknown",
                "target_endpoint": path,
                "payload": None,
                "confidence": float(result["risk_score"]),
                "blocked": bool(result["blocked"]),
                "metadata": {"request_id": request_id, "privacy": "raw_payload_excluded"},
            })
        self.request_logs.append({
            "request_id": request_id,
            "method": method,
            "uri": path,
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

    def _already_previewed(self, request_id: str) -> bool:
        # Preview deduplication is only used by async local telemetry. The view is
        # bounded and the request id is already stored in threat metadata when relevant.
        return any(row.get("metadata", {}).get("request_id") == request_id for row in self._view.recent_threats(500)) or any(row.get("request_id") == request_id for row in self.request_logs[-1:])

    def record_audit(self, *, actor: str, action: str, target: str, outcome: str, request_id: str) -> None:
        self.audit.append(audit_record(actor=actor, action=action, target=target, outcome=outcome, request_id=request_id))

    def stats(self) -> dict[str, int | float]:
        return self._view.stats()

    def recent_threats(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._view.recent_threats(limit)


class SupabaseRESTStore:
    """Server-only Supabase REST persistence with a local bounded runtime view."""

    def __init__(self, url: str, service_role_key: str, timeout: float = 5.0) -> None:
        if not url.startswith("https://"):
            raise ValueError("Supabase REST storage requires an https:// URL")
        if not service_role_key:
            raise ValueError("server-side Supabase service-role key is required")
        self.base_url = url.rstrip("/")
        self.key = service_role_key
        self.timeout = timeout
        self._view = _RuntimeView()
        self._previewed: set[str] = set()
        self._preview_lock = Lock()

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

    def record_decision_view(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> None:
        with self._preview_lock:
            first = request_id not in self._previewed
            if first:
                self._previewed.add(request_id)
        if first:
            self._view.record(request_id=request_id, source_ip=source_ip, method=method, uri=uri, result=result)

    def record_decision(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> None:
        path = _safe_path(uri)
        self.record_decision_view(request_id=request_id, source_ip=source_ip, method=method, uri=uri, result=result)
        source_hash = hash_identifier(source_ip) or "unknown"
        if result.get("threat_detected"):
            self._post("threats", {
                "threat_type": result["threat_type"],
                "severity": result["severity"],
                "source_ip": source_hash,
                "target_endpoint": path,
                "payload": None,
                "confidence": float(result["risk_score"]),
                "blocked": bool(result["blocked"]),
                "metadata": {"request_id": request_id, "privacy": "raw_payload_excluded"},
            })
        self._post("request_logs", {
            "request_id": request_id,
            "method": method,
            "uri": path,
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

    def stats(self) -> dict[str, int | float]:
        return self._view.stats()

    def recent_threats(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._view.recent_threats(limit)
