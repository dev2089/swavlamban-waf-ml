"""Real HTTP reverse-proxy seam for the ML-integrated WAF.

The gateway performs the security decision before forwarding an allowed request to
an upstream application, then inspects the outbound response before returning it
back to the client. It is deliberately independent from Supabase so the request
path has no synchronous database dependency.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from aiohttp import ClientSession, ClientTimeout, web

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.storage.production import MemorySecurityStore


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    upstream_url: str = "http://127.0.0.1:9000"
    listen_host: str = "127.0.0.1"
    listen_port: int = 18081
    request_timeout_seconds: float = 10.0
    max_body_bytes: int = 1_048_576
    max_response_bytes: int = 10_485_760
    rate_limit_per_minute: int = 120

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "GatewayConfig":
        e = os.environ if env is None else env
        upstream = e.get("WAF_UPSTREAM_URL", "http://127.0.0.1:9000").rstrip("/")
        if not upstream.startswith(("http://", "https://")):
            raise ValueError("WAF_UPSTREAM_URL must use http:// or https://")
        values = {
            "upstream_url": upstream,
            "listen_host": e.get("WAF_GATEWAY_HOST", "127.0.0.1"),
            "listen_port": int(e.get("WAF_GATEWAY_PORT", "18081")),
            "request_timeout_seconds": float(e.get("WAF_REQUEST_TIMEOUT_SECONDS", "10.0")),
            "max_body_bytes": int(e.get("WAF_MAX_BODY_BYTES", "1048576")),
            "max_response_bytes": int(e.get("WAF_MAX_RESPONSE_BYTES", "10485760")),
            "rate_limit_per_minute": int(e.get("WAF_RATE_LIMIT_PER_MINUTE", "120")),
        }
        if not 1 <= values["listen_port"] <= 65535:
            raise ValueError("WAF_GATEWAY_PORT must be in [1, 65535]")
        if values["request_timeout_seconds"] <= 0:
            raise ValueError("WAF_REQUEST_TIMEOUT_SECONDS must be positive")
        if values["max_body_bytes"] <= 0 or values["max_response_bytes"] <= 0:
            raise ValueError("body/response size limits must be positive")
        if values["rate_limit_per_minute"] < 1:
            raise ValueError("WAF_RATE_LIMIT_PER_MINUTE must be positive")
        return cls(**values)


@dataclass(slots=True)
class _RateLimiter:
    limit: int
    window_seconds: float = 60.0
    buckets: dict[str, Deque[float]] = field(default_factory=dict)

    def allow(self, key: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        q = self.buckets.setdefault(key, deque())
        cutoff = current - self.window_seconds
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(current)
        if len(self.buckets) > 4096:
            stale = [k for k, values in self.buckets.items() if not values or values[-1] <= cutoff]
            for k in stale[:1024]:
                self.buckets.pop(k, None)
        return True


class WAFGateway:
    """Inspect, decide, forward only allowed traffic, then inspect outbound responses."""

    def __init__(self, waf: EdgeWAF, config: GatewayConfig | None = None) -> None:
        self.waf = waf
        self.config = config or GatewayConfig.from_env()
        self.store = MemorySecurityStore()
        self._rate = _RateLimiter(self.config.rate_limit_per_minute)
        self._session: ClientSession | None = None

    async def startup(self, app: web.Application) -> None:
        self._session = ClientSession(timeout=ClientTimeout(total=self.config.request_timeout_seconds))

    async def cleanup(self, app: web.Application) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @staticmethod
    def _source_ip(request: web.Request) -> str:
        return request.headers.get("X-WAF-Source-IP") or request.remote or "unknown"

    @staticmethod
    def _request_id(request: web.Request) -> str:
        supplied = request.headers.get("X-Request-ID", "")
        if supplied and 8 <= len(supplied) <= 128 and all(c.isalnum() or c in "._:-" for c in supplied):
            return supplied
        return str(uuid.uuid4())

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "service": "swavlamban-waf-gateway",
            "upstream_configured": bool(self.config.upstream_url),
            "rate_limit_per_minute": self.config.rate_limit_per_minute,
        })

    async def handle(self, request: web.Request) -> web.StreamResponse:
        source_ip = self._source_ip(request)
        if not self._rate.allow(source_ip):
            return web.json_response({"error": "rate limit exceeded"}, status=429)

        body = await request.content.read(self.config.max_body_bytes + 1)
        if len(body) > self.config.max_body_bytes:
            return web.json_response({"error": "request body exceeds configured limit"}, status=413)

        rid = self._request_id(request)
        envelope = RequestEnvelope(
            request_id=rid,
            method=request.method.upper(),
            scheme=request.headers.get("X-Forwarded-Proto", request.scheme),
            host=request.headers.get("Host", "localhost"),
            path=request.rel_url.path,
            query=request.rel_url.query_string,
            headers=dict(request.headers),
            body=body,
            source_ip=source_ip,
            timestamp=time.time(),
        )
        try:
            result = self.waf.analyze(envelope)
        except ValueError:
            return web.json_response({"error": "invalid security input", "request_id": rid}, status=400)

        safe_summary = {
            "threat_detected": result.decision is not Decision.ALLOW,
            "threat_type": "WAF_DECISION",
            "severity": "high" if result.decision is Decision.BLOCK else "low",
            "risk_score": result.risk_score,
            "blocked": result.decision is Decision.BLOCK,
        }
        self.store.record_decision(request_id=rid, source_ip=source_ip, method=request.method, uri=request.rel_url.path_qs, result=safe_summary)

        if result.decision is Decision.BLOCK:
            return web.json_response({
                "request_id": rid,
                "decision": "block",
                "blocked": True,
                "risk_score": result.risk_score,
                "reasons": list(result.reasons),
                "rule_ids": list(result.rule_ids),
                "evidence": {
                    "schema_version": result.evidence.schema_version if result.evidence else None,
                    "explanation": result.evidence.explanation if result.evidence else None,
                    "privacy": dict(result.evidence.privacy) if result.evidence else {},
                },
            }, status=403, headers={"X-Swavalamban-WAF-Decision": "block", "X-Swavalamban-WAF-Request-ID": rid})

        if self._session is None:
            return web.json_response({"error": "gateway not ready"}, status=503)

        target = self.config.upstream_url + request.rel_url.raw_path_qs
        hop_by_hop = {
            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
            "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
        }
        forwarded_headers = {k: v for k, v in request.headers.items() if k.lower() not in hop_by_hop}
        forwarded_headers["X-Swavalamban-WAF-Decision"] = result.decision.value
        forwarded_headers["X-Swavalamban-WAF-Risk"] = f"{result.risk_score:.6f}"
        forwarded_headers["X-Swavalamban-WAF-Request-ID"] = rid
        forwarded_headers["X-WAF-Source-IP"] = source_ip

        try:
            async with self._session.request(request.method, target, data=body, headers=forwarded_headers) as upstream:
                payload = await upstream.content.read(self.config.max_response_bytes + 1)
                if len(payload) > self.config.max_response_bytes:
                    return web.json_response({"error": "upstream response exceeds configured limit", "request_id": rid}, status=502)
                outbound = self.waf.inspect_response(upstream.status, dict(upstream.headers), payload)
                response_headers = {
                    k: v for k, v in upstream.headers.items() if k.lower() not in hop_by_hop
                }
                response_headers.update({
                    "X-Swavalamban-WAF-Decision": result.decision.value,
                    "X-Swavalamban-WAF-Risk": f"{result.risk_score:.6f}",
                    "X-Swavalamban-WAF-Request-ID": rid,
                    "X-Swavalamban-WAF-Outbound-Risk": f"{outbound.score:.6f}",
                    "X-Swavalamban-WAF-Outbound-Decision": "alert" if outbound.score >= 0.5 else "allow",
                    "X-Swavalamban-WAF-Outbound-Detector": outbound.detector,
                })
                return web.Response(status=upstream.status, body=payload, headers=response_headers)
        except asyncio.TimeoutError:
            return web.json_response({"error": "upstream timeout", "request_id": rid}, status=504)
        except Exception:
            return web.json_response({"error": "upstream unavailable", "request_id": rid}, status=502)


def create_gateway_app(*, env: dict[str, str] | None = None) -> web.Application:
    source = os.environ if env is None else env
    waf = EdgeWAF(WAFConfig.from_env(source))
    gateway = WAFGateway(waf, GatewayConfig.from_env(source))
    app = web.Application(client_max_size=gateway.config.max_body_bytes)
    app.router.add_get("/__waf_health", gateway.health)
    app.router.add_route("*", "/{path_info:.*}", gateway.handle)
    app.on_startup.append(gateway.startup)
    app.on_cleanup.append(gateway.cleanup)
    return app


if __name__ == "__main__":
    app = create_gateway_app()
    config = GatewayConfig.from_env()
    web.run_app(app, host=config.listen_host, port=config.listen_port)
