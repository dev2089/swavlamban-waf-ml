from __future__ import annotations

import asyncio
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

from aiohttp import ClientSession, ClientTimeout, web

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.telemetry.events import decision_event


class WAFReverseProxy:
    """Inspect requests before forwarding and enforce bounded proxy I/O."""

    def __init__(self, config: WAFConfig) -> None:
        self.config = config
        self.waf = EdgeWAF(config)
        self.events: list[dict] = []
        self.client: ClientSession | None = None
        self.app = web.Application(client_max_size=config.max_body_bytes + 1)
        self.app.router.add_get("/__waf/health", self.health)
        self.app.router.add_route("*", "/{tail:.*}", self.handle)

    async def start(self) -> None:
        if self.client is None:
            self.client = ClientSession(
                timeout=ClientTimeout(total=self.config.request_timeout_seconds)
            )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "service": "swavlamban-waf-edge",
                "pipeline_version": self.config.pipeline_version,
                "feature_schema_version": self.config.feature_schema_version,
                "upstream": self.config.upstream_url,
            }
        )

    @staticmethod
    def _request_id(request: web.Request) -> str:
        supplied = request.headers.get("X-Request-ID", "")
        return supplied[:128] if supplied else str(uuid.uuid4())

    async def _read_bounded_response(self, response) -> bytes | None:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.content.iter_chunked(64 * 1024):
            total += len(chunk)
            if total > self.config.max_response_bytes:
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    async def handle(self, request: web.Request) -> web.Response:
        await self.start()
        started = time.perf_counter()
        request_id = self._request_id(request)

        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response(
                {
                    "error": "request_too_large",
                    "request_id": request_id,
                    "max_body_bytes": self.config.max_body_bytes,
                },
                status=413,
            )

        envelope = RequestEnvelope(
            request_id=request_id,
            method=request.method,
            scheme=request.scheme,
            host=request.headers.get("Host", ""),
            path=request.path,
            query=request.query_string,
            headers=dict(request.headers),
            body=body,
            source_ip=request.remote,
        )

        try:
            result = self.waf.analyze(envelope)
        except ValueError as exc:
            return web.json_response(
                {
                    "error": "invalid_security_input",
                    "detail": str(exc),
                    "request_id": request_id,
                },
                status=400,
            )

        event = decision_event(result)
        event["latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            3,
        )
        event["source_ip"] = request.remote
        self.events.append(event)
        self.events = self.events[-1000:]

        waf_headers = {
            "X-WAF-Decision": result.decision.value,
            "X-WAF-Request-ID": request_id,
            "X-WAF-Risk": str(result.risk_score),
        }

        if result.decision is Decision.BLOCK:
            return web.json_response(
                {
                    "blocked": True,
                    "request_id": request_id,
                    "decision": result.decision.value,
                    "risk_score": result.risk_score,
                    "reasons": list(result.reasons),
                    "rule_ids": list(result.rule_ids),
                },
                status=403,
                headers=waf_headers,
            )

        upstream = urlsplit(self.config.upstream_url)
        if upstream.scheme not in {"http", "https"} or not upstream.netloc:
            return web.json_response(
                {"error": "invalid_upstream_configuration", "request_id": request_id},
                status=500,
                headers=waf_headers,
            )

        target = urlunsplit(
            (upstream.scheme, upstream.netloc, request.path, request.query_string, "")
        )
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower()
            not in {"host", "content-length", "connection", "transfer-encoding"}
        }
        headers.update(
            {
                "Host": upstream.netloc,
                "X-Forwarded-For": request.remote or "",
                "X-Forwarded-Proto": request.scheme,
                "X-WAF-Decision": result.decision.value,
                "X-WAF-Risk": str(result.risk_score),
            }
        )

        assert self.client is not None
        try:
            async with self.client.request(
                request.method,
                target,
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as response:
                payload = await self._read_bounded_response(response)
                if payload is None:
                    return web.Response(
                        status=502,
                        text="upstream response too large",
                        headers={"X-WAF-Request-ID": request_id},
                    )
                passthrough = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() not in {"connection", "transfer-encoding", "content-length"}
                }
                passthrough.update(waf_headers)
                return web.Response(
                    status=response.status,
                    headers=passthrough,
                    body=payload,
                )
        except asyncio.TimeoutError:
            return web.json_response(
                {"error": "upstream_timeout", "request_id": request_id},
                status=504,
                headers=waf_headers,
            )
        except (OSError, asyncio.CancelledError):
            raise
        except Exception:
            return web.json_response(
                {"error": "upstream_unavailable", "request_id": request_id},
                status=502,
                headers=waf_headers,
            )


def build_app(config: WAFConfig | None = None) -> web.Application:
    config = config or WAFConfig.from_env()
    proxy = WAFReverseProxy(config)
    app = proxy.app
    app["waf_proxy"] = proxy
    app.on_startup.append(lambda _app: proxy.start())
    app.on_cleanup.append(lambda _app: proxy.close())
    return app
