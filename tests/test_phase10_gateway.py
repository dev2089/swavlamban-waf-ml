import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient as AiohttpTestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waf.core.config import WAFConfig
from waf.edge.pipeline import EdgeWAF
from waf.gateway.proxy import GatewayConfig, WAFGateway

SECRET = "phase10-gateway-secret-" + "x" * 32


def _waf() -> EdgeWAF:
    return EdgeWAF(WAFConfig.from_env({"WAF_ENV": "development", "WAF_AUTH_SECRET": SECRET}))


@pytest.mark.asyncio
async def test_gateway_forwards_benign_and_blocks_sql():
    calls = []

    async def protected(request):
        calls.append(request.path_qs)
        return web.Response(text="SWAVLAMBAN_UPSTREAM_REACHED")

    upstream_app = web.Application()
    upstream_app.router.add_route("*", "/{path:.*}", protected)
    upstream = TestServer(upstream_app)
    await upstream.start_server()
    gateway_server = None
    try:
        config = GatewayConfig(upstream_url=str(upstream.make_url("/")).rstrip("/"), rate_limit_per_minute=100)
        gateway = WAFGateway(_waf(), config)
        app = web.Application(client_max_size=config.max_body_bytes + 1)
        app.router.add_get("/__waf_health", gateway.health)
        app.router.add_route("*", "/{path_info:.*}", gateway.handle)
        app.on_startup.append(gateway.startup)
        app.on_cleanup.append(gateway.cleanup)
        gateway_server = TestServer(app)
        await gateway_server.start_server()
        async with AiohttpTestClient(gateway_server) as client:
            benign = await client.get("/health")
            assert benign.status == 200
            assert "SWAVLAMBAN_UPSTREAM_REACHED" in await benign.text()
            assert calls == ["/health"]

            sql = await client.get("/search?q=' OR 1=1--")
            assert sql.status == 403
            assert "block" in (await sql.text()).lower()
            assert calls == ["/health"]
    finally:
        if gateway_server is not None:
            await gateway_server.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_gateway_rejects_oversized_and_rate_limited_requests():
    async def protected(request):
        return web.Response(text="SWAVLAMBAN_UPSTREAM_REACHED")

    upstream_app = web.Application()
    upstream_app.router.add_route("*", "/{path:.*}", protected)
    upstream = TestServer(upstream_app)
    await upstream.start_server()
    gateway_server = None
    try:
        config = GatewayConfig(upstream_url=str(upstream.make_url("/")).rstrip("/"), max_body_bytes=8, rate_limit_per_minute=2)
        gateway = WAFGateway(_waf(), config)
        app = web.Application(client_max_size=config.max_body_bytes + 1)
        app.router.add_route("*", "/{path_info:.*}", gateway.handle)
        app.on_startup.append(gateway.startup)
        app.on_cleanup.append(gateway.cleanup)
        gateway_server = TestServer(app)
        await gateway_server.start_server()
        async with AiohttpTestClient(gateway_server) as client:
            too_big = await client.post("/upload", data="123456789")
            assert too_big.status == 413
            first = await client.get("/one")
            second = await client.get("/two")
            assert first.status == 200
            assert second.status == 429
    finally:
        if gateway_server is not None:
            await gateway_server.close()
        await upstream.close()
