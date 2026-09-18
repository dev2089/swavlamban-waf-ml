from __future__ import annotations

import asyncio

from aiohttp import ClientSession, web

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.edge.reverse_proxy import WAFReverseProxy


def test_edge_uses_http_v2() -> None:
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(
        RequestEnvelope("1", "GET", "https", "example.test", "/health")
    )
    assert result.decision is Decision.ALLOW
    assert waf.features.schema_version == "http-v2"


def test_double_encoded_xss_is_blocked() -> None:
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(
        RequestEnvelope(
            "2",
            "GET",
            "https",
            "example.test",
            "/",
            "q=%25253Cscript%253Ealert(1)%25253C/script%253E",
        )
    )
    assert result.decision is Decision.BLOCK
    assert "WAF-XSS-001" in result.rule_ids


def test_multi_class_payload_is_blocked() -> None:
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(
        RequestEnvelope(
            "3",
            "POST",
            "https",
            "example.test",
            "/login",
            "q=1%20union%20select",
            headers={"content-type": "application/x-www-form-urlencoded"},
            body=b"name=x; id",
        )
    )
    assert result.decision is Decision.BLOCK
    assert result.rule_ids


def test_live_phase3_enforcement() -> None:
    async def go() -> None:
        hits = {"n": 0}

        async def upstream(_request: web.Request) -> web.Response:
            hits["n"] += 1
            return web.Response(text="upstream-ok")

        upstream_app = web.Application()
        upstream_app.router.add_route("*", "/{tail:.*}", upstream)
        upstream_runner = web.AppRunner(upstream_app)
        await upstream_runner.setup()
        await web.TCPSite(upstream_runner, "127.0.0.1", 19200).start()

        proxy = WAFReverseProxy(
            WAFConfig(upstream_url="http://127.0.0.1:19200", listen_port=18200)
        )
        proxy_runner = web.AppRunner(proxy.app)
        await proxy_runner.setup()
        await web.TCPSite(proxy_runner, "127.0.0.1", 18200).start()

        try:
            async with ClientSession() as client:
                allowed = await client.get(
                    "http://127.0.0.1:18200/ok?q=a=1&a=2",
                    headers={"X-Request-ID": "phase3-allow"},
                )
                assert allowed.status == 200
                assert await allowed.text() == "upstream-ok"
                assert allowed.headers["X-WAF-Decision"] == "allow"
                assert allowed.headers["X-WAF-Request-ID"] == "phase3-allow"
                assert hits["n"] == 1

                blocked = await client.get(
                    "http://127.0.0.1:18200/?q=%25253Cscript%253Ealert(1)%25253C/script%253E",
                    headers={"X-Request-ID": "phase3-block"},
                )
                assert blocked.status == 403
                assert blocked.headers["X-WAF-Decision"] == "block"
                assert blocked.headers["X-WAF-Request-ID"] == "phase3-block"
                assert hits["n"] == 1
        finally:
            await proxy.close()
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(go())


def test_request_size_is_enforced() -> None:
    async def go() -> None:
        hits = {"n": 0}

        async def upstream(_request: web.Request) -> web.Response:
            hits["n"] += 1
            return web.Response(text="upstream-ok")

        upstream_app = web.Application()
        upstream_app.router.add_route("*", "/{tail:.*}", upstream)
        upstream_runner = web.AppRunner(upstream_app)
        await upstream_runner.setup()
        await web.TCPSite(upstream_runner, "127.0.0.1", 19201).start()

        proxy = WAFReverseProxy(
            WAFConfig(
                upstream_url="http://127.0.0.1:19201",
                listen_port=18201,
                max_body_bytes=16,
            )
        )
        proxy_runner = web.AppRunner(proxy.app)
        await proxy_runner.setup()
        await web.TCPSite(proxy_runner, "127.0.0.1", 18201).start()

        try:
            async with ClientSession() as client:
                response = await client.post(
                    "http://127.0.0.1:18201/upload",
                    data=b"x" * 64,
                    headers={"X-Request-ID": "too-large"},
                )
                assert response.status == 413
                assert hits["n"] == 0
        finally:
            await proxy.close()
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(go())


def test_upstream_response_is_bounded() -> None:
    async def go() -> None:
        async def upstream(_request: web.Request) -> web.Response:
            return web.Response(body=b"x" * 128)

        upstream_app = web.Application()
        upstream_app.router.add_route("*", "/{tail:.*}", upstream)
        upstream_runner = web.AppRunner(upstream_app)
        await upstream_runner.setup()
        await web.TCPSite(upstream_runner, "127.0.0.1", 19202).start()

        proxy = WAFReverseProxy(
            WAFConfig(
                upstream_url="http://127.0.0.1:19202",
                listen_port=18202,
                max_response_bytes=64,
            )
        )
        proxy_runner = web.AppRunner(proxy.app)
        await proxy_runner.setup()
        await web.TCPSite(proxy_runner, "127.0.0.1", 18202).start()

        try:
            async with ClientSession() as client:
                response = await client.get("http://127.0.0.1:18202/large")
                assert response.status == 502
                assert await response.text() == "upstream response too large"
        finally:
            await proxy.close()
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(go())
