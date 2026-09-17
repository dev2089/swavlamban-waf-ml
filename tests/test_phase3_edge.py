from __future__ import annotations

import asyncio

from aiohttp import ClientSession, web

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.edge.reverse_proxy import WAFReverseProxy


def test_edge_uses_http_v2():
    waf = EdgeWAF(WAFConfig())
    result = waf.analyze(RequestEnvelope("1", "GET", "https", "example.test", "/health"))
    assert result.decision is Decision.ALLOW
    assert waf.features.schema_version == "http-v2"


def test_real_enforcement_with_phase3_features():
    async def go():
        hits = {"n": 0}

        async def upstream(_request):
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
                allowed = await client.get("http://127.0.0.1:18200/ok?q=a=1&a=2")
                assert allowed.status == 200
                assert allowed.headers["X-WAF-Decision"] == "allow"
                assert hits["n"] == 1

                blocked = await client.get(
                    "http://127.0.0.1:18200/?q=%25253Cscript%25253Ealert(1)%25253C/script%25253E"
                )
                assert blocked.status == 403
                assert blocked.headers["X-WAF-Decision"] == "block"
                assert hits["n"] == 1
        finally:
            await proxy.close()
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(go())
