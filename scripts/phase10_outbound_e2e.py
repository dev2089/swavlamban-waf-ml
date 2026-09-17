"""Exercise the real gateway's outbound HTTP response inspection seam."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "phase10_outbound_evidence.json"


async def build_upstream() -> tuple[web.AppRunner, int]:
    async def normal(_: web.Request) -> web.Response:
        return web.Response(text='{"ok":true,"items":[1,2,3]}', content_type="application/json")

    async def anomalous(_: web.Request) -> web.Response:
        body = b"Traceback (most recent call last): Exception secret password=admin"
        return web.Response(body=body, status=500, content_type="text/plain")

    app = web.Application()
    app.router.add_get("/normal", normal)
    app.router.add_get("/anomalous", anomalous)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 19191)
    await site.start()
    return runner, 19191


async def main_async() -> dict[str, object]:
    upstream, _ = await build_upstream()
    gateway = None
    gateway_runner = None
    try:
        from waf.gateway.proxy import create_gateway_app

        env = dict(os.environ)
        env.update({
            "WAF_ENV": "development",
            "WAF_GATEWAY_HOST": "127.0.0.1",
            "WAF_GATEWAY_PORT": "18182",
            "WAF_UPSTREAM_URL": "http://127.0.0.1:19191",
            "WAF_RATE_LIMIT_PER_MINUTE": "1000",
        })
        gateway = create_gateway_app(env=env)
        gateway_runner = web.AppRunner(gateway)
        await gateway_runner.setup()
        await web.TCPSite(gateway_runner, "127.0.0.1", 18182).start()

        async with ClientSession(timeout=ClientTimeout(total=10)) as client:
            normal = await client.get("http://127.0.0.1:18182/normal")
            normal_body = await normal.text()
            anomalous = await client.get("http://127.0.0.1:18182/anomalous")
            anomaly_body = await anomalous.text()

        evidence = {
            "normal": {
                "status": normal.status,
                "outbound_decision": normal.headers.get("X-Swavalamban-WAF-Outbound-Decision"),
                "outbound_detector": normal.headers.get("X-Swavalamban-WAF-Outbound-Detector"),
                "outbound_risk": float(normal.headers.get("X-Swavalamban-WAF-Outbound-Risk", "0")),
                "body_preserved": "\"ok\":true" in normal_body,
            },
            "anomalous": {
                "status": anomalous.status,
                "outbound_decision": anomalous.headers.get("X-Swavalamban-WAF-Outbound-Decision"),
                "outbound_detector": anomalous.headers.get("X-Swavalamban-WAF-Outbound-Detector"),
                "outbound_risk": float(anomalous.headers.get("X-Swavalamban-WAF-Outbound-Risk", "0")),
                "body_preserved": "Traceback" in anomaly_body,
            },
            "direction": "upstream response -> client",
            "feature_schema": "http-response-v1",
            "raw_response_retained": False,
        }
        EVIDENCE.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        assert evidence["normal"]["status"] == 200
        assert evidence["normal"]["outbound_decision"] == "allow"
        assert evidence["anomalous"]["status"] == 500
        assert evidence["anomalous"]["outbound_decision"] == "alert"
        assert evidence["anomalous"]["outbound_risk"] >= 0.5
        return evidence
    finally:
        if gateway_runner is not None:
            await gateway_runner.cleanup()
        await upstream.cleanup()


def main() -> int:
    evidence = asyncio.run(main_async())
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
