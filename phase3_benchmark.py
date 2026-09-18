from __future__ import annotations

import asyncio
import time

from aiohttp import ClientSession, TCPConnector, web

from waf.core.config import WAFConfig
from waf.core.models import RequestEnvelope
from waf.edge.reverse_proxy import WAFReverseProxy
from waf.features.http_v2 import ProductionHTTPFeatureExtractor


def feature_benchmark() -> float:
    extractor = ProductionHTTPFeatureExtractor()
    requests = [
        RequestEnvelope(
            str(i),
            "GET" if i % 3 else "POST",
            "https",
            "example.test",
            f"/api/item/{i % 31}",
            f"q={i}&a={i % 7}",
            headers={"content-type": "application/json", "x-client": "bench"},
            body=b'{"item":"x"}',
        )
        for i in range(100_000)
    ]
    started = time.perf_counter()
    for request in requests:
        extractor.extract(request)
    elapsed = time.perf_counter() - started
    return len(requests) / elapsed


async def e2e_benchmark(total: int = 5_000, concurrency: int = 100) -> tuple[float, int, int, int]:
    hits = {"n": 0}

    async def upstream(_request: web.Request) -> web.Response:
        hits["n"] += 1
        return web.Response(text="ok")

    upstream_app = web.Application()
    upstream_app.router.add_route("*", "/{tail:.*}", upstream)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    await web.TCPSite(upstream_runner, "127.0.0.1", 19310).start()

    proxy = WAFReverseProxy(
        WAFConfig(upstream_url="http://127.0.0.1:19310", listen_port=18310)
    )
    proxy_runner = web.AppRunner(proxy.app)
    await proxy_runner.setup()
    await web.TCPSite(proxy_runner, "127.0.0.1", 18310).start()

    sem = asyncio.Semaphore(concurrency)

    async def one(client: ClientSession, index: int) -> int:
        async with sem:
            url = (
                "http://127.0.0.1:18310/?q=1%20union%20select%20password"
                if index % 10 == 0
                else f"http://127.0.0.1:18310/ok?q={index}"
            )
            async with client.get(url) as response:
                await response.read()
                return response.status

    started = time.perf_counter()
    try:
        connector = TCPConnector(limit=concurrency)
        async with ClientSession(connector=connector) as client:
            statuses = await asyncio.gather(*(one(client, i) for i in range(total)))
    finally:
        elapsed = time.perf_counter() - started
        await proxy.close()
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()

    allowed = statuses.count(200)
    blocked = statuses.count(403)
    expected_ok = allowed == total - total // 10 and blocked == total // 10
    if not expected_ok:
        raise AssertionError(
            f"unexpected benchmark distribution: allowed={allowed} blocked={blocked}"
        )
    return total / elapsed, allowed, blocked, hits["n"]


async def main() -> None:
    feature_rps = feature_benchmark()
    e2e_rps, allowed, blocked, upstream_hits = await e2e_benchmark()
    print(
        {
            "feature_extractions": 100_000,
            "feature_rps": round(feature_rps, 2),
            "e2e_requests": 5_000,
            "e2e_concurrency": 100,
            "e2e_rps": round(e2e_rps, 2),
            "allowed": allowed,
            "blocked": blocked,
            "upstream_hits": upstream_hits,
            "events_retained": min(5_000, 1_000),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
