from __future__ import annotations

import asyncio
import time

from aiohttp import ClientSession, TCPConnector, web

from waf.core.config import WAFConfig
from waf.edge.reverse_proxy import WAFReverseProxy
from waf.ml.ensemble import Phase4MLEnsemble


def direct(total: int = 2000) -> dict[str, object]:
    from waf.core.models import RequestEnvelope
    from waf.edge.pipeline import EdgeWAF
    waf = EdgeWAF(WAFConfig())
    start = time.perf_counter()
    counts = {"allow": 0, "block": 0, "alert": 0}
    for i in range(total):
        req = RequestEnvelope(
            f"bench-{i}", "GET", "https", "example.test", f"/api/item/{i % 31}",
            "q=1%20UNION%20SELECT%20password%20FROM%20users" if i % 20 == 0 else f"q={i}",
            source_ip="127.0.0.1", timestamp=1000.0 + i * 0.1,
        )
        counts[waf.analyze(req).decision.value] += 1
    elapsed = time.perf_counter() - start
    return {"requests": total, "elapsed_seconds": round(elapsed, 4), "requests_per_second": round(total / elapsed, 2), **counts}


async def e2e(total: int = 1000, concurrency: int = 50) -> dict[str, object]:
    hits = 0

    async def upstream(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        return web.Response(text="upstream-ok")

    up = web.Application(client_max_size=2_000_000)
    up.router.add_route("*", "/{tail:.*}", upstream)
    up_runner = web.AppRunner(up)
    await up_runner.setup()
    await web.TCPSite(up_runner, "127.0.0.1", 19310).start()

    proxy = WAFReverseProxy(WAFConfig(upstream_url="http://127.0.0.1:19310", listen_port=18310))
    runner = web.AppRunner(proxy.app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18310).start()

    sem = asyncio.Semaphore(concurrency)
    statuses: dict[int, int] = {}
    latencies: list[float] = []
    async with ClientSession(connector=TCPConnector(limit=concurrency)) as client:
        async def one(i: int) -> None:
            async with sem:
                start = time.perf_counter()
                if i % 10 == 0:
                    url = "http://127.0.0.1:18310/?q=1%20UNION%20SELECT%20password%20FROM%20users"
                elif i % 20 == 1:
                    url = "http://127.0.0.1:18310/?q=%253Cscript%253Ealert(1)%253C%252Fscript%253E"
                else:
                    url = f"http://127.0.0.1:18310/api/item/{i % 31}?q={i}"
                try:
                    async with client.get(url, timeout=10) as response:
                        statuses[response.status] = statuses.get(response.status, 0) + 1
                        await response.read()
                except Exception:
                    statuses[500] = statuses.get(500, 0) + 1
                latencies.append((time.perf_counter() - start) * 1000)

        started = time.perf_counter()
        for offset in range(0, total, 250):
            await asyncio.gather(*(one(i) for i in range(offset, min(offset + 250, total))))
        elapsed = time.perf_counter() - started

    result = {
        "requests": total, "concurrency": concurrency, "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(total / elapsed, 2), "status_200": statuses.get(200, 0),
        "status_403": statuses.get(403, 0), "status_500": statuses.get(500, 0),
        "protected_upstream_hits": hits, "events_retained": len(proxy.events),
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 3),
        "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3),
    }
    await proxy.close()
    await runner.cleanup()
    await up_runner.cleanup()
    return result


def main() -> None:
    print("model_metadata", Phase4MLEnsemble.train_default().metadata())
    print("direct", direct())
    print("e2e", asyncio.run(e2e()))


if __name__ == "__main__":
    main()
