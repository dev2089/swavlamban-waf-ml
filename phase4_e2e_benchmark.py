from __future__ import annotations

import asyncio
import time
from statistics import median

from aiohttp import ClientSession, TCPConnector, web

from waf.core.config import WAFConfig
from waf.edge.reverse_proxy import WAFReverseProxy


async def run() -> dict[str, object]:
    hits = 0

    async def upstream(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        return web.Response(text="upstream-ok")

    up_app = web.Application(client_max_size=2_000_000)
    up_app.router.add_route("*", "/{tail:.*}", upstream)
    up_runner = web.AppRunner(up_app)
    await up_runner.setup()
    await web.TCPSite(up_runner, "127.0.0.1", 19200).start()

    proxy = WAFReverseProxy(WAFConfig(upstream_url="http://127.0.0.1:19200", listen_host="127.0.0.1", listen_port=18200))
    proxy_runner = web.AppRunner(proxy.app)
    await proxy_runner.setup()
    await web.TCPSite(proxy_runner, "127.0.0.1", 18200).start()

    total = 5000
    concurrent = 50
    sem = asyncio.Semaphore(concurrent)
    statuses = {200: 0, 403: 0, 500: 0}
    latencies: list[float] = []

    async with ClientSession(connector=TCPConnector(limit=concurrent)) as client:
        async def one(i: int) -> None:
            async with sem:
                start = time.perf_counter()
                if i % 10 == 0:
                    url = "http://127.0.0.1:18200/?q=1%20UNION%20SELECT%20password%20FROM%20users"
                elif i % 20 == 1:
                    url = "http://127.0.0.1:18200/?q=%253Cscript%253Ealert(1)%253C%252Fscript%253E"
                else:
                    url = f"http://127.0.0.1:18200/api/item/{i % 31}?q={i}"
                try:
                    async with client.get(url) as response:
                        statuses[response.status] = statuses.get(response.status, 0) + 1
                        await response.read()
                except Exception:
                    statuses[500] += 1
                latencies.append((time.perf_counter() - start) * 1000)

        started = time.perf_counter()
        batch_size = 250
        for offset in range(0, total, batch_size):
            await asyncio.gather(*(one(i) for i in range(offset, min(offset + batch_size, total))))
        elapsed = time.perf_counter() - started

    result = {
        "requests": total,
        "concurrency": concurrent,
        "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(total / elapsed, 1),
        "status_200": statuses.get(200, 0),
        "status_403": statuses.get(403, 0),
        "status_500": statuses.get(500, 0),
        "protected_upstream_hits": hits,
        "event_buffer": len(proxy.events),
        "p50_latency_ms": round(median(latencies), 3),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
    }
    await proxy.close()
    await proxy_runner.cleanup()
    await up_runner.cleanup()
    return result


if __name__ == "__main__":
    print(asyncio.run(run()))
