from __future__ import annotations

import asyncio
import time
from aiohttp import ClientSession, ClientTimeout, web

from waf.core.config import WAFConfig
from waf.edge.reverse_proxy import WAFReverseProxy


async def main() -> None:
    hits = {"n": 0}

    async def upstream(_request):
        hits["n"] += 1
        return web.Response(text="ok")

    upstream_app = web.Application()
    upstream_app.router.add_route("*", "/{tail:.*}", upstream)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    await web.TCPSite(upstream_runner, "127.0.0.1", 19300).start()

    proxy = WAFReverseProxy(WAFConfig(upstream_url="http://127.0.0.1:19300", listen_port=18300))
    proxy_runner = web.AppRunner(proxy.app)
    await proxy_runner.setup()
    await web.TCPSite(proxy_runner, "127.0.0.1", 18300).start()

    total = 5_000
    semaphore = asyncio.Semaphore(100)
    try:
        async with ClientSession(timeout=ClientTimeout(total=20)) as client:
            async def one(index: int) -> int:
                async with semaphore:
                    if index % 10 == 0:
                        url = "http://127.0.0.1:18300/?q=1%20union%20select%20password%20from%20users"
                    else:
                        url = f"http://127.0.0.1:18300/ok?q={index}&a={index % 7}"
                    async with client.get(url) as response:
                        return response.status

            started = time.perf_counter()
            statuses = await asyncio.gather(*(one(i) for i in range(total)))
            elapsed = time.perf_counter() - started
            print({
                "requests": total,
                "allowed": statuses.count(200),
                "blocked": statuses.count(403),
                "upstream_hits": hits["n"],
                "seconds": round(elapsed, 4),
                "requests_per_second": round(total / elapsed, 1),
                "bounded_events": len(proxy.events),
            })
    finally:
        await proxy.close()
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
