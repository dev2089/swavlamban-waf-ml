from __future__ import annotations
import argparse, asyncio, time
from aiohttp import ClientSession, TCPConnector, web
from waf.core.config import WAFConfig
from waf.edge.reverse_proxy import WAFReverseProxy

async def main() -> None:
    parser=argparse.ArgumentParser(description="Repeatable local Phase 2 end-to-end benchmark")
    parser.add_argument("--requests",type=int,default=5000)
    parser.add_argument("--concurrency",type=int,default=100)
    args=parser.parse_args()
    if args.requests<=0 or args.concurrency<=0: raise SystemExit("requests and concurrency must be positive")
    async def upstream_handler(_request: web.Request) -> web.Response:
        return web.Response(text="ok")
    ua=web.Application(); ua.router.add_route("*","/{tail:.*}",upstream_handler)
    ur=web.AppRunner(ua); await ur.setup(); us=web.TCPSite(ur,"127.0.0.1",19002); await us.start()
    proxy=WAFReverseProxy(WAFConfig(upstream_url="http://127.0.0.1:19002",listen_port=18102))
    pr=web.AppRunner(proxy.app); await pr.setup(); ps=web.TCPSite(pr,"127.0.0.1",18102); await ps.start()
    async def one(session: ClientSession,i:int)->bool:
        url="http://127.0.0.1:18102/?q=union+select" if i%10==0 else "http://127.0.0.1:18102/ok"
        expected=403 if i%10==0 else 200
        async with session.get(url) as response:
            await response.read()
            return response.status==expected
    started=time.perf_counter(); passed=0
    try:
        connector=TCPConnector(limit=max(args.concurrency,1))
        async with ClientSession(connector=connector) as session:
            sem=asyncio.Semaphore(args.concurrency)
            async def bounded(i:int)->bool:
                async with sem: return await one(session,i)
            passed=sum(await asyncio.gather(*(bounded(i) for i in range(args.requests))))
    finally:
        elapsed=time.perf_counter()-started; await proxy.close(); await pr.cleanup(); await ur.cleanup()
    rps=args.requests/elapsed
    print(f"requests={args.requests} concurrency={args.concurrency} passed={passed} elapsed_s={elapsed:.6f} req_per_sec={rps:.3f} events_retained={min(args.requests,1000)}")
    if passed!=args.requests: raise SystemExit(1)

if __name__=="__main__":
    asyncio.run(main())
