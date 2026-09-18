from __future__ import annotations
import asyncio
from aiohttp import ClientSession, web
from waf.core.config import WAFConfig
from waf.edge.reverse_proxy import WAFReverseProxy

async def main() -> None:
    async def upstream_handler(request: web.Request) -> web.Response:
        return web.json_response({"upstream": "protected", "path": request.path})
    upstream_app=web.Application(client_max_size=1024**2)
    upstream_app.router.add_route("*","/{tail:.*}",upstream_handler)
    upstream_runner=web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site=web.TCPSite(upstream_runner,"127.0.0.1",19001)
    await upstream_site.start()
    proxy=WAFReverseProxy(WAFConfig(upstream_url="http://127.0.0.1:19001",listen_port=18101))
    proxy_runner=web.AppRunner(proxy.app)
    await proxy_runner.setup()
    proxy_site=web.TCPSite(proxy_runner,"127.0.0.1",18101)
    await proxy_site.start()
    try:
        async with ClientSession() as client:
            good=await client.get("http://127.0.0.1:18101/health")
            print("ALLOW",good.status,good.headers.get("X-WAF-Decision"),await good.text())
            bad=await client.get("http://127.0.0.1:18101/?q=%3Cscript%3Ealert(1)%3C/script%3E")
            print("BLOCK",bad.status,bad.headers.get("X-WAF-Decision"),await bad.text())
    finally:
        await proxy.close(); await proxy_runner.cleanup(); await upstream_runner.cleanup()

if __name__=="__main__":
    asyncio.run(main())
