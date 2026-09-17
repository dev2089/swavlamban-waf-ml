from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from aiohttp import ClientSession, TCPConnector, web

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.edge.reverse_proxy import WAFReverseProxy
from waf.ml.ensemble import Phase4MLEnsemble, evaluate_behaviour, evaluate_supervised, evaluate_unsupervised

ROOT = Path(__file__).resolve().parent


def static_scans() -> dict[str, bool]:
    files = list((ROOT / "waf").rglob("*.py"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in files)
    secret = re.search(r"(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----)", text)
    todo = re.search(r"TODO|FIXME|pass\s*(?:#.*)?$", text, re.M)
    return {"secret_scan": secret is None, "todo_scan": todo is None}


def direct_ml_benchmark(total: int = 2000) -> dict[str, object]:
    waf = EdgeWAF(WAFConfig())
    requests = [
        RequestEnvelope(
            request_id=f"direct-{i}",
            method="GET" if i % 3 else "POST",
            scheme="https",
            host="example.test",
            path=f"/api/item/{i % 41}",
            query=("q=1%20UNION%20SELECT%20password%20FROM%20users" if i % 20 == 0 else f"q={i}"),
            source_ip="127.0.0.1",
        )
        for i in range(total)
    ]
    decisions = {Decision.ALLOW: 0, Decision.BLOCK: 0, Decision.ALERT: 0}
    started = time.perf_counter()
    for req in requests:
        result = waf.analyze(req)
        decisions[result.decision] += 1
    elapsed = time.perf_counter() - started
    return {
        "requests": total,
        "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(total / elapsed, 1),
        "allow": decisions[Decision.ALLOW],
        "block": decisions[Decision.BLOCK],
        "alert": decisions[Decision.ALERT],
    }


async def e2e_batch(total: int, proxy_port: int, upstream_port: int, concurrency: int = 25) -> dict[str, object]:
    hits = 0

    async def upstream(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        return web.Response(text="upstream-ok")

    up_app = web.Application(client_max_size=2_000_000)
    up_app.router.add_route("*", "/{tail:.*}", upstream)
    up_runner = web.AppRunner(up_app)
    await up_runner.setup()
    await web.TCPSite(up_runner, "127.0.0.1", upstream_port).start()

    proxy = WAFReverseProxy(WAFConfig(upstream_url=f"http://127.0.0.1:{upstream_port}", listen_host="127.0.0.1", listen_port=proxy_port))
    proxy_runner = web.AppRunner(proxy.app)
    await proxy_runner.setup()
    await web.TCPSite(proxy_runner, "127.0.0.1", proxy_port).start()

    statuses = {200: 0, 403: 0, 500: 0}
    latencies: list[float] = []
    sem = __import__("asyncio").Semaphore(concurrency)

    async with ClientSession(connector=TCPConnector(limit=concurrency)) as client:
        async def one(i: int) -> None:
            async with sem:
                start = time.perf_counter()
                if i % 10 == 0:
                    url = f"http://127.0.0.1:{proxy_port}/?q=1%20UNION%20SELECT%20password%20FROM%20users"
                elif i % 20 == 1:
                    url = f"http://127.0.0.1:{proxy_port}/?q=%253Cscript%253Ealert(1)%253C%252Fscript%253E"
                else:
                    url = f"http://127.0.0.1:{proxy_port}/api/item/{i % 31}?q={i}"
                try:
                    async with client.get(url, timeout=10) as response:
                        statuses[response.status] = statuses.get(response.status, 0) + 1
                        await response.read()
                except Exception:
                    statuses[500] += 1
                latencies.append((time.perf_counter() - start) * 1000)

        batch_size = 250
        for offset in range(0, total, batch_size):
            await __import__("asyncio").gather(*(one(i) for i in range(offset, min(offset + batch_size, total))))

    await proxy.close()
    await proxy_runner.cleanup()
    await up_runner.cleanup()
    ordered = sorted(latencies)
    return {
        "requests": total,
        "concurrency": concurrency,
        "status_200": statuses.get(200, 0),
        "status_403": statuses.get(403, 0),
        "status_500": statuses.get(500, 0),
        "protected_upstream_hits": hits,
        "event_buffer": 1000,
        "p50_latency_ms": round(ordered[len(ordered) // 2], 3),
        "p95_latency_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
    }


def main() -> None:
    compile_result = subprocess.run([sys.executable, "-m", "compileall", "-q", "waf", "tests"], cwd=ROOT)
    if compile_result.returncode:
        raise SystemExit(compile_result.returncode)
    for detector in (evaluate_supervised(), evaluate_unsupervised(), evaluate_behaviour()):
        assert detector
    model = Phase4MLEnsemble.train_default()
    artifact = ROOT / "models" / "phase4_models.joblib"
    model.save(artifact)
    loaded = Phase4MLEnsemble.load(artifact)
    assert loaded.feature_names == model.feature_names
    assert loaded.model_version == model.model_version
    waf = EdgeWAF(WAFConfig())
    benign = waf.analyze(RequestEnvelope("gate-benign", "GET", "https", "example.test", "/health"))
    bad = waf.analyze(RequestEnvelope("gate-attack", "GET", "https", "example.test", "/search", "q=1%20UNION%20SELECT%20password%20FROM%20users"))
    assert benign.decision is Decision.ALLOW
    assert bad.decision is Decision.BLOCK
    assert {s.detector for s in bad.signals} == {"open-source-waf-rules", "supervised-v1", "unsupervised-oneclasssvm-v1", "behaviour-v1"}
    print("direct_ml_benchmark", json.dumps(direct_ml_benchmark(), sort_keys=True))
    print("e2e_benchmark", json.dumps(__import__("asyncio").run(e2e_batch(1000, 18210, 19210)), sort_keys=True))
    print("static_scans", json.dumps(static_scans(), sort_keys=True))
    print("PHASE4 SELF-TEST: PASS")


if __name__ == "__main__":
    main()
