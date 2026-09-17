"""Configurable reproducible HTTP load harness for the WAF edge.

This reports achieved values only. It never converts a short local run into a
million-request claim. The same harness supports bounded duration/rate/concurrency
profiles and can target a local network endpoint or any reachable test endpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from urllib.parse import urljoin

import aiohttp


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


async def run(args: argparse.Namespace) -> dict[str, object]:
    sem = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    errors = 0
    started_at = time.perf_counter()
    deadline = started_at + args.duration
    request_count = 0
    lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=max(args.concurrency, 1), ssl=False)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def one(index: int) -> None:
            nonlocal errors, request_count
            async with sem:
                profile = args.payload_profile
                if profile == "benign":
                    path = "/health"
                elif profile == "mixed":
                    path = "/search?q=hello" if index % 10 else "/search?q=' OR 1=1--"
                else:
                    path = "/search?q=' OR 1=1--"
                url = urljoin(args.url.rstrip("/") + "/", path.lstrip("/"))
                t0 = time.perf_counter()
                try:
                    async with session.get(url) as response:
                        await response.read()
                        elapsed = (time.perf_counter() - t0) * 1000
                        latencies.append(elapsed)
                        statuses[response.status] += 1
                except Exception:
                    errors += 1
                async with lock:
                    request_count += 1

        workers: set[asyncio.Task[None]] = set()
        interval = 0.0 if args.rate <= 0 else 1.0 / args.rate
        index = 0
        while time.perf_counter() < deadline:
            task = asyncio.create_task(one(index))
            workers.add(task)
            task.add_done_callback(workers.discard)
            index += 1
            if interval:
                await asyncio.sleep(interval)
            elif len(workers) >= args.concurrency:
                await asyncio.sleep(0)
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)

    elapsed_total = max(0.001, time.perf_counter() - started_at)
    successful = sum(code < 500 for code in statuses.elements())
    result = {
        "url": args.url,
        "duration_seconds": round(elapsed_total, 4),
        "concurrency": args.concurrency,
        "requested_rate_per_second": args.rate,
        "payload_profile": args.payload_profile,
        "requests_attempted": request_count,
        "responses": sum(statuses.values()),
        "achieved_requests_per_second": round(request_count / elapsed_total, 3),
        "success_like_responses": successful,
        "error_count": errors,
        "error_rate": round(errors / max(1, request_count), 6),
        "status_counts": dict(statuses),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 4),
            "p95": round(percentile(latencies, 95), 4),
            "p99": round(percentile(latencies, 99), 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "scope": "reproducible bounded local test run",
        "millions_of_requests_claimed": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--payload-profile", choices=["benign", "mixed", "attack"], default="benign")
    parser.add_argument("--output", default="phase10_load_evidence.json")
    args = parser.parse_args()
    if args.duration <= 0 or args.concurrency <= 0 or args.rate < 0 or args.timeout <= 0:
        raise SystemExit("duration/concurrency/timeout must be positive and rate must be non-negative")
    result = asyncio.run(run(args))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
