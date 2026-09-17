"""Reusable CI smoke for the Phase 10 gateway startup boundary."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "WAF_ENV": "development",
        "WAF_GATEWAY_HOST": "127.0.0.1",
        "WAF_GATEWAY_PORT": "18082",
        "WAF_UPSTREAM_URL": "http://127.0.0.1:19091",
        "WAF_RATE_LIMIT_PER_MINUTE": "1000",
    })
    upstream = subprocess.Popen(
        [sys.executable, "-m", "http.server", "19091", "--bind", "127.0.0.1", "--directory", str(ROOT)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )
    gateway = subprocess.Popen(
        [sys.executable, "-m", "waf.gateway.proxy"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if gateway.poll() is not None:
                output = gateway.stdout.read()[-4000:] if gateway.stdout else ""
                print(output, file=sys.stderr)
                return 1
            try:
                with urlopen("http://127.0.0.1:18082/__waf_health", timeout=1) as response:
                    if response.status == 200:
                        print("phase10-ci-startup: PASS")
                        return 0
            except Exception:
                time.sleep(0.25)
        return 1
    finally:
        for process in (gateway, upstream):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
