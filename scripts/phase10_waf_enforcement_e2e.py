"""Prove real client -> ModSecurity -> ML gateway -> upstream enforcement.

The test is intentionally process-level: a protected upstream emits a unique marker
when reached. A malicious request must return 403 and the marker must not appear.
The CI workflow installs the open-source nginx ModSecurity connector before running it.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.error import HTTPError
from urllib.request import Request as URLRequest, urlopen

ROOT = Path(__file__).resolve().parents[1]


class _UpstreamState:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.lock = Lock()


class _UpstreamHandler(BaseHTTPRequestHandler):
    state: _UpstreamState

    def do_GET(self) -> None:  # noqa: N802
        with self.state.lock:
            self.state.paths.append(self.path)
        body = b"SWAVLAMBAN_UPSTREAM_REACHED"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET

    def log_message(self, *_args: object) -> None:
        return


def _wait_http(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - diagnostic retry loop
            last = exc
        time.sleep(0.25)
    raise RuntimeError(f"service did not become ready: {url}: {last}")


def _request(url: str) -> tuple[int, bytes]:
    try:
        with urlopen(URLRequest(url, method="GET"), timeout=5) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def _start_process(command: list[str], env: dict[str, str], log: Path) -> subprocess.Popen[str]:
    handle = log.open("w", encoding="utf-8")
    return subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)


def main() -> int:
    if shutil.which("nginx") is None:
        raise RuntimeError("nginx is required for Phase 10 enforcement evidence")
    module_candidates = list(Path("/usr/lib/nginx/modules").glob("ngx_http_modsecurity_module.so"))
    if not module_candidates:
        raise RuntimeError("ngx_http_modsecurity_module.so is required for Phase 10 enforcement evidence")
    module_path = module_candidates[0]

    state = _UpstreamState()
    handler_cls = type("ProtectedHandler", (_UpstreamHandler,), {"state": state})
    upstream = ThreadingHTTPServer(("127.0.0.1", 19090), handler_cls)
    upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    gateway_env = dict(os.environ)
    gateway_env.update({
        "WAF_ENV": "development",
        "WAF_GATEWAY_HOST": "127.0.0.1",
        "WAF_GATEWAY_PORT": "18081",
        "WAF_UPSTREAM_URL": "http://127.0.0.1:19090",
        "WAF_RATE_LIMIT_PER_MINUTE": "1000",
        "PYTHONPATH": str(ROOT),
    })

    with tempfile.TemporaryDirectory(prefix="swavlamban-phase10-nginx-") as tmp:
        tmpdir = Path(tmp)
        log_dir = tmpdir / "logs"
        log_dir.mkdir()
        nginx_conf = tmpdir / "nginx.conf"
        modsec_conf = ROOT / "deploy/nginx/phase10-modsecurity.conf"
        nginx_conf.write_text(
            "\n".join([
                f"load_module {module_path};",
                "worker_processes 1;",
                f"pid {tmpdir / 'nginx.pid'};",
                "events { worker_connections 256; }",
                "http {",
                f"  access_log {log_dir / 'access.log'};",
                f"  error_log {log_dir / 'error.log'} notice;",
                "  server {",
                "    listen 18080;",
                "    server_name localhost;",
                "    location / {",
                "      modsecurity on;",
                f"      modsecurity_rules_file {modsec_conf};",
                "      proxy_set_header X-WAF-Source-IP $remote_addr;",
                "      proxy_set_header X-Forwarded-Proto http;",
                "      proxy_pass http://127.0.0.1:18081;",
                "    }",
                "  }",
                "}",
                "",
            ]),
            encoding="utf-8",
        )
        gateway_log = tmpdir / "gateway.log"
        nginx_log = tmpdir / "nginx-process.log"
        gateway = _start_process([os.fspath("python"), "-m", "waf.gateway.proxy"], gateway_env, gateway_log)
        nginx_env = dict(os.environ)
        nginx = _start_process(["nginx", "-p", str(tmpdir), "-c", str(nginx_conf), "-g", "daemon off;"], nginx_env, nginx_log)
        try:
            _wait_http("http://127.0.0.1:18081/__waf_health")
            _wait_http("http://127.0.0.1:18080/health")

            before = list(state.paths)
            benign_status, benign_body = _request("http://127.0.0.1:18080/health")
            with state.lock:
                after_benign = list(state.paths)
            if benign_status != 200 or b"SWAVLAMBAN_UPSTREAM_REACHED" not in benign_body:
                raise AssertionError(f"benign request was not forwarded: {benign_status} {benign_body!r}")
            if len(after_benign) != len(before) + 1:
                raise AssertionError("protected upstream did not record benign forwarded request")

            malicious_status, malicious_body = _request("http://127.0.0.1:18080/search?q=%27%20OR%201%3D1--")
            with state.lock:
                after_sql = list(state.paths)
            if malicious_status != 403:
                raise AssertionError(f"ModSecurity did not block SQL request: {malicious_status} {malicious_body!r}")
            if len(after_sql) != len(after_benign):
                raise AssertionError("blocked request reached the protected upstream")
            if b"SWAVLAMBAN_UPSTREAM_REACHED" in malicious_body:
                raise AssertionError("blocked response leaked upstream success marker")

            summary = {
                "client_to_nginx": "PASS",
                "nginx_modsecurity_connector": "PASS",
                "benign_forwarded_to_protected_upstream": True,
                "sql_blocked_at_waf": True,
                "blocked_request_reached_upstream": False,
                "protected_upstream_marker": "SWAVLAMBAN_UPSTREAM_REACHED",
                "module": str(module_path),
                "modsecurity_policy": str(modsec_conf.relative_to(ROOT)),
                "log_files": [str(gateway_log), str(nginx_log), str(log_dir / "access.log"), str(log_dir / "error.log")],
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            (ROOT / "phase10_waf_enforcement_evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            return 0
        finally:
            for process in (nginx, gateway):
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
