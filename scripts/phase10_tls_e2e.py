"""Process-level HTTPS termination test for the WAF gateway.

A local self-signed certificate terminates TLS at nginx. The decrypted request is
then inspected by the Swavlamban gateway before reaching the protected upstream.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
MARKER = b"SWAVLAMBAN_TLS_UPSTREAM_REACHED"


class Handler(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):  # noqa: N802
        type(self).hits += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(MARKER)

    def log_message(self, *args):
        return


def request(url: str) -> tuple[int, bytes]:
    try:
        with urlopen(url, timeout=5) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def wait_port(host: str, port: int, timeout: float = 12.0):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"port {host}:{port} did not open")


def main() -> int:
    for tool in ("nginx", "openssl", "curl"):
        if not shutil.which(tool):
            raise SystemExit(f"required tool missing: {tool}")
    with tempfile.TemporaryDirectory(prefix="swavlamban-phase10-tls-") as td:
        tmp = Path(td)
        cert_dir = tmp / "certs"
        cert_dir.mkdir()
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", str(cert_dir / "privkey.pem"),
            "-out", str(cert_dir / "fullchain.pem"),
            "-subj", "/CN=swavlamban.local",
            "-addext", "subjectAltName=DNS:swavlamban.local",
        ], check=True, capture_output=True)

        upstream = ThreadingHTTPServer(("127.0.0.1", 19092), Handler)
        thread = Thread(target=upstream.serve_forever, daemon=True)
        thread.start()

        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(ROOT),
            "WAF_ENV": "development",
            "WAF_GATEWAY_HOST": "127.0.0.1",
            "WAF_GATEWAY_PORT": "18082",
            "WAF_UPSTREAM_URL": "http://127.0.0.1:19092",
            "WAF_RATE_LIMIT_PER_MINUTE": "1000",
        })
        gateway_log = tmp / "gateway.log"
        gateway = subprocess.Popen(["python", "-m", "waf.gateway.proxy"], cwd=str(ROOT), env=env, stdout=gateway_log.open("w"), stderr=subprocess.STDOUT)

        nginx_conf = tmp / "nginx.conf"
        nginx_conf.write_text(f"""events {{ worker_connections 64; }}
http {{
  server {{
    listen 18445 ssl;
    server_name swavlamban.local;
    ssl_certificate {cert_dir / 'fullchain.pem'};
    ssl_certificate_key {cert_dir / 'privkey.pem'};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_session_tickets off;
    add_header Strict-Transport-Security \"max-age=300\" always;
    location / {{
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto https;
      proxy_pass http://127.0.0.1:18082;
    }}
  }}
}}
""", encoding="utf-8")
        subprocess.run(["nginx", "-t", "-c", str(nginx_conf), "-p", str(tmp)], check=True, capture_output=True, text=True)
        nginx = subprocess.Popen(["nginx", "-c", str(nginx_conf), "-p", str(tmp), "-g", f"pid {tmp / 'nginx.pid'}; daemon off;"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_port("127.0.0.1", 19092)
            wait_port("127.0.0.1", 18082)
            wait_port("127.0.0.1", 18445)
            allow_code, allow_body = request("https://127.0.0.1:18445/health")
            # urllib does not trust self-signed certs by default, so use curl for the secure requests.
            allow = subprocess.run(["curl", "-sk", "-o", str(tmp / "allow.body"), "-w", "%{http_code}", "https://swavlamban.local:18445/health", "--resolve", "swavlamban.local:18445:127.0.0.1"], capture_output=True, text=True, check=True)
            allow_code = int(allow.stdout)
            allow_body = (tmp / "allow.body").read_bytes()
            before = Handler.hits
            block = subprocess.run(["curl", "-sk", "-o", str(tmp / "block.body"), "-w", "%{http_code}", "https://swavlamban.local:18445/search?q=%27%20OR%201%3D1--", "--resolve", "swavlamban.local:18445:127.0.0.1"], capture_output=True, text=True, check=True)
            block_code = int(block.stdout)
            block_body = (tmp / "block.body").read_bytes()
            after = Handler.hits
            summary = {
                "nginx_tls_config_test": "PASS",
                "https_allow_status": allow_code,
                "https_allow_reached_upstream": MARKER in allow_body,
                "https_sql_block_status": block_code,
                "https_block_reached_upstream": after != before,
                "certificate_scope": "local self-signed",
                "tls_termination_point": "nginx",
                "decrypted_request_path": "nginx -> Swavlamban WAF gateway -> protected upstream",
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            (ROOT / "phase10_tls_evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0 if allow_code == 200 and block_code == 403 and not summary["https_block_reached_upstream"] else 1
        finally:
            for proc in (nginx, gateway):
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=4)
                    except subprocess.TimeoutExpired: proc.kill()
            upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
