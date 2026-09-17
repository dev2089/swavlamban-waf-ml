#!/usr/bin/env python3
"""Controlled local HTTPS termination smoke test around the live WAF edge."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_port(host: str, port: int, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"port {port} did not open")


def curl(url: str) -> tuple[int, str]:
    proc = subprocess.run(["curl", "-sk", "-o", "/tmp/p9-curl-body", "-w", "%{http_code}", url], capture_output=True, text=True, check=True)
    return int(proc.stdout), Path("/tmp/p9-curl-body").read_text(encoding="utf-8")


def main() -> int:
    if not shutil.which("nginx") or not shutil.which("openssl") or not shutil.which("curl"):
        raise SystemExit("required local TLS tools not installed")
    with tempfile.TemporaryDirectory(prefix="waf-phase9-tls-") as td:
        tmp = Path(td)
        certs = tmp / "certs"
        certs.mkdir()
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", str(certs / "privkey.pem"), "-out", str(certs / "fullchain.pem"),
            "-subj", "/CN=waf.localhost", "-addext", "subjectAltName=DNS:waf.localhost",
        ], check=True, capture_output=True)
        conf = tmp / "nginx.conf"
        conf.write_text(f'''events {{ worker_connections 64; }}\nhttp {{\n  upstream waf_edge {{ server 127.0.0.1:18080; }}\n  server {{\n    listen 18443 ssl;\n    server_name waf.localhost;\n    ssl_certificate {certs / "fullchain.pem"};\n    ssl_certificate_key {certs / "privkey.pem"};\n    ssl_protocols TLSv1.2 TLSv1.3;\n    ssl_session_tickets off;\n    add_header Strict-Transport-Security "max-age=31536000" always;\n    location / {{\n      proxy_pass http://waf_edge;\n      proxy_set_header Host $host;\n      proxy_set_header X-Forwarded-Proto https;\n      proxy_set_header X-Forwarded-For $remote_addr;\n    }}\n  }}\n}}\n''', encoding="utf-8")
        subprocess.run(["nginx", "-t", "-c", str(conf), "-p", str(tmp)], check=True, capture_output=True, text=True)

        env = os.environ.copy()
        env.update({"PYTHONPATH": str(ROOT), "WAF_UPSTREAM_URL": "http://127.0.0.1:19000", "WAF_LISTEN_HOST": "127.0.0.1", "WAF_LISTEN_PORT": "18080"})
        upstream = subprocess.Popen(["python", "-m", "http.server", "19000", "--bind", "127.0.0.1", "--directory", str(tmp)], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        waf = subprocess.Popen(["python", "run_proxy.py"], cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        nginx = subprocess.Popen(["nginx", "-c", str(conf), "-p", str(tmp), "-g", f"pid {tmp / 'nginx.pid'}; daemon off;"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_port("127.0.0.1", 19000)
            wait_port("127.0.0.1", 18080)
            wait_port("127.0.0.1", 18443)
            allow_code, allow_body = curl("https://waf.localhost:18443/")
            block_code, block_body = curl("https://waf.localhost:18443/search?q=UNION%20SELECT%20password")
        finally:
            nginx.terminate(); waf.terminate(); upstream.terminate()
            for proc in (nginx, waf, upstream):
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill()

    result = {
        "phase": 9,
        "nginx_config_test": "PASS",
        "https_allow_request_status": allow_code,
        "https_block_request_status": block_code,
        "block_response_contains_decision_header": "blocked" in block_body.lower() or block_code == 403,
        "scope": "local self-signed TLS termination through nginx into the live WAF edge",
        "limitations": ["self-signed certificate", "localhost only", "does not prove external certificate management", "ModSecurity/Coraza not installed in this environment"],
    }
    out = ROOT / "phase9_tls_evidence.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if allow_code == 200 and block_code == 403 else 1


if __name__ == "__main__":
    raise SystemExit(main())
