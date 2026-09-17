#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP_LOG="$(mktemp)"
WAF_LOG="$(mktemp)"
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  rm -f "$UP_LOG" "$WAF_LOG" /tmp/swavlamban-waf-access.log /tmp/swavlamban-waf-error.log /tmp/waf-block-response
}
trap cleanup EXIT
python - <<'PY' >"$UP_LOG" 2>&1 &
from aiohttp import web
async def up(_request):
    return web.Response(text='protected-upstream')
app = web.Application()
app.router.add_route('*', '/{tail:.*}', up)
web.run_app(app, host='127.0.0.1', port=19000)
PY
PIDS+=("$!")
WAF_UPSTREAM_URL=http://127.0.0.1:19000 WAF_LISTEN_PORT=18080 python "$ROOT/run_proxy.py" >"$WAF_LOG" 2>&1 &
PIDS+=("$!")
sleep 0.6
nginx -t -c "$ROOT/nginx.phase2.conf" >/dev/null
nginx -c "$ROOT/nginx.phase2.conf" -g 'daemon off;' >/dev/null 2>&1 &
PIDS+=("$!")
sleep 0.6
allow_headers="$(curl -fsSI http://127.0.0.1:18088/health)"
grep -q 'HTTP/1.1 200' <<<"$allow_headers"
grep -q 'X-WAF-Decision: allow' <<<"$allow_headers"
code="$(curl -sS -o /tmp/waf-block-response -w '%{http_code}' 'http://127.0.0.1:18088/?q=%3Cscript%3Ealert(1)%3C/script%3E')"
[[ "$code" == "403" ]]
grep -q 'blocked' /tmp/waf-block-response
printf 'NGINX_INTEGRATION_PASS allow=200 block=403\n'
