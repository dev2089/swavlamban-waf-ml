import asyncio

from aiohttp import ClientSession, web

from waf.core.config import WAFConfig
from waf.core.models import Decision
from waf.edge.reverse_proxy import WAFReverseProxy


def test_decision_engine_blocks_sql():
    from waf.core.models import RequestEnvelope
    from waf.edge.pipeline import EdgeWAF

    result = EdgeWAF(WAFConfig()).analyze(RequestEnvelope('1', 'GET', 'http', 'x', '/u', 'id=1 OR 1=1'))
    assert result.decision is Decision.BLOCK
    assert 'WAF-SQL-001' in result.rule_ids


def test_decision_engine_allows_benign():
    from waf.core.models import RequestEnvelope
    from waf.edge.pipeline import EdgeWAF

    result = EdgeWAF(WAFConfig()).analyze(RequestEnvelope('2', 'GET', 'http', 'x', '/health'))
    assert result.decision is Decision.ALLOW


def test_encoded_xss_and_traversal_block():
    from waf.core.models import RequestEnvelope
    from waf.edge.pipeline import EdgeWAF

    waf = EdgeWAF(WAFConfig())
    assert waf.analyze(RequestEnvelope('3', 'GET', 'http', 'x', '/', 'q=%3Cscript%3Ealert(1)%3C/script%3E')).decision is Decision.BLOCK
    assert waf.analyze(RequestEnvelope('4', 'GET', 'http', 'x', '/../etc/passwd')).decision is Decision.BLOCK


def test_bounded_body_is_enforced():
    from waf.core.models import RequestEnvelope
    from waf.edge.pipeline import EdgeWAF

    cfg = WAFConfig(max_body_bytes=16)
    result = EdgeWAF(cfg).analyze(RequestEnvelope('5', 'POST', 'http', 'x', '/upload', body=b'A' * 64))
    assert result.request_id == '5'


def test_config_from_env_defaults_are_safe(monkeypatch):
    for name in (
        'WAF_PIPELINE_VERSION', 'WAF_BLOCK_THRESHOLD', 'WAF_ALERT_THRESHOLD',
        'WAF_MAX_BODY_BYTES', 'WAF_FEATURE_SCHEMA', 'WAF_UPSTREAM_URL',
        'WAF_LISTEN_HOST', 'WAF_LISTEN_PORT', 'WAF_REQUEST_TIMEOUT_SECONDS',
        'WAF_MAX_RESPONSE_BYTES',
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = WAFConfig.from_env()
    assert cfg.pipeline_version == 'phase3'
    assert cfg.feature_schema_version == 'http-v2'
    assert cfg.listen_port == 8080
    assert cfg.max_body_bytes == 1_048_576


def test_end_to_end_enforcement():
    async def go():
        hit = {'n': 0}

        async def up(req):
            hit['n'] += 1
            return web.Response(text='upstream-ok')

        ua = web.Application()
        ua.router.add_route('*', '/{tail:.*}', up)
        ur = web.AppRunner(ua)
        await ur.setup()
        us = web.TCPSite(ur, '127.0.0.1', 19100)
        await us.start()

        p = WAFReverseProxy(WAFConfig(upstream_url='http://127.0.0.1:19100', listen_port=18100))
        pr = web.AppRunner(p.app)
        await pr.setup()
        ps = web.TCPSite(pr, '127.0.0.1', 18100)
        await ps.start()

        try:
            async with ClientSession() as client:
                ok = await client.get('http://127.0.0.1:18100/ok')
                assert ok.status == 200
                assert await ok.text() == 'upstream-ok'
                assert ok.headers['X-WAF-Decision'] == 'allow'

                bad = await client.get('http://127.0.0.1:18100/?q=%3Cscript%3Ealert(1)%3C/script%3E')
                assert bad.status == 403
                assert bad.headers['X-WAF-Decision'] == 'block'
                assert hit['n'] == 1

                body_bad = await client.post('http://127.0.0.1:18100/login', data='x=1; id')
                assert body_bad.status == 403
                assert hit['n'] == 1
        finally:
            await p.close()
            await pr.cleanup()
            await ur.cleanup()

    asyncio.run(go())
