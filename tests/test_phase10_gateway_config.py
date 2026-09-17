from waf.gateway.proxy import GatewayConfig


def test_gateway_config_from_env_uses_dataclass_defaults_with_slots():
    config = GatewayConfig.from_env({})
    assert config.upstream_url == "http://127.0.0.1:9000"
    assert config.listen_host == "127.0.0.1"
    assert config.listen_port == 18081
    assert config.request_timeout_seconds == 10.0
    assert config.max_body_bytes == 1_048_576
    assert config.max_response_bytes == 10_485_760
    assert config.rate_limit_per_minute == 120


def test_gateway_config_from_env_accepts_overrides():
    config = GatewayConfig.from_env({
        "WAF_UPSTREAM_URL": "https://upstream.example",
        "WAF_GATEWAY_HOST": "0.0.0.0",
        "WAF_GATEWAY_PORT": "18181",
        "WAF_REQUEST_TIMEOUT_SECONDS": "4.5",
        "WAF_MAX_BODY_BYTES": "2048",
        "WAF_MAX_RESPONSE_BYTES": "8192",
        "WAF_RATE_LIMIT_PER_MINUTE": "200",
    })
    assert config.upstream_url == "https://upstream.example"
    assert config.listen_host == "0.0.0.0"
    assert config.listen_port == 18181
    assert config.request_timeout_seconds == 4.5
    assert config.max_body_bytes == 2048
    assert config.max_response_bytes == 8192
    assert config.rate_limit_per_minute == 200
