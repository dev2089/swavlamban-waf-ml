#!/usr/bin/env python3
"""Canonical command-line entry point for the Swavlamban WAF edge."""
from __future__ import annotations

from aiohttp import web

from waf.core.config import WAFConfig
from waf.gateway.proxy import create_gateway_app


def main() -> None:
    config = WAFConfig.from_env()
    web.run_app(create_gateway_app(), host=config.listen_host, port=config.listen_port)


if __name__ == "__main__":
    main()
