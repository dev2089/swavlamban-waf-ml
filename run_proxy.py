from aiohttp import web

from waf.core.config import WAFConfig
from waf.edge.reverse_proxy import build_app


if __name__ == '__main__':
    config = WAFConfig.from_env()
    web.run_app(build_app(config), host=config.listen_host, port=config.listen_port)
