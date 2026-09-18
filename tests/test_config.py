from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from waf.core.config import WAFConfig


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        c = WAFConfig()
        self.assertEqual(c.pipeline_version, "phase2")
        self.assertLess(c.alert_threshold, c.block_threshold)
        self.assertGreater(c.max_body_bytes, 0)
        self.assertGreater(c.max_response_bytes, 0)

    def test_environment_validation(self):
        with patch.dict(os.environ, {"WAF_BLOCK_THRESHOLD": "1.2"}, clear=False):
            with self.assertRaises(ValueError):
                WAFConfig.from_env()
        with patch.dict(os.environ, {"WAF_MAX_BODY_BYTES": "0"}, clear=False):
            with self.assertRaises(ValueError):
                WAFConfig.from_env()
        with patch.dict(os.environ, {"WAF_LISTEN_PORT": "70000"}, clear=False):
            with self.assertRaises(ValueError):
                WAFConfig.from_env()
        with patch.dict(os.environ, {"WAF_REQUEST_TIMEOUT_SECONDS": "0"}, clear=False):
            with self.assertRaises(ValueError):
                WAFConfig.from_env()


if __name__ == "__main__":
    unittest.main()
