from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from waf.core.config import WAFConfig


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        c = WAFConfig()
        self.assertEqual(c.pipeline_version, "phase3")
        self.assertEqual(c.feature_schema_version, "http-v2")
        self.assertLess(c.alert_threshold, c.block_threshold)

    def test_environment_validation(self):
        with patch.dict(os.environ, {"WAF_BLOCK_THRESHOLD": "1.2"}, clear=False):
            with self.assertRaises(ValueError):
                WAFConfig.from_env()
        with patch.dict(os.environ, {"WAF_MAX_BODY_BYTES": "0"}, clear=False):
            with self.assertRaises(ValueError):
                WAFConfig.from_env()


if __name__ == "__main__":
    unittest.main()
