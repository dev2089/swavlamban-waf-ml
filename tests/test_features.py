from __future__ import annotations

import unittest

from waf.core.models import RequestEnvelope
from waf.features.http import HTTPFeatureExtractor


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.extractor = HTTPFeatureExtractor()

    def req(self, **kwargs):
        data = dict(request_id="f-1", method="GET", scheme="https", host="example.test", path="/")
        data.update(kwargs)
        return RequestEnvelope(**data)

    def test_schema_and_normalized_ranges(self):
        fv = self.extractor.extract(self.req())
        self.assertEqual(fv.schema_version, "http-v2")
        self.assertGreaterEqual(len(fv.values), 30)
        for value in fv.values.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_known_payload_indicators(self):
        fv = self.extractor.extract(self.req(query="q=1 union select x from y", body=b"<script>alert(1)</script>"))
        self.assertEqual(fv.values["has_sql_keyword"], 1.0)
        self.assertEqual(fv.values["has_xss_token"], 1.0)

    def test_unicode_and_invalid_utf8_are_safe(self):
        fv = self.extractor.extract(self.req(body=b"hello\xff\xfe\xe2\x82\xac"))
        self.assertIn("body_length", fv.values)
        self.assertGreaterEqual(fv.values["body_utf8_replacement_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
